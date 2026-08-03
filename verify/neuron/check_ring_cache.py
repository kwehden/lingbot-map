"""Desk-side verification of C1 (``NeuronRingKVCacheBackend``) against the GPU cache manager.

spec/neuron-port/design.md V3: the port's visible KV region must match
``FlashInferKVCacheManager``'s element for element, across append / evict / rollback, through
ring wrap.

This runs the **real** ``flashinfer_cache.FlashInferKVCacheManager`` as the reference, not a
reimplementation of it. FlashInfer itself is not installed in the desk container and is not
needed: the only use of the ``flashinfer`` module in the class is the
``BatchPrefillWithPagedKVCacheWrapper`` constructor, so a stub for that lets the genuine
page-allocation, eviction, rollback and page-table code run unmodified. Comparing against a
hand-written model of the GPU cache would only prove I can restate my own assumptions.

Two of these checks exist because they caught real bugs in the first draft of C1:

  * ``fixed_shapes_across_all_frames`` — the naive reading of design.md's ``visible_kv``
    contract gathers only the live slots, whose count changes every frame for the first 72
    frames. That is a new NEFF per frame, i.e. the exact failure the port exists to avoid.
  * ``naive_prefix_mask_breaks_after_wrap`` — the ring's live window is circular, so a prefix
    bound over the *physical* buffer is wrong once the ring wraps, silently. The gather
    resolves the wrap and is what makes prefix masking legitimate.

Run:
    docker exec lingbot-tier1 python \
        /home/kwehden/lingbot-tier1/lingbot-map/verify/neuron/check_ring_cache.py
"""

from __future__ import annotations

import json
import os
import sys
import types

import torch
import torch.nn.functional as F

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)

# --- Stub just enough of `flashinfer` to construct the real manager ------------------
_fi = types.ModuleType("flashinfer")


class _StubWrapper:
    """Stands in for BatchPrefillWithPagedKVCacheWrapper.

    Only the constructor is reached: this harness never calls ``compute_attention`` on the
    GPU manager, because V3 is about which keys are visible and in what order. The attention
    numerics are checked separately against ``F.scaled_dot_product_attention``.
    """

    def __init__(self, *a, **kw):
        pass

    def plan(self, *a, **kw):
        raise AssertionError("V3 must not invoke the FlashInfer kernel")

    def run(self, *a, **kw):
        raise AssertionError("V3 must not invoke the FlashInfer kernel")


_fi.BatchPrefillWithPagedKVCacheWrapper = _StubWrapper
sys.modules.setdefault("flashinfer", _fi)

from lingbot_map.layers import flashinfer_cache as fic  # noqa: E402

fic.FLASHINFER_AVAILABLE = True
fic.flashinfer = _fi

from lingbot_map.layers.flashinfer_cache import FlashInferKVCacheManager  # noqa: E402
from lingbot_map.layers.neuron_attention import merge_online_softmax  # noqa: E402
from lingbot_map.layers.neuron_kv_cache import (  # noqa: E402
    NeuronRingKVCacheBackend,
    plan_tiles,
)

RESULTS: list[dict] = []


def emit(name: str, ok: bool, **kw) -> None:
    RESULTS.append({"check": name, "pass": bool(ok), **kw})
    extra = "  ".join(f"{k}={v}" for k, v in kw.items())
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {extra}", flush=True)


# ====================================================================================
# Reference: expand the GPU manager's page table into its visible token sequence.
# ====================================================================================
def gpu_visible_tokens(mgr: FlashInferKVCacheManager, block_idx: int):
    """Return (K, V) as [n_live, H, D], in the GPU's visible order.

    Reconstructed exactly as FlashInfer would consume it: page IDs from
    ``build_visible_page_table()`` (scale -> window -> special), every page contributing its
    valid tokens, with the final page bounded by ``compute_last_page_len()``.
    """
    table = mgr.build_visible_page_table(block_idx)
    last_len = mgr.compute_last_page_len(block_idx)
    cache = mgr.kv_caches[block_idx]
    n_patch_pages = len(mgr.scale_patch_pages[block_idx]) + len(
        mgr.live_window_patch_pages[block_idx]
    )

    ks, vs = [], []
    for pos, page_id in enumerate(table):
        is_last = pos == len(table) - 1
        if pos < n_patch_pages:
            n = last_len if is_last else mgr.patches_per_frame
        else:
            n = last_len if is_last else mgr.page_size
        ks.append(cache[page_id, 0, :n])
        vs.append(cache[page_id, 1, :n])
    if not ks:
        H, D = mgr.num_heads, mgr.head_dim
        z = torch.zeros(0, H, D, dtype=mgr.dtype, device=mgr.device)
        return z, z
    return torch.cat(ks, 0), torch.cat(vs, 0)


def sdpa(q, k, v):
    """[Sq,H,D] x [Sk,H,D] -> [Sq,H,D], no mask (design.md F3)."""
    qq = q.permute(1, 0, 2).unsqueeze(0).float()
    kk = k.permute(1, 0, 2).unsqueeze(0).float()
    vv = v.permute(1, 0, 2).unsqueeze(0).float()
    o = F.scaled_dot_product_attention(qq, kk, vv, scale=q.shape[-1] ** -0.5)
    return o.squeeze(0).permute(1, 0, 2)


def rel_err(a, b) -> float:
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-12)).item()


# ====================================================================================
def main() -> int:
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emit("device", True, device=str(dev),
         gpu=(torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"))

    # Scaled-down geometry: same STRUCTURE as production (specials-prefix, scale quota,
    # sliding window, headroom), small enough to force many ring wraps quickly.
    CFG = dict(
        num_blocks=2, tokens_per_frame=16, num_heads=4, head_dim=8,
        num_special_tokens=6, scale_frames=3, sliding_window=5, max_total_frames=80,
    )
    P = CFG["tokens_per_frame"] - CFG["num_special_tokens"]      # 10 patches/frame
    HEADROOM = 4
    dtype = torch.float32

    gpu = FlashInferKVCacheManager(
        # max_num_frames is in the GPU manager's signature but unread in its body; the pool
        # is sized from scale_frames + sliding_window + 16. Passed for parity.
        max_num_frames=CFG["scale_frames"] + CFG["sliding_window"] + HEADROOM,
        dtype=dtype, device=dev, force_fp32=True, fa3=False,
        **{k: v for k, v in CFG.items()},
    )
    ring = NeuronRingKVCacheBackend(
        dtype=dtype, device=dev, headroom=HEADROOM,
        **{k: v for k, v in CFG.items()},
    )
    emit("page_size_is_exact", gpu.page_size == P, page_size=gpu.page_size, P=P)
    emit("win_capacity", ring.win_capacity == CFG["sliding_window"] + HEADROOM,
         got=ring.win_capacity)

    # ---- 1. lockstep replay: append / evict / rollback, well past ring wrap ---------
    # A deterministic keyframe pattern that exercises both branches of block.py's
    # skip_append path: keep most frames, discard some (rollback), through many wraps.
    N_FRAMES = 60
    KEEP = [not (i % 7 == 5) for i in range(N_FRAMES)]   # ~1 in 7 discarded

    order_mismatch = None
    shape_set = set()
    idx_len_set = set()
    n_rollback = n_kept = n_wraps = 0
    prev_head = 0
    stats_mismatch = None

    for i in range(N_FRAMES):
        k = torch.randn(CFG["tokens_per_frame"], CFG["num_heads"], CFG["head_dim"],
                        dtype=dtype, device=dev)
        v = torch.randn_like(k)

        # block.py:243-278's speculative order: defer -> append -> attend -> commit/discard.
        for mgr in (gpu, ring):
            mgr.set_defer_eviction(True)
        for b in range(CFG["num_blocks"]):
            gpu.append_frame(b, k, v)
            ring.append_frame(b, k, v)

        # Compare the visible region while the speculative frame is still in place — this is
        # the state attention actually sees.
        for b in range(CFG["num_blocks"]):
            gk, gv = gpu_visible_tokens(gpu, b)
            rk, rv, rvalid = ring.visible_kv_concat(b)
            if order_mismatch is None:
                if gk.shape != rk.shape:
                    order_mismatch = f"frame {i} block {b}: shape {tuple(gk.shape)} vs {tuple(rk.shape)}"
                elif not torch.equal(gk, rk) or not torch.equal(gv, rv):
                    nbad = int((gk != rk).any(dim=(1, 2)).sum().item())
                    order_mismatch = f"frame {i} block {b}: {nbad}/{gk.shape[0]} tokens differ"
                elif int(rvalid.item()) != gk.shape[0]:
                    order_mismatch = f"frame {i} block {b}: valid_len {int(rvalid.item())} != {gk.shape[0]}"

            # Fixed-shape evidence: the staging buffers and the gather index must never
            # change length, at any frame, in either warmup or steady state.
            sk_, sv_, _ = ring.visible_patch_kv(b)
            shape_set.add((tuple(sk_.shape), tuple(sv_.shape)))
            idx_len_set.add(int(ring.visible_index(b).numel()))

            g = gpu.get_cache_stats(b) if hasattr(gpu, "get_cache_stats") else None
            r = ring.get_cache_stats(b)
            if g is not None and stats_mismatch is None:
                for key in ("frame_count", "scale_pages", "live_pages", "special_tokens"):
                    if key in g and g[key] != r[key]:
                        stats_mismatch = f"frame {i} block {b}: {key} {g[key]} != {r[key]}"

        if KEEP[i]:
            gpu.execute_deferred_eviction_all_blocks(CFG["scale_frames"], CFG["sliding_window"])
            ring.execute_deferred_eviction_all_blocks(CFG["scale_frames"], CFG["sliding_window"])
            n_kept += 1
        else:
            gpu.rollback_last_frame_all_blocks()
            ring.rollback_last_frame_all_blocks()
            n_rollback += 1
        for mgr in (gpu, ring):
            mgr.set_defer_eviction(False)

        if ring.win_head[0] < prev_head:
            n_wraps += 1
        prev_head = ring.win_head[0]

    emit("visible_kv_matches_flashinfer_exactly", order_mismatch is None,
         frames=N_FRAMES, kept=n_kept, rolled_back=n_rollback, ring_wraps=n_wraps,
         detail=order_mismatch or "element-for-element identical, K and V, both blocks")
    emit("ring_actually_wrapped", n_wraps >= 2, wraps=n_wraps,
         detail="if 0, the wrap-correctness checks below are vacuous")
    emit("rollback_exercised", n_rollback > 0, n=n_rollback)
    emit("fixed_shapes_across_all_frames",
         len(shape_set) == 1 and len(idx_len_set) == 1,
         staging_shapes=len(shape_set), index_lengths=sorted(idx_len_set),
         detail="one shape and one index length for every frame -> one NEFF")
    emit("cache_stats_agree", stats_mismatch is None,
         detail=stats_mismatch or "frame_count/scale_pages/live_pages/special_tokens match")

    # ---- 2. the wrap bug: prove the gather is load-bearing ------------------------
    # Naive alternative: mask a prefix of the PHYSICAL patch buffer instead of gathering.
    # Correct before the ring wraps, wrong after — which is why it needs a test, not a
    # reading.
    b = 0
    idx = ring.visible_index(b)
    n_live_frames = ring._n_visible_frames(b)
    live_slots = idx[:n_live_frames].tolist()
    physical_prefix = list(range(n_live_frames))
    gk, _ = gpu_visible_tokens(gpu, b)
    naive_k = ring.patch_k[b][:n_live_frames].reshape(-1, CFG["num_heads"], CFG["head_dim"])
    naive_matches = naive_k.shape == gk[: naive_k.shape[0]].shape and torch.equal(
        naive_k, gk[: naive_k.shape[0]]
    )
    emit("naive_prefix_mask_breaks_after_wrap", not naive_matches,
         live_slots=str(live_slots), physical_prefix=str(physical_prefix),
         detail="post-wrap the live window is circular, so a physical prefix is the WRONG "
                "key set; if this PASSES as a match, the gather was unnecessary")

    # ---- 3. attention numerics: C1 groups + C2 merge == SDPA over the GPU's keys ----
    # Uses the same cte_stub contract as check_tile_merge.py: this is the composed form
    # (two groups, several tiles each, partial tails, empty tiles) driven by REAL cache
    # state at a post-wrap frame.
    def cte_stub(q_k, k_t, v_t, used: int):
        H, Sq, D = q_k.shape
        Sk = k_t.shape[1]
        logits = torch.einsum("hqd,hkd->hqk", q_k.float(), k_t.float())
        if used < Sk:
            cols = torch.arange(Sk, device=q_k.device)
            logits = logits.masked_fill((cols >= used).view(1, 1, Sk), float("-inf"))
        if used == 0:
            return (torch.zeros(H, Sq, D, device=q_k.device),
                    torch.zeros(H, Sq, 1, device=q_k.device),
                    torch.zeros(H, Sq, 1, device=q_k.device))
        m = logits.max(-1, keepdim=True).values
        p = torch.exp(logits - m)
        l = p.sum(-1, keepdim=True)
        return torch.einsum("hqk,hkd->hqd", p, v_t.float()) / l, -m, 1.0 / l

    q = torch.randn(CFG["tokens_per_frame"], CFG["num_heads"], CFG["head_dim"],
                    dtype=dtype, device=dev)
    scale = CFG["head_dim"] ** -0.5
    q_k = (q * scale).permute(1, 0, 2).contiguous()

    parts, n_tiles_total, n_empty = [], 0, 0
    for gk_, gv_, gvalid, plan in ring.visible_groups(b):
        remaining = int(gvalid.item())
        for t in range(plan.n_tiles):
            sl = slice(t * plan.tile, (t + 1) * plan.tile)
            used = max(0, min(plan.tile, remaining - t * plan.tile))
            o, nm, sr = cte_stub(q_k, gk_[sl].permute(1, 0, 2), gv_[sl].permute(1, 0, 2), used)
            parts.append((o, nm, sr, torch.tensor(used, device=dev)))
            n_tiles_total += 1
            n_empty += int(used == 0)
    got = merge_online_softmax(parts).permute(1, 0, 2)
    ref = sdpa(q, *gpu_visible_tokens(gpu, b))
    err = rel_err(got, ref)
    emit("grouped_attention_matches_sdpa_over_gpu_keys", err < 1e-4,
         rel_err=round(err, 8), n_tiles=n_tiles_total, n_empty_tiles=n_empty,
         n_keys=int(gpu_visible_tokens(gpu, b)[0].shape[0]))

    # ---- 4. production geometry: shapes, tile plan, and what it costs ---------------
    prod = NeuronRingKVCacheBackend(
        num_blocks=1, tokens_per_frame=1375, num_heads=1, head_dim=1,
        dtype=torch.bfloat16, device=dev, num_special_tokens=6, scale_frames=8,
        sliding_window=64, max_total_frames=1124, headroom=16,
    )
    rep = prod.tiling_report()
    emit("prod_live_keys_match_design_D2", prod.max_live_keys == 105_312,
         got=prod.max_live_keys)
    emit("prod_all_tiles_under_kernel_ceiling",
         prod.patch_plan.tile <= 36864 and prod.special_plan.tile <= 36864,
         patch_tile=prod.patch_plan.tile, special_tile=prod.special_plan.tile)
    emit("prod_tiles_are_equal_length_per_group",
         prod.patch_plan.padded == prod.patch_plan.n_tiles * prod.patch_plan.tile,
         patch=f"{prod.patch_plan.n_tiles}x{prod.patch_plan.tile}",
         special=f"{prod.special_plan.n_tiles}x{prod.special_plan.tile}",
         detail="equal tiles per group -> one NEFF per group (design.md D1a)")
    emit("prod_traversal_overhead_reported", True,
         traversed=rep["traversed_keys"], live_max=rep["live_max_keys"],
         overhead_pct=rep["overhead_pct"], n_neff_shapes=rep["n_neff_shapes"],
         detail="cost of fixed shapes + headroom; the perf-vs-GPU number to beat")

    # ---- 4b. headroom sweep: what the traversal overhead actually buys --------------
    # headroom inflates num_patch_slots, which inflates the patch group's tile plan, which
    # the kernel walks EVERY frame whether those slots are live or not. The GPU inherits
    # +16 from its page pool (flashinfer_cache.py:133), but the ring's requirement is
    # different: it only needs room for the speculative frame that deferred eviction leaves
    # in place, i.e. sliding_window + 1. Anything beyond that is pure traversal cost.
    #
    # This measures rather than argues: replay the same 60-frame keep/discard pattern at each
    # headroom and record whether the ring overflows.
    sweep = []
    for hr in (1, 2, 4, 8, 16):
        probe = NeuronRingKVCacheBackend(
            dtype=dtype, device=dev, headroom=hr,
            **{k: v for k, v in CFG.items()},
        )
        kk_ = torch.zeros(CFG["tokens_per_frame"], CFG["num_heads"], CFG["head_dim"],
                          dtype=dtype, device=dev)
        overflow = None
        try:
            for i in range(N_FRAMES):
                probe.set_defer_eviction(True)
                for bb in range(CFG["num_blocks"]):
                    probe.append_frame(bb, kk_, kk_)
                if KEEP[i]:
                    probe.execute_deferred_eviction_all_blocks(
                        CFG["scale_frames"], CFG["sliding_window"])
                else:
                    probe.rollback_last_frame_all_blocks()
                probe.set_defer_eviction(False)
        except RuntimeError as e:
            overflow = f"frame {i}: {e}"

        big = NeuronRingKVCacheBackend(
            num_blocks=1, tokens_per_frame=1375, num_heads=1, head_dim=1,
            dtype=torch.bfloat16, device=dev, num_special_tokens=6, scale_frames=8,
            sliding_window=64, max_total_frames=1124, headroom=hr,
        )
        r = big.tiling_report()
        sweep.append({
            "headroom": hr, "survived_replay": overflow is None,
            "overflow": overflow,
            "prod_traversed": r["traversed_keys"],
            "prod_overhead_pct": r["overhead_pct"],
            "prod_patch_tiles": f"{big.patch_plan.n_tiles}x{big.patch_plan.tile}",
            "prod_gather_MiB_per_block_per_frame": round(
                big.num_patch_slots * big.patches_per_frame * 16 * 64 * 2 * 2 / 2 ** 20, 1),
        })
    for s in sweep:
        print(f"        headroom={s['headroom']:>2}  survived={s['survived_replay']}  "
              f"traversed={s['prod_traversed']}  ovh={s['prod_overhead_pct']:>5.2f}%  "
              f"tiles={s['prod_patch_tiles']}", flush=True)
    min_ok = min((s["headroom"] for s in sweep if s["survived_replay"]), default=None)
    emit("headroom_2_suffices_for_deferred_eviction",
         min_ok is not None and min_ok <= 2,
         min_surviving_headroom=min_ok,
         detail="deferred eviction leaves at most sliding_window+1 frames, so capacity "
                "beyond +1 is unused; the GPU's +16 is a page-pool inheritance, not a "
                "requirement. Overhead 21.18% -> 2.70% at headroom=2")

    # Every check above passes headroom explicitly, so none of them would catch a bad
    # DEFAULT. Pin it: the shipped default must survive the same replay and must be the
    # cheap one, not the GPU's inherited 16.
    dflt = NeuronRingKVCacheBackend(
        dtype=dtype, device=dev, **{k: v for k, v in CFG.items()}
    )
    dflt_survived = None
    kk_ = torch.zeros(CFG["tokens_per_frame"], CFG["num_heads"], CFG["head_dim"],
                      dtype=dtype, device=dev)
    try:
        for i in range(N_FRAMES):
            dflt.set_defer_eviction(True)
            for bb in range(CFG["num_blocks"]):
                dflt.append_frame(bb, kk_, kk_)
            if KEEP[i]:
                dflt.execute_deferred_eviction_all_blocks(
                    CFG["scale_frames"], CFG["sliding_window"])
            else:
                dflt.rollback_last_frame_all_blocks()
            dflt.set_defer_eviction(False)
        dflt_survived = True
    except RuntimeError as e:
        dflt_survived = f"{e}"
    prod_dflt = NeuronRingKVCacheBackend(
        num_blocks=1, tokens_per_frame=1375, num_heads=1, head_dim=1,
        dtype=torch.bfloat16, device=dev, num_special_tokens=6, scale_frames=8,
        sliding_window=64, max_total_frames=1124,
    ).tiling_report()
    emit("shipped_default_headroom_survives_and_is_cheap",
         dflt_survived is True and dflt.headroom <= 2
         and prod_dflt["overhead_pct"] < 5.0,
         default_headroom=dflt.headroom, survived=dflt_survived,
         prod_traversed=prod_dflt["traversed_keys"],
         prod_overhead_pct=prod_dflt["overhead_pct"],
         prod_patch_tiles=f"{prod_dflt['patch_plan']['n_tiles']}"
                          f"x{prod_dflt['patch_plan']['tile']}")

    # ---- 5. special-stream capacity: measure both, do not assume they match ---------
    gpu_special_slots = (gpu.max_num_pages - gpu.max_patch_pages) * gpu.page_size
    ring_special_slots = ring.special_capacity
    gpu_frames = gpu_special_slots // CFG["num_special_tokens"]
    ring_frames = ring_special_slots // CFG["num_special_tokens"]
    emit("special_capacity_divergence_is_ring_stricter",
         ring_frames <= gpu_frames,
         gpu_frame_limit=gpu_frames, ring_frame_limit=ring_frames,
         configured_max_total_frames=CFG["max_total_frames"],
         detail="the GPU over-allocates specials by 16 whole pages, so it raises LATER "
                "than max_total_frames; the ring raises exactly AT it. Both honor the "
                "configured limit; the ring is strictly stricter. NOT the bit-identical "
                "matched failure design.md FM4 assumes -- design.md needs amending")

    # ---- 6. exhaustion actually raises, and names the right knob --------------------
    small = NeuronRingKVCacheBackend(
        num_blocks=1, tokens_per_frame=16, num_heads=1, head_dim=1, dtype=dtype,
        device=dev, num_special_tokens=6, scale_frames=1, sliding_window=2,
        max_total_frames=3, headroom=1,
    )
    kk = torch.randn(16, 1, 1, dtype=dtype, device=dev)
    raised = ""
    try:
        for _ in range(10):
            small.append_frame(0, kk, kk)
            small.evict_frames(0, 1, 2)
    except AssertionError as e:
        raised = str(e)
    emit("special_exhaustion_raises_naming_max_total_frames",
         "max_total_frames" in raised and small.frame_count[0] == 3,
         at_frame=small.frame_count[0], msg=raised[:80] or "(did not raise)")

    # ---- 7. reset returns to a pristine state --------------------------------------
    ring.reset()
    emit("reset_clears_all_state",
         all(ring.frame_count[i] == 0 and ring.win_len[i] == 0 and ring.win_head[i] == 0
             and ring.special_token_count[i] == 0 for i in range(CFG["num_blocks"]))
         and ring.visible_len(0) == 0,
         detail="buffers intentionally NOT reallocated")

    n_fail = sum(1 for r in RESULTS if not r["pass"])
    out = {
        "suite": "neuron C1 ring-cache vs FlashInfer (design.md V3)",
        "device": str(dev),
        "reference": "REAL flashinfer_cache.FlashInferKVCacheManager (wrapper stubbed)",
        "n_checks": len(RESULTS), "n_fail": n_fail,
        "verdict": "PASS" if n_fail == 0 else "FAIL",
        "prod_tiling_report": rep,
        "results": RESULTS,
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ring_cache_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n=== {out['verdict']}: {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    print(f"=== wrote {path}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
