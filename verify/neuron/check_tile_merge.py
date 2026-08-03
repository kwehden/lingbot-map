"""Desk-side numerical check of the ODQ7 composed form: KV tiling + online-softmax merge
+ padded-tail masking, verified against full SDPA.

spec/neuron-port/design.md ODQ7 is the leading residual risk after V1:

    "V1 verified the combine (uniform tiles) and the chaining (single call) SEPARATELY,
     never together. Phase 2's first check must be that composed form."

This script is that check, run WITHOUT Neuron hardware. It cannot validate `attention_cte`
itself (that needs trn2), but it validates the thing that is actually novel and the thing
most likely to be wrong: whether merging per-tile online-softmax statistics — with a
partially-valid final tile and, crucially, with ENTIRELY EMPTY tiles — reproduces a single
full-length attention.

The empty-tile case is not in design.md and is why this script exists in this form. At
production defaults the attended length is
    patches_per_frame * (n_scale + n_window) + num_special * N
which is 11_000 right after the scale prefill and only crosses the 36_864 ceiling at frame
27 and the second ceiling at frame 54. So for the first 26 frames of EVERY run, two of the
three tiles are pure padding. An unmasked all-zero-key tile softmaxes over 36_864 equal
logits and returns a large, confident-looking denominator that then dominates the merge.

Run:
    docker exec lingbot-tier1 python /home/kwehden/lingbot-tier1/lingbot-map/verify/neuron/check_tile_merge.py
"""

from __future__ import annotations

import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from lingbot_map.layers.neuron_attention import (  # noqa: E402
    MAX_SEQLEN,
    merge_online_softmax,
    num_tiles,
    padded_seqlen_k,
    tile_valid_lens,
)

RESULTS: list[dict] = []


def emit(name: str, ok: bool, **kw) -> None:
    rec = {"check": name, "pass": bool(ok), **kw}
    RESULTS.append(rec)
    flag = "PASS" if ok else "FAIL"
    extra = "  ".join(f"{k}={v}" for k, v in kw.items())
    print(f"[{flag}] {name}  {extra}", flush=True)


# ----------------------------------------------------------------------------------
# A pure-torch stand-in for attention_cte(cache_softmax=True).
#
# This models the KERNEL'S CONTRACT as V1 established it, not its implementation:
#   returns (output, out_neg_max, out_sum_recip) where
#       out_neg_max   = -(row max of logits)
#       out_sum_recip = 1 / (row sum of exp(logits - max))
# and `prior_used_len` bounds which keys participate (V1: bit-exact as a valid-length mask).
#
# If this stand-in's convention were wrong, the merge would still "pass" here and fail on
# hardware — so the convention is pinned by V1's measured result (convention A), and the
# discriminating power of the test is checked explicitly by `check_convention_b_fails`.
# ----------------------------------------------------------------------------------
def cte_stub(q, k, v, used_len: int, softmax_dtype=torch.float32):
    """q: [H, Sq, D]  k: [H, Sk, D]  v: [H, Sk, D] -> (out, neg_max, sum_recip)."""
    H, Sq, D = q.shape
    Sk = k.shape[1]
    logits = torch.einsum("hqd,hkd->hqk", q.to(softmax_dtype), k.to(softmax_dtype))
    if used_len < Sk:
        cols = torch.arange(Sk, device=q.device)
        logits = logits.masked_fill((cols >= used_len).view(1, 1, Sk), float("-inf"))
    if used_len == 0:
        # Fully masked: define the identity so the merge's neutralisation is what is
        # actually under test (a real kernel returns garbage here, which is the point).
        out = torch.zeros(H, Sq, D, dtype=q.dtype, device=q.device)
        neg_max = torch.full((H, Sq, 1), 0.0, dtype=softmax_dtype, device=q.device)
        sum_recip = torch.zeros(H, Sq, 1, dtype=softmax_dtype, device=q.device)
        return out, neg_max, sum_recip
    m = logits.max(dim=-1, keepdim=True).values
    p = torch.exp(logits - m)
    l = p.sum(dim=-1, keepdim=True)
    out = torch.einsum("hqk,hkd->hqd", p, v.to(softmax_dtype)) / l
    return out.to(q.dtype), (-m), (1.0 / l)


def cte_stub_unmasked_empty(q, k, v, used_len: int, softmax_dtype=torch.float32):
    """Same, but an empty tile is NOT special-cased — it softmaxes over zero-keys.

    This models what a real kernel does when handed an all-padding tile with no valid-length
    bound: uniform attention over Sk keys, giving l = Sk. Used to prove the merge's
    neutralisation is load-bearing rather than decorative.
    """
    H, Sq, D = q.shape
    Sk = k.shape[1]
    logits = torch.einsum("hqd,hkd->hqk", q.to(softmax_dtype), k.to(softmax_dtype))
    if 0 < used_len < Sk:
        cols = torch.arange(Sk, device=q.device)
        logits = logits.masked_fill((cols >= used_len).view(1, 1, Sk), float("-inf"))
    m = logits.max(dim=-1, keepdim=True).values
    p = torch.exp(logits - m)
    l = p.sum(dim=-1, keepdim=True)
    out = torch.einsum("hqk,hkd->hqd", p, v.to(softmax_dtype)) / l
    return out.to(q.dtype), (-m), (1.0 / l)


def sdpa_reference(q, k, v, valid_len: int):
    """Full-length reference: [H, Sq, D] with only the first `valid_len` keys visible."""
    q_b = q.unsqueeze(0).float()
    k_b = k[:, :valid_len].unsqueeze(0).float()
    v_b = v[:, :valid_len].unsqueeze(0).float()
    out = F.scaled_dot_product_attention(q_b, k_b, v_b, scale=1.0)
    return out.squeeze(0)


def rel_err(a, b) -> float:
    a, b = a.float(), b.float()
    denom = b.abs().max().clamp(min=1e-12)
    return ((a - b).abs().max() / denom).item()


def tiled_attend(q, k_pad, v_pad, valid_len: int, tile: int, stub=cte_stub):
    """The composed form: tile, per-tile stub call with a valid-length bound, then merge."""
    n_t = k_pad.shape[1] // tile
    vl = torch.tensor(valid_len, dtype=torch.int32, device=q.device)
    valids = tile_valid_lens(vl, n_t, tile)
    parts = []
    for i in range(n_t):
        sl = slice(i * tile, (i + 1) * tile)
        used = int(valids[i].item())   # host-side ONLY in this desk stub; the real adapter
                                       # passes the device tensor straight to the kernel
        o, nm, sr = stub(q, k_pad[:, sl], v_pad[:, sl], used)
        parts.append((o, nm, sr, valids[i]))
    return merge_online_softmax(parts)


# ==================================================================================
def main() -> int:
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    emit("device", True, device=dev,
         gpu=(torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"))

    # ---- 1. tiling arithmetic matches design.md D1a -------------------------------
    prod_seqlen_k = 105_312
    emit("D1a_tile_count", num_tiles(prod_seqlen_k) == 3, got=num_tiles(prod_seqlen_k))
    emit("D1a_padded_total", padded_seqlen_k(prod_seqlen_k) == 110_592,
         got=padded_seqlen_k(prod_seqlen_k))
    emit("D1a_tile_is_kernel_max", MAX_SEQLEN == 36_864, got=MAX_SEQLEN)

    # ---- 2. per-tile valid lengths across the whole frame range -------------------
    # This is the table that revealed the empty-tile hazard.
    P, SC, W, SP = 1369, 8, 64, 6
    schedule = []
    for N in (8, 12, 20, 26, 27, 53, 54, 72, 100, 1124):
        visible = P * (min(N, SC) + min(max(N - SC, 0), W)) + SP * N
        vl = torch.tensor(visible, dtype=torch.int32)
        lens = [int(t.item()) for t in tile_valid_lens(vl, 3, MAX_SEQLEN)]
        schedule.append({"frame": N, "visible": visible, "tile_valid": lens})
        assert sum(lens) == min(visible, 3 * MAX_SEQLEN), (N, visible, lens)
    n_empty_early = sum(1 for s in schedule if s["frame"] <= 26 and s["tile_valid"][1] == 0)
    emit("empty_tiles_occur_in_production", n_empty_early > 0,
         detail="tiles 1,2 are pure padding for frames <27 -- NOT in design.md",
         frame8=str(schedule[0]["tile_valid"]),
         frame1124=str(schedule[-1]["tile_valid"]))

    # ---- 3. THE CORE CHECK: composed form vs full SDPA ----------------------------
    # Small tile so the test is fast but the structure is identical to production:
    # 3 tiles, a partially-valid final tile, and cases where trailing tiles are EMPTY.
    H, Sq, D = 16, 128, 64
    TILE = 256
    N_T = 3
    padded = TILE * N_T

    for dtype in (torch.float32, torch.bfloat16):
        q = torch.randn(H, Sq, D, dtype=dtype, device=dev) * (D ** -0.5)
        k_pad = torch.zeros(H, padded, D, dtype=dtype, device=dev)
        v_pad = torch.zeros(H, padded, D, dtype=dtype, device=dev)

        for label, valid in (
            ("all_three_tiles_partial_tail", 700),   # [256,256,188]
            ("two_tiles_third_empty",        400),   # [256,144,  0]
            ("one_tile_two_empty",           200),   # [200,  0,  0]
            ("exactly_one_full_tile",        256),   # [256,  0,  0]
            ("exactly_two_full_tiles",       512),   # [256,256,  0]
            ("completely_full",              768),   # [256,256,256]
        ):
            k_pad.zero_(); v_pad.zero_()
            k_pad[:, :valid] = torch.randn(H, valid, D, dtype=dtype, device=dev)
            v_pad[:, :valid] = torch.randn(H, valid, D, dtype=dtype, device=dev)

            got = tiled_attend(q, k_pad, v_pad, valid, TILE)
            ref = sdpa_reference(q, k_pad, v_pad, valid)
            err = rel_err(got, ref)
            # bf16 noise floor measured on hardware in V1 was ~0.003; fp32 should be ~1e-6.
            tol = 0.02 if dtype == torch.bfloat16 else 1e-4
            emit(f"composed_{label}_{str(dtype).split('.')[-1]}", err < tol,
                 rel_err=round(err, 6), tol=tol, valid=valid,
                 finite=bool(torch.isfinite(got).all().item()))

    # ---- 4. discriminating power: the test must be able to FAIL --------------------
    # 4a. Convention B (max/sum instead of neg-max/recip-sum) must fail badly.
    q = torch.randn(H, Sq, D, dtype=torch.float32, device=dev) * (D ** -0.5)
    k_pad = torch.zeros(H, padded, D, dtype=torch.float32, device=dev)
    v_pad = torch.zeros(H, padded, D, dtype=torch.float32, device=dev)
    valid = 700
    k_pad[:, :valid] = torch.randn(H, valid, D, device=dev)
    v_pad[:, :valid] = torch.randn(H, valid, D, device=dev)
    ref = sdpa_reference(q, k_pad, v_pad, valid)

    def stub_conv_b(qq, kk, vv, used):
        o, nm, sr = cte_stub(qq, kk, vv, used)
        return o, -nm, torch.where(sr > 0, 1.0 / sr.clamp(min=1e-30), torch.zeros_like(sr))
    err_b = rel_err(tiled_attend(q, k_pad, v_pad, valid, TILE, stub=stub_conv_b), ref)
    emit("convention_B_fails_as_expected", err_b > 0.02, rel_err=round(err_b, 6),
         detail="if this PASSES, the composed test is vacuous")

    # 4b. Unmasked empty tiles must fail — proves the neutralisation is load-bearing.
    valid_e = 200   # [200, 0, 0]: two empty tiles
    k_pad.zero_(); v_pad.zero_()
    k_pad[:, :valid_e] = torch.randn(H, valid_e, D, device=dev)
    v_pad[:, :valid_e] = torch.randn(H, valid_e, D, device=dev)
    ref_e = sdpa_reference(q, k_pad, v_pad, valid_e)
    got_masked = tiled_attend(q, k_pad, v_pad, valid_e, TILE)
    err_masked = rel_err(got_masked, ref_e)

    # Same merge, but tiles are NOT neutralised and the kernel is not told the bound:
    # feed every tile its full length so empty tiles softmax over zero-keys.
    n_t = N_T
    parts_bad = []
    for i in range(n_t):
        sl = slice(i * TILE, (i + 1) * TILE)
        o, nm, sr = cte_stub_unmasked_empty(q, k_pad[:, sl], v_pad[:, sl], TILE)
        parts_bad.append((o, nm, sr, torch.tensor(TILE, device=dev)))  # claim all live
    err_unmasked = rel_err(merge_online_softmax(parts_bad), ref_e)
    emit("empty_tile_neutralisation_is_load_bearing",
         err_masked < 1e-4 and err_unmasked > 0.05,
         masked_rel_err=round(err_masked, 8), unmasked_rel_err=round(err_unmasked, 6),
         detail="unmasked empty tiles dominate the merge -- the hazard design.md missed")

    # ---- 5. merge is associative across tile counts (V1 saw 2x512 == 4x256) --------
    valid = 700
    k_pad.zero_(); v_pad.zero_()
    k_pad[:, :valid] = torch.randn(H, valid, D, device=dev)
    v_pad[:, :valid] = torch.randn(H, valid, D, device=dev)
    ref = sdpa_reference(q, k_pad, v_pad, valid)
    errs = {}
    for tile in (128, 192, 256, 384, 768):
        n_t = math.ceil(768 / tile)
        pad_to = n_t * tile
        kp = torch.zeros(H, pad_to, D, device=dev)
        vp = torch.zeros(H, pad_to, D, device=dev)
        kp[:, :valid] = k_pad[:, :valid]
        vp[:, :valid] = v_pad[:, :valid]
        errs[f"tile{tile}_n{n_t}"] = round(rel_err(tiled_attend(q, kp, vp, valid, tile), ref), 8)
    emit("merge_invariant_to_tile_count", all(e < 1e-4 for e in errs.values()), **errs)

    # ---- 6. no host sync in the merge ---------------------------------------------
    # tile_valid_lens must accept a 0-d device tensor and return device tensors, never
    # ints -- a Python int would bake the frame length into the NEFF (design.md C1).
    vl = torch.tensor(700, dtype=torch.int32, device=dev)
    lens = tile_valid_lens(vl, 3, TILE)
    emit("valid_lens_stay_on_device",
         all(torch.is_tensor(t) and t.device.type == vl.device.type for t in lens),
         types=str([type(t).__name__ for t in lens]))

    # ---- summary -------------------------------------------------------------------
    n_fail = sum(1 for r in RESULTS if not r["pass"])
    out = {
        "suite": "neuron ODQ7 composed-form desk check",
        "spec": "spec/neuron-port/design.md ODQ7 / D1a / C2",
        "device": dev,
        "n_checks": len(RESULTS),
        "n_fail": n_fail,
        "verdict": "PASS" if n_fail == 0 else "FAIL",
        "frame_schedule": schedule,
        "results": RESULTS,
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tile_merge_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n=== {out['verdict']}: {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    print(f"=== wrote {path}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
