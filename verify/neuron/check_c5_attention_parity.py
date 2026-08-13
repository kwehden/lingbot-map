"""C5 attention-output parity — the gate the 34-check suite does not have.

``check_c5_camera_cache.py`` verifies the cache CONTENTS (``append``/``visible_kv``) against the
real camera head, bit-exactly and correctly: a write is a copy, so ``torch.equal`` is the right
gate there. But it says nothing about what attention DOES with those contents. Established by the
2026-08-03 adversarial parity review and reproduced here:

* ``rel_err`` is defined in that suite (``:201``) with **zero call sites**.
* The single gate that calls ``compute_attention`` (``:990``) drives a probe whose
  ``attend_groups`` returns ``torch.zeros(...)`` — structurally incapable of disagreeing.
* So the only method that reaches the NEFF has no numerical gate at any tolerance.

This file closes that gap with two checks that can actually fail, and one that establishes the
tolerance policy:

1. ``attention_output_matches_gpu_sdpa`` — ``compute_attention`` through the REAL
   ``NeuronAttentionAdapter`` vs ``F.scaled_dot_product_attention`` over the identical visible
   keys, at production geometry. **Tolerance, not equality** — see below.
2. ``bit_equality_would_be_the_wrong_gate`` — the vacuity guard for check 1. Asserts the
   correct implementation is NOT bit-exact, so a future author cannot "tighten" check 1 to
   ``torch.equal`` and get a permanently-red gate. The tile/merge online-softmax reassociates an
   fp32 reduction; ~1e-6 is the floor, and that is arithmetic, not a defect.
3. ``prefix_alignment_is_gated`` — the missing alignment gate. The 34-check suite's docstring
   (``:332-335``) declines a key-ORDER injection as "genuinely harmless". That is only half
   right, and the wrong half is load-bearing: order IS invariant *within* ``[0, valid_len)``
   (verified here, ~1e-7), but ``valid_len`` bounds a PHYSICAL prefix, so any transform that
   moves live keys OUT of those rows attends padding zeros and drops real keys. Measured:
   whole-buffer ``flip(0)`` -> rel_err 1.0; ``roll(1, 0)`` with ``valid_len`` untouched -> ~0.3.
   This check asserts ``visible_group``'s live prefix equals the reference reshape, and the
   accompanying injections prove it discriminates.
4. ``tail_masking_is_load_bearing`` — the ODQ7 dependency, made a check. Re-runs with the
   stub ignoring ``prior_used_len`` and asserts check 1 fails. Records the magnitude, which
   is the uncomfortable part: measured **9.917e-01** at 8 frames but **1.420e-02** at 1124
   frames (``c5_attention_parity_results.json``), because the live fraction grows with
   depth. A broken tail mask presents as ~1% noise at production depth, so "the numbers
   look close" is not evidence here. The run's own ``shallow_rel_err``/``deep_rel_err``
   fields are authoritative; these two name their artifact rather than floating free of it.

A fifth mode, added 2026-08-13, measures rather than checks: ``--measure-reference-floor``
runs ``TASK-N08`` owed step 3 and writes its own artifact. It does **not** run the four checks
above and does not touch their artifact, because the 7/7 verdict is cited elsewhere and a
measurement is not a member of it. See the block above :func:`measure_reference_floor`.

Run::

    python3 verify/neuron/check_c5_attention_parity.py
    python3 verify/neuron/check_c5_attention_parity.py --inject axes_mixed
    python3 verify/neuron/check_c5_attention_parity.py --measure-reference-floor \
        --tensors-out /path/outside/this/repo

Exit codes. The parity suite uses ``0`` pass / ``1`` fail. ``--measure-reference-floor`` adds
two, because "measured" and "usable as a tolerance bar" are different answers:

===  ==========================================================================
 0   MEASURED -- every guard passed and ``reference_floor_c5`` may be consumed
 1   a guard failed; the floor is not established
 2   ESCALATE -- measured, but a one-ulp input move rivals the whole family
     spread, so the maximum is a finding and not a bar (``TASK-N08`` owed
     step 3's own words)
 3   BLOCKED -- no CUDA device; this is an A10G desk measurement
===  ==========================================================================

Nothing in ``attention.py``/``camera_head.py`` is edited; the GPU reference is only a valid
oracle while that stays true.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from lingbot_map.heads.neuron_camera_cache import NeuronCameraCache  # noqa: E402
from lingbot_map.layers.neuron_attention import (  # noqa: E402
    NeuronAttentionAdapter,
    merge_online_softmax,
)

RESULTS: list = []

# Geometry, in one place. The parity suite below and the reference-floor measurement further
# down must run at the SAME shape -- owed step 3's floor is only the floor of the comparison
# the suite makes -- and two copies of "16, 128, 1124" is exactly how commit 6f2e299's drift
# happened. Read, never reassigned; anything that assigns these shadows them by accident.
NUM_HEADS, HEAD_DIM, MTF = 16, 128, 1124


def rel_err(a, b) -> float:
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-12)).item()


def emit(check: str, ok: bool, **kw) -> None:
    RESULTS.append({"check": check, "pass": bool(ok), **kw})
    print(f"  [{'PASS' if ok else 'FAIL'}] {check}"
          + ("".join(f"\n           {k}={v}" for k, v in kw.items()) if kw else ""))


# ------------------------------------------------------------------------------------
# The kernel fake. Convention A, exactly as check_tile_merge.py pins it: attention over
# k_prior[:prior_used_len] ++ k_active, returning (out, neg_max, sum_recip) when
# cache_softmax=True. This is the contract V2 confirmed on trn2 hardware; faking it here
# keeps the check desk-runnable. The tail-masking half is what ODQ7's trn2 half must confirm.
# ------------------------------------------------------------------------------------
def install_cte_stub(honour_prior_used_len: bool = True):
    """Patch ``NeuronAttentionAdapter._load_kernel`` (the instance method the adapter calls).

    Layouts follow the real call site (``neuron_attention.py:368-374``), and they are
    asymmetric: ``q_k`` is ``[H, seqlen_q, D]`` seq-major, ``k_t`` is ``[H, D, tile]``
    **d-major**, ``v_t`` is ``[H, tile, D]`` seq-major, and ``scale=1.0`` because the adapter has
    already folded its own scale into q. ``k_prior``/``v_prior`` carry the whole tile with
    ``prior_used_len`` as the only thing selecting live keys, so the active pair is empty.
    """
    def stub(q_k, k_active, v_active, *, k_prior=None, v_prior=None, prior_used_len=None,
             cache_softmax=False, causal_mask=False, scale=1.0, **kwargs):
        if k_prior is not None:
            n = int(prior_used_len) if (honour_prior_used_len and prior_used_len is not None) \
                else k_prior.shape[-1]
            k = k_prior[:, :, :n].float()          # [H, D, n]
            v = v_prior[:, :n, :].float()          # [H, n, D]
        else:
            k = k_active.float()
            v = v_active.float()
        qf = q_k.float() * float(scale)            # [H, sq, D]
        logits = torch.einsum("hqd,hdk->hqk", qf, k)               # [H, sq, n]
        if logits.shape[-1] == 0:
            # All-padding tile. merge_online_softmax neutralises it on tile_valid == 0
            # (:151-156), so the values here only have to be finite and right-shaped.
            zeros = torch.zeros(*logits.shape[:2], qf.shape[-1], dtype=q_k.dtype,
                               device=q_k.device)
            f = torch.zeros(*logits.shape[:2], 1, device=q_k.device)
            return (zeros, f, f) if cache_softmax else zeros
        # Convention A, per merge_online_softmax's contract (:110-113): the NEGATED row max
        # and the RECIPROCAL row sum, both [..., seqlen_q, 1].
        neg_max = -logits.max(dim=-1, keepdim=True).values          # [H, sq, 1]
        e = torch.exp(logits + neg_max)
        s = e.sum(dim=-1, keepdim=True)                             # [H, sq, 1]
        out = torch.einsum("hqk,hkd->hqd", e, v) / s                # [H, sq, D]
        if cache_softmax:
            return out.to(q_k.dtype), neg_max, (1.0 / s)
        return out.to(q_k.dtype)

    NeuronAttentionAdapter._load_kernel = lambda self: stub        # noqa: SLF001
    return stub


def build(head_dim=128, num_heads=16, mtf=1124, device="cpu", adapter=True):
    ad = None
    if adapter:
        probe = NeuronCameraCache(num_iterations=1, trunk_depth=1, num_heads=num_heads,
                                  head_dim=head_dim, device=device, max_total_frames=mtf)
        ad = NeuronAttentionAdapter(head_dim=head_dim, max_seqlen_k=probe.plan.padded,
                                    tile=probe.plan.tile, check_env=False)
    return NeuronCameraCache(num_iterations=1, trunk_depth=1, num_heads=num_heads,
                             head_dim=head_dim, device=device, max_total_frames=mtf,
                             attention_adapter=ad)


def drive(c5, n_frames, num_heads, head_dim, device, gen):
    """Append n_frames keyframes; return the live k/v as [n, H, D]."""
    ks, vs = [], []
    for _ in range(n_frames):
        k = torch.randn(1, num_heads, 1, 1, head_dim, generator=gen, device=device)
        v = torch.randn(1, num_heads, 1, 1, head_dim, generator=gen, device=device)
        c5.append(0, 0, k, v)
        ks.append(k[0, :, 0, 0, :])          # [H, D]
        vs.append(v[0, :, 0, 0, :])
    return torch.stack(ks, 0), torch.stack(vs, 0)     # [n, H, D]


def gpu_sdpa(q_bhsd, k_nhd, v_nhd):
    """Reference: full attention over the live keys, the all-ones-mask case the camera head is in."""
    k = k_nhd.permute(1, 0, 2).unsqueeze(0)           # [1,H,n,D]
    v = v_nhd.permute(1, 0, 2).unsqueeze(0)
    return F.scaled_dot_product_attention(q_bhsd.float(), k.float(), v.float())


INJECTIONS = ("axes_mixed", "key_axis_reversed", "prefix_rolled", "contents_zeroed",
              "k_v_swapped", "in_prefix_permuted")


def apply_injection(c5, kind):
    """Corrupt visible_group only — the graph-facing method the 34-check suite never inspects."""
    real = c5.visible_group

    def bugged(i, j):
        k, v, valid, plan = real(i, j)
        n = int(valid)
        if kind == "axes_mixed":
            kb, vb, _ = c5.visible_kv(i, j)
            k = kb[0].reshape(plan.padded, c5.num_heads, c5.head_dim).contiguous()
            v = vb[0].reshape(plan.padded, c5.num_heads, c5.head_dim).contiguous()
        elif kind == "key_axis_reversed":
            k, v = k.flip(0).contiguous(), v.flip(0).contiguous()
        elif kind == "prefix_rolled":
            k, v = k.roll(1, 0).contiguous(), v.roll(1, 0).contiguous()
        elif kind == "contents_zeroed":
            k, v = torch.zeros_like(k), torch.zeros_like(v)
        elif kind == "k_v_swapped":
            k, v = v, k
        elif kind == "in_prefix_permuted":
            # The SAFE case: a permutation entirely inside [0, valid_len). Must NOT be caught
            # by the parity gate -- softmax is permutation-invariant over keys. This injection
            # exists to keep the alignment gate honest about what it does and does not claim.
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(7))
            k, v = k.clone(), v.clone()
            k[:n], v[:n] = k[:n][perm], v[:n][perm]
        return k, v, valid, plan

    c5.visible_group = bugged


# ------------------------------------------------------------------------------------------
# reference_floor(C5) -- TASK-N08 owed step 3 (added 2026-08-13)
#
# WHY IT LIVES HERE. TASK-N21's Phase 4 tolerance is max(reference_floor, 3 x noise_floor),
# and its reference_floor term was an analytic n*u forward-error bound. That bound does not
# cover the comparison it gates: it models one of the three reductions in this path, omits
# QK^T and the softmax entirely, bounds a summation relative to sum|x_i| rather than
# |sum x_i|, and carries an unbounded data-dependent amplification under cancellation. What
# covers it is a measurement, and the measurement is desk work in the harness that already
# owns this shape -- evaluate the SAME mathematics several legitimate ways and take the
# largest pairwise disagreement. That maximum is an empirical statement of how far two
# correct evaluations of this attention may sit apart here.
#
# WHAT IT IS NOT, AND THIS MATTERS. Every member runs on one device, and two of the four go
# through install_cte_stub(), so the family is GPU-vs-GPU and stops exactly one axis short of
# cross-device. It is not a Neuron figure and cannot become one here. A TASK-N14a discrepancy
# above the resulting gate is escalated per TASK-N21's step-5 rule and is never used to move
# this floor: the value being scored cannot re-derive the bar that scores it.
#
# WHY THE ULP PROBE SITS BESIDE IT. The family measures sensitivity to evaluation ORDER. If
# the shape is also sensitive to its INPUTS at the last bit, an order-derived maximum
# understates what two correct implementations can do, and the maximum is a finding rather
# than a bar. Owed step 3 left "large" unquantified; the comparison here is
# measured-against-measured and invents nothing -- escalate when a one-ulp input move shifts
# the output by at least as much as the entire family spread, because a floor that does not
# cover a one-ulp input change is not a bound on legitimate disagreement. No absolute cutoff
# appears anywhere below. A cutoff would let the cutoff, and not the measurement, decide the
# outcome, which is the defect that put this measurement here in the first place.
# ------------------------------------------------------------------------------------------
FLOOR_FRAMES = (8, 100, MTF)
FLOOR_SEQLEN_Q = (1, 5)
FLOOR_PARTS = 3          # the trunk's merge depth; this shape's own plan is 1 x 1152
FLOOR_SEED_BASE = 20260813


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _harness_revision(repo_root: str) -> dict:
    """Which revision produced this artifact -- or why that is unknown, never a plausible SHA.

    ``dirty`` is separate from ``commit`` on purpose: with an uncommitted edit in the tree the
    commit does not describe what ran, and the reader has to be told rather than handed a SHA
    that looks authoritative.

    **There is no git inside the Tier 1 container**, and every run of this harness happens
    there -- torch cannot be installed on the host at all. So self-observation is unavailable in
    the only environment that matters, and a provenance field that can only ever say
    "unavailable" is not provenance. ``LINGBOT_HARNESS_REV`` / ``_BRANCH`` / ``_DIRTY`` let the
    host wrapper, where git does exist, supply it.

    ``source`` is the load-bearing field. A supplied value is recorded as ``caller_asserted_env``
    and never as ``self_observed_git``: the harness did not verify it and cannot, so a reader
    weighing this floor has to be able to tell which of the two they are holding. Missing both is
    still not fatal -- losing a measurement over provenance would be the wrong trade.
    """
    def _git(*a):
        return subprocess.run(("git", "-C", repo_root) + a, capture_output=True, text=True,
                              timeout=20)
    try:
        head = _git("rev-parse", "HEAD")
        if head.returncode == 0:
            return {"source": "self_observed_git", "commit": head.stdout.strip(),
                    "branch": _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip(),
                    "subject": _git("log", "-1", "--pretty=%s").stdout.strip(),
                    "dirty": bool(_git("status", "--porcelain").stdout.strip())}
        why = (head.stderr or "git rev-parse failed").strip()[:200]
    except (OSError, subprocess.SubprocessError) as exc:
        why = f"{type(exc).__name__}: {exc}"[:200]

    asserted = os.environ.get("LINGBOT_HARNESS_REV", "").strip()
    if not asserted:
        return {"source": None, "commit": None,
                "unavailable_because": why,
                "remedy": "no git here and LINGBOT_HARNESS_REV unset; set it from the host, "
                          "where git exists, before docker exec"}
    dirty_env = os.environ.get("LINGBOT_HARNESS_DIRTY", "").strip()
    return {"source": "caller_asserted_env",
            "commit": asserted,
            "branch": os.environ.get("LINGBOT_HARNESS_BRANCH", "").strip() or None,
            "dirty": {"1": True, "0": False}.get(dirty_env),
            "self_observation_unavailable_because": why,
            "caveat": "supplied by the caller and NOT verified by this harness -- weaker "
                      "evidence than self_observed_git, and a caller that supplied the wrong "
                      "SHA would not be caught here"}


def _ulp_up(t):
    """One unit in the last place toward +inf -- the actual next representable float.

    Not ``t * (1 + eps)``: that is a scaled epsilon, which is a different perturbation and
    would make the amplification a function of the scaling I chose.
    """
    return torch.nextafter(t, torch.full_like(t, float("inf")))


def chunked_merge(q_bhsd, k_nhd, v_nhd, stub, n_parts):
    """The explicit path with the key reduction split, combined by the REAL merge.

    This is the one member that exercises ``merge_online_softmax`` as an actual reassociation.
    At this geometry ``plan_tiles(1124 + 1)`` is ``1 x 1152``, so the production adapter takes
    the single-part branch (``acc_m, acc_l, acc_o = m_i, l_i, o_i`` / ``continue``) and the
    merge is bit-exact pass-through -- contributing exactly zero. Splitting the key axis here
    makes it run, which is the reassociation the trunk performs and this shape does not, and
    is why the trunk's 3-tile-merge figure was never this shape's floor.
    """
    n = int(k_nhd.shape[0])
    n_parts = max(1, min(int(n_parts), n))
    edges = [(i * n) // n_parts for i in range(n_parts + 1)]
    # The adapter folds its own scale into q and calls the kernel with scale=1.0
    # (neuron_attention.py:302, :351, :418) -- so the same folding has to happen here, or this
    # member is not evaluating the same mathematics as the others.
    q_k = q_bhsd[0].float() * (HEAD_DIM ** -0.5)                    # [H, sq, D] seq-major
    parts, spans = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        k_t = k_nhd[lo:hi].permute(1, 2, 0).contiguous().float()     # [H, D, part] d-major
        v_t = v_nhd[lo:hi].permute(1, 0, 2).contiguous().float()     # [H, part, D] seq-major
        out_i, neg_max_i, sum_recip_i = stub(q_k, k_t, v_t, cache_softmax=True)
        parts.append((out_i, neg_max_i, sum_recip_i,
                      torch.tensor(hi - lo, device=out_i.device)))
        spans.append([lo, hi])
    return merge_online_softmax(parts).unsqueeze(0), spans


def _pairwise(members: dict) -> list:
    """Every unordered pair, by the harness's own ``rel_err``, symmetrised.

    ``rel_err`` normalises by its SECOND argument, so it is not symmetric. Taking the larger of
    the two directions keeps the reported disagreement independent of which member happens to
    be listed first; it is still ``rel_err`` and not some new metric introduced here.
    """
    names = sorted(members)
    return [{"pair": f"{a}|{b}",
             "rel_err": max(rel_err(members[a], members[b]), rel_err(members[b], members[a]))}
            for i, a in enumerate(names) for b in names[i + 1:]]


FLOOR_FAMILY = {
    "sdpa_fused": "F.scaled_dot_product_attention over the live keys -- the existing reference, "
                  "one fused kernel",
    "adapter_einsum": "compute_attention through the real NeuronAttentionAdapter: einsum-ordered "
                      "explicit softmax, scale folded into q, merge at one part. This member's "
                      "disagreement with sdpa_fused is the ~1.2e-6 TASK-N08 recorded in 2026-08-03 "
                      "-- which is what makes that figure ONE MEMBER of the family and not the floor",
    "chunked_merge": "the same explicit softmax with the key reduction split across FLOOR_PARTS "
                     "parts and recombined by the production merge_online_softmax",
    "in_prefix_permuted": "compute_attention with the live prefix permuted inside [0, valid_len) "
                          "-- exact-arithmetic-invariant by policy point 2, recorded there at ~1e-7",
}

ULP_CHECK = "ulp_amplification_does_not_dominate_the_family"


def measure_reference_floor(args) -> int:
    """TASK-N08 owed step 3. Returns the exit code documented in the module docstring."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not args.allow_cpu:
        print("BLOCKED: reference_floor(C5) is an A10G desk measurement (owed step 2's venue) "
              "and no CUDA device is visible. Re-run on the desk GPU, or pass --allow-cpu to "
              "produce an artifact stamped not_reportable for smoke-testing only.")
        return 3
    stub = install_cte_stub(honour_prior_used_len=True)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    print(f"\n=== reference_floor(C5): {len(FLOOR_FRAMES) * len(FLOOR_SEQLEN_Q)} cells, "
          f"{len(FLOOR_FAMILY)} implementations, device={device} ===\n")

    cells, floor, floor_at, dtypes = [], 0.0, None, set()
    for n_frames in FLOOR_FRAMES:
        for sq in FLOOR_SEQLEN_Q:
            # Per-cell seed, not one generator drawn down across cells: a cell whose inputs
            # depend on every draw before it cannot be reproduced on its own, and owed step 2
            # has to stage these inputs to trn2 cell by cell.
            seed = FLOOR_SEED_BASE + n_frames * 10 + sq
            gen = torch.Generator(device=device).manual_seed(seed)
            c5 = build(head_dim=HEAD_DIM, num_heads=NUM_HEADS, mtf=MTF, device=device)
            k_live, v_live = drive(c5, n_frames, NUM_HEADS, HEAD_DIM, device, gen)
            q = torch.randn(1, NUM_HEADS, sq, HEAD_DIM, generator=gen, device=device)
            dtypes.update(str(t.dtype) for t in (q, k_live, v_live))

            members, failed, spans = {}, {}, None

            def _try(name, fn):
                try:
                    members[name] = fn()
                except Exception as exc:            # noqa: BLE001 - a member that cannot run is
                    failed[name] = f"{type(exc).__name__}: {exc}"[:200]   # data, not a crash

            _try("sdpa_fused", lambda: gpu_sdpa(q, k_live, v_live))
            # Before the permutation member, which rebinds c5.visible_group for good.
            _try("adapter_einsum", lambda: c5.compute_attention(0, 0, q))

            def _chunked():
                nonlocal spans
                out, spans = chunked_merge(q, k_live, v_live, stub, FLOOR_PARTS)
                return out
            _try("chunked_merge", _chunked)

            def _permuted():
                apply_injection(c5, "in_prefix_permuted")
                return c5.compute_attention(0, 0, q)
            _try("in_prefix_permuted", _permuted)

            pairs = _pairwise(members)
            cell_max = max((p["rel_err"] for p in pairs), default=0.0)
            if cell_max > floor:
                floor, floor_at = cell_max, {"frames": n_frames, "seqlen_q": sq,
                                             "pair": max(pairs, key=lambda p: p["rel_err"])["pair"]}

            # The ulp probe, through the fused reference so it measures the shape and not the
            # adapter: same mathematics, inputs moved one ulp.
            probe: dict = {}
            try:
                qp, kp, vp = _ulp_up(q), _ulp_up(k_live), _ulp_up(v_live)
                in_move = max(rel_err(qp, q), rel_err(kp, k_live), rel_err(vp, v_live))
                out_move = rel_err(gpu_sdpa(qp, kp, vp), members["sdpa_fused"])
                probe = {"input_rel_move": in_move, "output_rel_move": out_move,
                         "kappa": (out_move / in_move) if in_move > 0 else None}
            except Exception as exc:                # noqa: BLE001
                probe = {"unavailable_because": f"{type(exc).__name__}: {exc}"[:200]}

            retained = None
            if args.tensors_out:
                os.makedirs(args.tensors_out, exist_ok=True)
                path = os.path.join(args.tensors_out, f"c5_floor_{n_frames}f_sq{sq}.pt")
                torch.save({"seed": seed, "dtype": str(q.dtype), "device": device,
                            "frames": n_frames, "seqlen_q": sq, "num_heads": NUM_HEADS,
                            "head_dim": HEAD_DIM, "max_total_frames": MTF,
                            "generator": "torch.Generator(device).manual_seed(seed); "
                                         "drive() draws k,v per frame then q",
                            "inputs": {"q": q.detach().cpu(), "k_live": k_live.detach().cpu(),
                                       "v_live": v_live.detach().cpu()},
                            "outputs": {k: v.detach().cpu() for k, v in members.items()}}, path)
                retained = {"path": path, "sha256": _sha256(path), "bytes": os.path.getsize(path)}

            cells.append({"frames": n_frames, "seqlen_q": sq, "seed": seed,
                          "members_completed": len(members), "members_failed": failed,
                          "chunked_spans": spans, "n_parts_requested": FLOOR_PARTS,
                          "pairs": pairs, "cell_max_rel_err": cell_max,
                          "bit_identical_pairs": [p["pair"] for p in pairs
                                                  if p["rel_err"] == 0.0],
                          "ulp_probe": probe, "retained_tensors": retained})
            print(f"    frames={n_frames:5d} sq={sq}  members={len(members)}/{len(FLOOR_FAMILY)}"
                  f"  max_pairwise={cell_max:.3e}"
                  f"  ulp_out_move={probe.get('output_rel_move', float('nan')):.3e}")

    # ---- guards ---------------------------------------------------------------------------
    expected_cells = len(FLOOR_FRAMES) * len(FLOOR_SEQLEN_Q)
    thin = [(c["frames"], c["seqlen_q"], c["members_completed"]) for c in cells
            if c["members_completed"] < 3]
    emit("floor_geometry_fully_covered",
         len(cells) == expected_cells and not thin and dtypes == {"torch.float32"},
         cells=len(cells), expected=expected_cells, cells_under_three_members=thin,
         input_dtypes=sorted(dtypes),
         detail="owed step 3 requires the maximum over the FULL geometry at fp32: every "
                "frames x seqlen_q cell present, and at least three of the four evaluations "
                "completing in each. A maximum over whichever cells happened to run is not the "
                "maximum, and a floor measured at another dtype is not this shape's floor")

    agreeing = [c for c in cells if c["cell_max_rel_err"] == 0.0]
    emit("floor_family_members_disagree", not agreeing and floor > 0.0,
         floor=f"{floor:.3e}", cells_with_zero_spread=[(c["frames"], c["seqlen_q"])
                                                      for c in agreeing],
         detail="a family that agrees bit-exactly measured nothing, and per policy point 1 "
                "bit-equality is this harness's failure signal rather than a tight result. Two "
                "members differing is the whole content of the measurement")

    missing = [(c["frames"], c["seqlen_q"]) for c in cells if not c["retained_tensors"]]
    emit("per_implementation_outputs_retained", bool(args.tensors_out) and not missing,
         tensors_out=args.tensors_out, cells_missing_tensors=missing,
         detail="a bare maximum with no retained members cannot be re-derived when REQ-088's "
                "SDK tuple moves -- the same rule TASK-N21's vacuity guard applies to its "
                "repeats. Pass --tensors-out pointing OUTSIDE this repository: both of its "
                "remotes are public and the deep cells are ~19 MB of tensors each. The path "
                "and sha256 land in this artifact so the figures stay traceable to bytes")

    probed = [c for c in cells if c["ulp_probe"].get("output_rel_move") is not None]
    worst_probe = max((c["ulp_probe"]["output_rel_move"] for c in probed), default=None)
    ulp_ok = bool(probed) and len(probed) == len(cells) and worst_probe < floor
    emit(ULP_CHECK, ulp_ok,
         worst_output_rel_move=(f"{worst_probe:.3e}" if worst_probe is not None else None),
         floor=f"{floor:.3e}", cells_probed=f"{len(probed)}/{len(cells)}",
         worst_kappa=max((c["ulp_probe"].get("kappa") or 0.0 for c in probed), default=None),
         detail="the family measures sensitivity to evaluation ORDER; this probe measures "
                "sensitivity to the INPUTS at one ulp. Both numbers are measured, and the "
                "comparison is between them -- no absolute cutoff is applied, because a cutoff "
                "would decide the outcome instead of the measurement. If a one-ulp input move "
                "shifts the output as far as the whole family spread, the order-derived maximum "
                "is not representative of this shape: it is a finding to escalate per owed "
                "step 3, not a bar to score Phase 4 against")

    n_fail = sum(1 for r in RESULTS if not r["pass"])
    other_fail = sum(1 for r in RESULTS if not r["pass"] and r["check"] != ULP_CHECK)
    if other_fail:
        verdict, code = "FAIL", 1
    elif not ulp_ok:
        verdict, code = "ESCALATE", 2
    else:
        verdict, code = "MEASURED", 0

    with open(args.floor_out, "w") as fh:
        json.dump({"suite": "reference_floor(C5) across implementations "
                            "(TASK-N08 owed step 3)",
                   "device": device,
                   "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                   "not_reportable": device != "cuda",
                   "harness_revision": _harness_revision(repo_root),
                   "geometry": {"frames": list(FLOOR_FRAMES), "seqlen_q": list(FLOOR_SEQLEN_Q),
                                "num_heads": NUM_HEADS, "head_dim": HEAD_DIM,
                                "max_total_frames": MTF, "input_dtypes": sorted(dtypes),
                                "plan": "plan_tiles(1124 + 1) = 1 tile x 1152"},
                   "family": FLOOR_FAMILY,
                   "reference_floor_c5": floor,
                   "reference_floor_at": floor_at,
                   "floor_usable_as_a_bar": ulp_ok,
                   "cross_device": False,
                   "cross_device_note": "every member ran on one device and two of the four "
                                        "went through install_cte_stub(), so this reaches one "
                                        "axis short of cross-device. It bounds evaluation-order "
                                        "disagreement, not Neuron-vs-GPU disagreement",
                   "supersedes": "the analytic n*u interim 1.37e-4 named in TASK-N21's Phase 4 "
                                 "floor, which bounds neither this comparison nor the value "
                                 "taken from it",
                   "worst_ulp_output_rel_move": worst_probe,
                   # Every cell, every pair, both probe figures, the chunk spans and the sha256
                   # of the retained tensors. Without this the floor is a bare scalar and
                   # re-deriving it when REQ-088's SDK tuple moves means re-running the GPU;
                   # owed step 3's verification clause requires the members be retained, and a
                   # maximum whose inputs are not in the record is not retained.
                   "cells": cells,
                   "n_checks": len(RESULTS), "n_fail": n_fail, "verdict": verdict,
                   "results": RESULTS}, fh, indent=2)

    print(f"\n    reference_floor(C5) = {floor:.6e}  at {floor_at}")
    print(f"\n===== {verdict}: {len(RESULTS) - n_fail}/{len(RESULTS)} guards passed, "
          f"exit {code}\n")
    return code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", choices=INJECTIONS, default=None)
    ap.add_argument("--honour-prior-used-len", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                 "c5_attention_parity_results.json"))
    ap.add_argument("--measure-reference-floor", action="store_true",
                    help="TASK-N08 owed step 3: measure reference_floor(C5) across "
                         "implementations instead of running the parity suite")
    ap.add_argument("--floor-out", default=os.path.join(os.path.dirname(__file__),
                                                       "c5_reference_floor_results.json"))
    ap.add_argument("--tensors-out", default=None,
                    help="directory to retain per-implementation outputs and inputs in; "
                         "required by owed step 3's verification, and must be outside this "
                         "repository (public remotes, ~19 MB per deep cell)")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="permit the floor measurement without a GPU, stamping the artifact "
                         "not_reportable")
    args = ap.parse_args()

    # Owed step 3 is a measurement, not a check. It writes its own artifact and deliberately
    # does not join the 7/7 parity verdict other documents cite by count.
    if args.measure_reference_floor:
        return measure_reference_floor(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    install_cte_stub(honour_prior_used_len=bool(args.honour_prior_used_len))
    H, D = NUM_HEADS, HEAD_DIM          # MTF is the module constant; do not rebind it here
    gen = torch.Generator(device=device).manual_seed(1234)

    print(f"\n=== C5 attention parity (device={device}, H={H}, D={D}, mtf={MTF}, "
          f"inject={args.inject}) ===\n")
    emit("device", True, device=device,
         gpu=torch.cuda.get_device_name(0) if device == "cuda" else "cpu")

    # ---- 1. attention output vs GPU SDPA, at several depths ---------------------------
    TOL = 1e-4              # generous vs the ~1e-6 floor; catches every real corruption
    worst, rows = 0.0, []
    for n_frames, sq in ((8, 1), (100, 1), (MTF, 1), (MTF, 5)):
        c5 = build(head_dim=D, num_heads=H, mtf=MTF, device=device)
        k_live, v_live = drive(c5, n_frames, H, D, device, gen)
        if args.inject:
            apply_injection(c5, args.inject)
        q = torch.randn(1, H, sq, D, generator=gen, device=device)
        got = c5.compute_attention(0, 0, q)
        want = gpu_sdpa(q, k_live, v_live)
        e = rel_err(got, want)
        eq = torch.equal(got.float(), want.float())
        worst = max(worst, e)
        rows.append({"frames": n_frames, "seqlen_q": sq, "rel_err": e, "bit_equal": eq})
        print(f"    frames={n_frames:5d} sq={sq}  rel_err={e:.3e}  bit_equal={eq}")

    # The in-prefix permutation is the one injection that MUST pass: softmax does not care
    # about key order inside the live prefix. Everything else must be caught.
    expect_pass = args.inject in (None, "in_prefix_permuted")
    emit("attention_output_matches_gpu_sdpa", (worst < TOL) == expect_pass,
         worst_rel_err=f"{worst:.3e}", tol=TOL, rows=rows,
         expect="within tol" if expect_pass else "OUTSIDE tol (injection must be caught)",
         detail="compute_attention through the real NeuronAttentionAdapter vs "
                "F.scaled_dot_product_attention over the identical visible keys. This is the "
                "gate the 34-check suite lacks entirely (rel_err defined, never called)")

    # ---- 2. vacuity guard: bit-equality would be the WRONG gate -----------------------
    if args.inject is None:
        any_bit_equal = any(r["bit_equal"] for r in rows)
        emit("bit_equality_would_be_the_wrong_gate", not any_bit_equal,
             bit_equal_anywhere=any_bit_equal, floor=f"{worst:.3e}",
             detail="The correct implementation is NOT bit-exact: the tile/merge online-softmax "
                    "reassociates an fp32 reduction. If this ever passes bit-exactly, the "
                    "tolerance gate above has become vacuous -- do NOT 'tighten' it to "
                    "torch.equal")

    # ---- 3. the missing prefix-alignment gate ----------------------------------------
    c5 = build(head_dim=D, num_heads=H, mtf=MTF, device=device)
    drive(c5, 12, H, D, device, gen)
    if args.inject:
        apply_injection(c5, args.inject)
    k_nhd, v_nhd, valid, plan = c5.visible_group(0, 0)
    kb, vb, _ = c5.visible_kv(0, 0)
    n = int(valid)
    want_k = kb[0].reshape(H, plan.padded, D).permute(1, 0, 2)[:n]
    want_v = vb[0].reshape(H, plan.padded, D).permute(1, 0, 2)[:n]
    aligned = torch.equal(k_nhd[:n], want_k) and torch.equal(v_nhd[:n], want_v)
    # in_prefix_permuted deliberately reorders within the prefix, so alignment is expected to
    # differ there while the ATTENTION result stays correct -- that is the whole point.
    align_expect = args.inject in (None,)
    emit("prefix_alignment_is_gated", aligned == align_expect,
         aligned=aligned, valid_len=n,
         detail="visible_group's live prefix must equal the reference reshape. valid_len bounds "
                "a PHYSICAL prefix, so a transform moving live keys out of [0,valid_len) "
                "attends padding and drops real keys -- NOT permutation-invariant, contrary to "
                "check_c5_camera_cache.py:332-335's stated rationale")

    # ---- 4. and prove the safe case really is safe -----------------------------------
    if args.inject is None:
        c5b = build(head_dim=D, num_heads=H, mtf=MTF, device=device)
        k_live, v_live = drive(c5b, 12, H, D, device, gen)
        q = torch.randn(1, H, 1, D, generator=gen, device=device)
        base = c5b.compute_attention(0, 0, q)
        apply_injection(c5b, "in_prefix_permuted")
        perm_out = c5b.compute_attention(0, 0, q)
        e_perm = rel_err(perm_out, base)
        emit("in_prefix_permutation_is_genuinely_harmless", e_perm < 1e-4,
             rel_err=f"{e_perm:.3e}",
             detail="Softmax IS permutation-invariant inside the live prefix. This is the half "
                    "of the suite's docstring rationale that is correct, and it is why the "
                    "alignment gate above must not over-claim")

    # ---- 5. the tail mask is load-bearing, and its failure SHRINKS with depth ---------
    # This is the ODQ7 dependency made explicit. The whole padded-tail argument rests on
    # attention_cte honouring prior_used_len per tile inside a combine; V1 proved it for a
    # single call, and the trn2 half is still open. Re-run the stub WITHOUT that behaviour
    # and confirm the gate above would catch it -- otherwise check 1 is only testing the
    # stub's agreement with itself.
    #
    # The magnitudes are the reason this is a check and not a comment: unhonoured padding
    # cost rel_err 9.917e-01 at 8 frames but only 1.420e-02 at 1124 in the committed run
    # (c5_attention_parity_results.json), because the live fraction grows. At production
    # depth a broken tail mask looks like ~1% noise, not a bug. Do not restate those as
    # rounded prose again -- the emit() below interpolates what THIS run measured.
    if args.inject is None:
        install_cte_stub(honour_prior_used_len=False)
        deep_err = shallow_err = 0.0
        for n_frames, sink in ((8, "shallow"), (MTF, "deep")):
            c5c = build(head_dim=D, num_heads=H, mtf=MTF, device=device)
            k_live, v_live = drive(c5c, n_frames, H, D, device, gen)
            q = torch.randn(1, H, 1, D, generator=gen, device=device)
            e = rel_err(c5c.compute_attention(0, 0, q), gpu_sdpa(q, k_live, v_live))
            if sink == "deep":
                deep_err = e
            else:
                shallow_err = e
        install_cte_stub(honour_prior_used_len=True)      # restore
        emit("tail_masking_is_load_bearing", deep_err > TOL and shallow_err > TOL,
             shallow_rel_err=f"{shallow_err:.3e}", deep_rel_err=f"{deep_err:.3e}", tol=TOL,
             detail="With prior_used_len ignored, check 1 must fail -- proving it tests the "
                    "tail mask and not the stub agreeing with itself. Note the error SHRINKS "
                    f"with depth ({shallow_err:.3e} at 8 frames vs {deep_err:.3e} at {MTF}): "
                    f"at production depth a broken tail mask presents as {deep_err:.1%} "
                    "noise. Figures interpolated from this run, never restated -- the fields "
                    "above are the same two numbers. ODQ7's trn2 half is what confirms the "
                    "real kernel honours it per tile inside a combine")

    # ---- 6. adapter geometry guard ----------------------------------------------------
    trunk_like = NeuronAttentionAdapter(head_dim=64, max_seqlen_k=1152, tile=1152,
                                        check_env=False)
    try:
        build(head_dim=D, num_heads=H, mtf=MTF, device=device, adapter=False)
        NeuronCameraCache(num_iterations=1, trunk_depth=1, num_heads=H, head_dim=D,
                          device=device, max_total_frames=MTF,
                          attention_adapter=trunk_like)
        rejected, msg = False, ""
    except ValueError as exc:
        rejected, msg = True, str(exc)[:90]
    emit("mismatched_adapter_head_dim_is_rejected", rejected, msg=msg,
         detail="The adapter folds scale=head_dim**-0.5 into q, so the trunk's head_dim=64 "
                "adapter on the head_dim=128 camera cache is silently wrong (rel_err ~0.3, "
                "finite, right-shaped). Must raise at construction")

    n_fail = sum(1 for r in RESULTS if not r["pass"])
    verdict = "PASS" if n_fail == 0 else "FAIL"
    with open(args.out, "w") as fh:
        json.dump({"suite": "C5 attention-output parity (V5(a) numerical half)",
                   "device": device, "injected_bug": args.inject,
                   "honour_prior_used_len": bool(args.honour_prior_used_len),
                   "n_checks": len(RESULTS), "n_fail": n_fail, "verdict": verdict,
                   "results": RESULTS}, fh, indent=2)
    print(f"\n===== {verdict}: {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed "
          f"(inject={args.inject})\n")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
