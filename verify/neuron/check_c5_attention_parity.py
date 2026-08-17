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

A sixth, added 2026-08-14, is ``TASK-N08`` owed step 1: ``--arm neuron`` selects the real
kernel instead of the stub. Everything above runs against ``install_cte_stub()``, which is what
makes it desk-runnable and is also the one thing that could make a *compiled* run vacuous — a
trn2 run that reports the stub's numbers has measured the desk twice. The arm therefore leaves
``NeuronAttentionAdapter._load_kernel`` alone, lets the real import happen, and records which
kernel it got **and which device ran it** — the second axis was added 2026-08-17, because
recording which kernel OBJECT loaded left a cpu-resolved trn2 run passing every provenance guard
— and it scores that provenance **before** it scores any ``rel_err``. Checks 2 and
5 above are the two the arm cannot inherit and re-states in its own terms: bit-equality means
something stronger there (two devices, two reduction orders) and the negative control has to
come from the INPUT, because the real kernel cannot be asked to ignore ``prior_used_len``. See
the block above :func:`run_neuron_arm`. Authoring the arm is desk work; its first *execution* is
Tier 2 and belongs to ``TASK-N21``'s C5-shape floor or ``TASK-N14a``.

Run::

    python3 verify/neuron/check_c5_attention_parity.py
    python3 verify/neuron/check_c5_attention_parity.py --inject axes_mixed

The floor measurement needs the Tier 1 container, and two things about that container
determine the invocation. There is no ``git`` in it, so provenance has to be handed in from
the host; and only ``/home/kwehden/lingbot-tier1`` is mounted, so a ``--tensors-out`` anywhere
else writes into the container's own filesystem and the retained tensors do not survive it --
the guard passes against files the host cannot see. Both are easy to get wrong silently::

    H=verify/neuron/check_c5_attention_parity.py
    docker exec \
      -e LINGBOT_HARNESS_REV="$(git rev-parse HEAD)" \
      -e LINGBOT_HARNESS_BRANCH="$(git rev-parse --abbrev-ref HEAD)" \
      -e LINGBOT_HARNESS_DIRTY="$([ -n "$(git status --porcelain -- $H)" ] && echo 1 || echo 0)" \
      lingbot-tier1 python "$PKG/verify/neuron/check_c5_attention_parity.py" \
        --measure-reference-floor \
        --tensors-out /home/kwehden/lingbot-tier1/c5_floor_tensors_<date>

``LINGBOT_HARNESS_DIRTY`` is computed over the harness **source**, not over
``verify/neuron/``: the run writes its own artifact into that directory, so a directory-scoped
status makes every run after the first report dirty whatever the code did.

Exit codes. The parity suite uses ``0`` pass / ``1`` fail. The two measuring modes add two more,
because "measured" and "usable as a verdict" are different answers:

===  ==========================================================================
 0   MEASURED -- every guard passed and the figure may be consumed
 1   a guard failed; the figure is not established
 2   measured, but not consumable as a bar or a verdict here.
     ``--measure-reference-floor``: ESCALATE, a one-ulp input move rivals the
     whole family spread, so the maximum is a finding and not a bar
     (``TASK-N08`` owed step 3's own words). ``--arm neuron``: the discrepancy
     was measured and no ``--tolerance`` was supplied to score it against, so
     it is a number no gate consumed -- never read this as a pass.
 3   BLOCKED -- this venue cannot perform the requested measurement, and
     ``blocked_because`` in the artifact names which venue property is
     missing: ``no_cuda`` for the floor (an A10G desk measurement),
     ``neuron_env_not_configured`` or ``neuron_kernel_unavailable`` for the arm
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


# The shipped method, captured at import before anything in this process can patch it, and the
# witness list that records every patch that did happen. Both exist for the Neuron arm's vacuity
# guards, and both have to be taken HERE rather than inside the arm: an arm that reads
# _load_kernel after the fact cannot tell a pristine method from one the stub restored, and
# "install_cte_stub() was never called" asserted by the caller is exactly the claim that needs
# evidence. STUB_INSTALLS is appended by install_cte_stub itself, so the evidence is produced by
# the thing being ruled out and not by the code hoping it did not happen.
_PRISTINE_LOAD_KERNEL = NeuronAttentionAdapter._load_kernel
STUB_INSTALLS: list = []


# ------------------------------------------------------------------------------------
# The kernel fake. Convention A, exactly as check_tile_merge.py pins it: attention over
# k_prior[:prior_used_len] ++ k_active, returning (out, neg_max, sum_recip) when
# cache_softmax=True. This is the contract V2 confirmed on trn2 hardware; faking it here
# keeps the check desk-runnable. The tail-masking half is what ODQ7's trn2 half must confirm.
# ------------------------------------------------------------------------------------
def make_cte_stub(honour_prior_used_len: bool = True):
    """Build the fake kernel and return it, **without** installing it anywhere.

    Split out from :func:`install_cte_stub` so that "make the fake" and "patch the adapter" are
    separable: the Neuron arm's guards treat any patch as disqualifying, so a test that needs a
    callable kernel without a patch (and without a witness entry it would then have to explain
    away) has to be able to get one. Every caller in the checks above wants both and calls
    :func:`install_cte_stub`.

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

    return stub


def install_cte_stub(honour_prior_used_len: bool = True):
    """Patch ``NeuronAttentionAdapter._load_kernel`` (the instance method the adapter calls)."""
    stub = make_cte_stub(honour_prior_used_len)
    NeuronAttentionAdapter._load_kernel = lambda self: stub        # noqa: SLF001
    # The witness. Recorded unconditionally, including the caller's line, so the arm's
    # "the stub was never installed in this process" guard is evidence rather than a hope.
    frame = sys._getframe(1)                                       # noqa: SLF001
    STUB_INSTALLS.append({"honour_prior_used_len": bool(honour_prior_used_len),
                          "called_from": f"{os.path.basename(frame.f_code.co_filename)}:"
                                         f"{frame.f_lineno} in {frame.f_code.co_name}"})
    return stub


def build_with_adapter(head_dim, num_heads, mtf, device, check_env):
    """The one construction site, returning ``(cache, adapter)``.

    ``check_env`` is a parameter and not a constant because the two arms need opposite answers:
    the stub arm must skip ``_assert_env`` (no Neuron env exists at the desk and the kernel is
    fake anyway), and the Neuron arm must NOT skip it, since ``check_env=False`` is precisely
    how a run reaches ``_load_kernel`` on a host that was never configured for Trainium.
    """
    probe = NeuronCameraCache(num_iterations=1, trunk_depth=1, num_heads=num_heads,
                              head_dim=head_dim, device=device, max_total_frames=mtf)
    ad = NeuronAttentionAdapter(head_dim=head_dim, max_seqlen_k=probe.plan.padded,
                                tile=probe.plan.tile, check_env=check_env)
    return NeuronCameraCache(num_iterations=1, trunk_depth=1, num_heads=num_heads,
                             head_dim=head_dim, device=device, max_total_frames=mtf,
                             attention_adapter=ad), ad


def build(head_dim=128, num_heads=16, mtf=1124, device="cpu", adapter=True):
    if not adapter:
        return NeuronCameraCache(num_iterations=1, trunk_depth=1, num_heads=num_heads,
                                 head_dim=head_dim, device=device, max_total_frames=mtf)
    return build_with_adapter(head_dim, num_heads, mtf, device, check_env=False)[0]


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


def replay(c5, k_nhd, v_nhd, num_heads, head_dim):
    """Rebuild the cache state from staged ``[n, H, D]`` k/v instead of from a generator.

    The Neuron arm cannot use :func:`drive` with the recorded seed. ``torch.Generator`` is
    device-scoped: a generator seeded N on one device does not produce another device's stream,
    so reseeding on trn2 would silently attend different keys than the staged reference was
    computed from -- the comparison would still print a number. Replaying the tensors is why
    owed step 2 stages them, and ``append`` is a copy, so the replayed prefix is bit-exact.
    """
    for i in range(int(k_nhd.shape[0])):
        c5.append(0, 0, k_nhd[i].reshape(1, num_heads, 1, 1, head_dim),
                  v_nhd[i].reshape(1, num_heads, 1, 1, head_dim))


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


# ------------------------------------------------------------------------------------------
# The Neuron arm -- TASK-N08 owed step 1 (added 2026-08-14)
#
# WHAT THE ARM IS FOR. Every check above runs against install_cte_stub(). That is correct for
# desk work and it is the only reason this file exists at the desk, but it means the numbers
# above are GPU-vs-GPU: REQ-086 makes invoking nkilib.core.attention.attention_cte the
# definition of "real Trainium/Inferentia hardware" for REQ-028, and the stub is not it. So a
# compiled trn2 run that reports the stub's figures has measured the desk twice and satisfied
# nothing. That is this arm's own vacuity failure mode, and it is the specific way a silent
# fallback would pass: the stub agrees with the GPU reference to ~1e-6, so a fallback looks
# exactly like a success.
#
# HOW IT IS PREVENTED, AND WHY IN THIS ORDER. The arm scores kernel provenance BEFORE it scores
# any rel_err, and returns without computing one if provenance fails -- a discrepancy figure
# produced by an unknown kernel is not a weaker result, it is a misleading one. The three
# provenance guards are: install_cte_stub() was never called in this process (witnessed by
# STUB_INSTALLS, which install_cte_stub appends itself), _load_kernel is still the shipped
# method (identity against _PRISTINE_LOAD_KERNEL, captured at import), and the object the real
# _load_kernel returned lives under nkilib.
#
# WHY check_env=True HERE. The adapter's _assert_env() wants PJRT_DEVICE=NEURON and
# NEURON_PLATFORM_TARGET_OVERRIDE=trn2, each of which cost a real trn2 run to learn. The arm
# must not skip it: check_env=False is how a run reaches _load_kernel on a host nobody
# configured. But note carefully what an env failure does and does not show. Blocking because
# the env vars are unset is NOT evidence that the arm raises rather than falling back -- it
# never reached the import. The two blocks are therefore separate causes with separate fields,
# and raises_rather_than_falling_back stays null in the env case rather than claiming a
# property this run did not exercise.
#
# WHAT IT COMPARES AGAINST. Owed step 2's staged tensors, and nothing else. A CPU SDPA computed
# on the trn2 host cannot discharge REQ-021's GPU comparison (the 2026-08-08 review's P1), and
# reseeding drive() on another device attends different keys than the reference was computed
# from (see :func:`replay`). Staging is transport, not trust, so every staged file is verified
# against the sha256 the floor artifact committed before its contents are used. Unlike that
# artifact this comparison IS cross-device, which is the axis it exists to cover.
#
# WHAT MAKES A GREEN RESULT MEAN ANYTHING. Provenance rules out the stub; two further guards
# rule out the two ways a REAL kernel can still produce a vacuous pass, and both are checks
# main() only has for the stub arm. The first is INVERTED: bit-equality with the staged
# reference is a FAILURE. Two devices with different reduction orders cannot agree exactly on
# correct output, so 0.0 means both sides ran the same path -- and the tolerance emit below
# passes on worst <= tol, which 0.0 satisfies for every tol anyone could supply. It is
# therefore scored whether or not --tolerance was given. The second is the negative control,
# input-side by necessity: main()'s desk control installs a stub that ignores prior_used_len,
# and the real kernel cannot be asked to ignore it, so the control instead hands attend_groups
# a valid_len of plan.padded -- the whole tile, so tile_valid_lens makes the kernel attend
# padding -- and requires the answer to move FURTHER from the reference than the measured cell
# did. Without it a rel_err is not evidence that this comparison discriminates at all. Cells
# run shallow first: the same breakage reads ~0.99 at 8 frames and ~0.014 at 1124, so sorted
# filename order, which puts 1124f ahead of 8f, would spend the scarce trn2 minutes on the
# insensitive depth first.
#
# WHAT IT DOES NOT DECIDE. No tolerance is hardcoded. TASK-N21's Phase 4 bar is
# max(reference_floor, 3 x noise_floor) and belongs to that task; the harness's own "generous"
# TOL=1e-4 above is not a derivation and must not become one by being reused here. --tolerance
# is supplied by the caller or the run reports MEASURED_NOT_SCORED and exits 2, because a
# measurement no gate consumed must not exit 0.
# ------------------------------------------------------------------------------------------
ARM_REFERENCE_MEMBER = "sdpa_fused"       # the fused GPU member: REQ-021's comparison, staged

# Device types that are NOT a Trainium device. An EXCLUSION list, and the direction is the whole
# design: an allowlist would have to name the string trn2's torch reports for a Neuron device, and
# no run in this repository has ever established it -- which is why --device is a parameter rather
# than a literal in the first place. A wrong literal there fails a run that DID execute on Neuron
# and blames the port, the same hazard case 14 of the desk suite measured the deep control to
# avoid. "cpu, cuda and meta are not Trainium" needs no trn2 run to be true and is falsifiable at
# the desk in both directions, so it is the only form of this guard that cannot cost a spot hour
# to a guess.
NON_NEURON_DEVICE_TYPES = ("cpu", "cuda", "meta")

# The three points at which a device is observable, plus the control's. All four are recorded per
# cell and all four are scored: a partial move -- inputs staged to the device, output materialised
# on the host, or a control that ran somewhere the measurement did not -- is exactly the case a
# single witness would miss.
DEVICE_WITNESSES = ("input_device", "kv_device", "compute_device", "control_device")


def _dev(t) -> str:
    """The device a tensor actually lives on, as a string.

    A one-line function on purpose: it is the seam the desk suite patches so its fake-kernel cases
    can report an xla device, which lies about the OBSERVATION and leaves ``NON_NEURON_DEVICE_TYPES``
    shipped exactly as written. A test that widened the tuple instead would delete the guard it is
    there to exercise.
    """
    return str(t.device)


def _device_type(name: str) -> str:
    """The type half of a device string. Total: never raises on an unfamiliar name."""
    try:
        return torch.device(name).type
    except Exception:                                              # noqa: BLE001
        return str(name).split(":")[0]


def _xla_runtime():
    """``torch_xla.core.xla_model`` if importable, else None. **Never raises.**

    ``bench_streaming.py``'s ``_xla`` raises when torch_xla is absent, which is correct for an
    explicit ``--device xla`` request and wrong here: this is called above both BLOCKED returns, so
    an exception would turn the desk's two exit-3 paths into a traceback and delete the only
    desk-verifiable evidence for the raise-not-fallback property.
    """
    try:
        import torch_xla.core.xla_model as xm                      # noqa: PLC0415
        return xm
    except Exception:                                              # noqa: BLE001
        return None


def _resolve_arm_device(requested, cuda_available, xla, env_pjrt):
    """Which device the arm runs on, as ``(device_string, how_it_was_decided)``.

    Pure and total -- every input is a parameter, nothing is imported here and nothing raises --
    which is what makes all four branches desk-testable, since torch_xla is installed nowhere in
    this project and the branch that matters most is the one this desk cannot reach.

    Returns a **string**, never a ``torch.device``: ``_write_arm`` json.dumps the body this value
    lands in, and ``torch.device`` is not JSON-serialisable, so returning ``xm.xla_device()``
    directly would raise while writing the artifact -- after the measurement, losing the whole run
    with nothing on disk.
    """
    if requested:
        return requested, "explicit"
    if xla is not None and (env_pjrt or "").upper() == "NEURON":
        try:
            return str(xla.xla_device()), "xla_runtime"
        except Exception as exc:                                   # noqa: BLE001
            return (("cuda" if cuda_available else "cpu"),
                    f"xla_runtime_present_but_failed: {type(exc).__name__}")
    if cuda_available:
        return "cuda", "cuda_visible"
    return "cpu", "nothing_identified_the_venue"


def _kernel_identity(fn) -> dict:
    """Everything observable about the object ``_load_kernel`` returned.

    ``attention_cte`` is a ``GenericKernel`` instance and not a function, so ``__module__`` may
    live on the instance, on its type, or on a wrapped callable. All of them are recorded and the
    guard accepts a match on any -- the alternative is a guard that fails on the real kernel for
    a reason that has nothing to do with which kernel it is.
    """
    inner = getattr(fn, "func", None) or getattr(fn, "__wrapped__", None)
    mods = [m for m in (getattr(fn, "__module__", None), type(fn).__module__,
                        getattr(inner, "__module__", None)) if isinstance(m, str)]
    return {"repr": repr(fn)[:200], "type": type(fn).__name__,
            "modules_observed": sorted(set(mods)),
            "qualname": getattr(fn, "__qualname__", getattr(fn, "__name__", None)),
            "inner_qualname": getattr(inner, "__qualname__", None)}


def _under_nkilib(ident: dict) -> bool:
    return any(m == "nkilib" or m.startswith("nkilib.") for m in ident["modules_observed"])


def _write_arm(args, body: dict, verdict: str, code: int) -> int:
    n_fail = sum(1 for r in RESULTS if not r["pass"])
    with open(args.arm_out, "w") as fh:
        json.dump({"suite": "C5 attention parity through the REAL Neuron kernel "
                            "(TASK-N08 owed step 1's arm)",
                   "arm": "neuron",
                   "reference_member": ARM_REFERENCE_MEMBER,
                   "cross_device": True,
                   "cross_device_note": "unlike c5_reference_floor_results.json, which is "
                                        "GPU-vs-GPU with two of four members going through "
                                        "install_cte_stub(), this compares a Neuron kernel "
                                        "against staged GPU outputs -- the axis that artifact "
                                        "says it stops one short of",
                   "harness_revision": _harness_revision(
                       os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))),
                   "geometry": {"num_heads": NUM_HEADS, "head_dim": HEAD_DIM,
                                "max_total_frames": MTF,
                                "plan": "plan_tiles(1124 + 1) = 1 tile x 1152"},
                   **body,
                   "n_checks": len(RESULTS), "n_fail": n_fail, "verdict": verdict,
                   "results": RESULTS}, fh, indent=2)
    print(f"\n===== {verdict}: {len(RESULTS) - n_fail}/{len(RESULTS)} guards passed, "
          f"exit {code}  ->  {args.arm_out}\n")
    return code


def run_neuron_arm(args) -> int:
    """TASK-N08 owed step 1. Returns the exit code documented in the module docstring."""
    # Ask the runtime for the device name instead of guessing it, and record how the answer was
    # reached. This line read `args.device or ("cuda" if cuda else "cpu")`, which on a trn2 host --
    # no CUDA -- resolved to **cpu**, and nothing downstream scored it: the arm would load the real
    # nkilib kernel, pass all nine provenance guards, and publish a rel_err from a comparison that
    # never touched Trainium. Neither half of the fix is sufficient alone. Derivation without a
    # guard is the silent-landing shape this arm exists to condemn, and a guard without derivation
    # makes an operator's guess at an unestablished device string the difference between a
    # measurement and a refusal at $8.5964/hr.
    xla = _xla_runtime()
    device, device_resolution = _resolve_arm_device(
        args.device, torch.cuda.is_available(), xla, os.environ.get("PJRT_DEVICE"))
    print(f"\n=== C5 attention parity, NEURON arm (device={device} [{device_resolution}], "
          f"H={NUM_HEADS}, D={HEAD_DIM}, mtf={MTF}) ===\n")

    # ---- provenance, scored first ---------------------------------------------------------
    emit("stub_was_never_installed_in_this_process", not STUB_INSTALLS,
         stub_installs=STUB_INSTALLS,
         witness="module-level STUB_INSTALLS, appended by install_cte_stub itself at every call",
         detail="REQ-086 makes invoking nkilib.core.attention.attention_cte the definition of "
                "real hardware for REQ-028, so a run reporting the stub's numbers is this arm's "
                "own vacuity failure and not a pass. The stub agrees with the GPU reference to "
                "~1e-6, which is why this cannot be left to inspection of the output")
    live = _kernel_identity(NeuronAttentionAdapter._load_kernel)
    pristine = NeuronAttentionAdapter._load_kernel is _PRISTINE_LOAD_KERNEL
    emit("load_kernel_is_still_the_shipped_method", pristine,
         observed=live, expected=_kernel_identity(_PRISTINE_LOAD_KERNEL),
         detail="identity against the method captured at import, before this process could "
                "patch it. install_cte_stub assigns a lambda whose qualname is "
                "install_cte_stub.<locals>.<lambda>; an equality-on-name test would also accept "
                "a same-named replacement, and identity does not")
    if not pristine or STUB_INSTALLS:
        return _write_arm(args, {"kernel": None, "raises_rather_than_falling_back": None,
                                 "not_scored_because": "kernel provenance failed; no rel_err "
                                                       "was computed, because a discrepancy "
                                                       "from an unknown kernel misleads rather "
                                                       "than under-informs"},
                          "FAIL", 1)

    # ---- the venue: env first, and an env block is NOT the fallback property --------------
    try:
        NeuronAttentionAdapter._assert_env()                       # noqa: SLF001
        env_why = None
    except RuntimeError as exc:
        env_why = str(exc)[:400]
    env = {"PJRT_DEVICE": os.environ.get("PJRT_DEVICE"),
           "NEURON_PLATFORM_TARGET_OVERRIDE": os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE"),
           "assert_env_raised": env_why}
    if env_why:
        print(f"  BLOCKED: {env_why}")
        return _write_arm(args, {"device": device, "device_resolution": device_resolution, "env": env, "kernel": None,
                                 "blocked_because": "neuron_env_not_configured",
                                 "raises_rather_than_falling_back": None,
                                 "raises_note": "null on purpose. This run never reached the "
                                                "nkilib import, so it demonstrates nothing "
                                                "about what happens when the kernel is absent "
                                                "-- exporting both variables is what makes that "
                                                "question reachable, at the desk included"},
                          "BLOCKED", 3)

    # ---- the real import. No stub, no patch: whatever _load_kernel does is the answer -----
    cache, adapter = build_with_adapter(HEAD_DIM, NUM_HEADS, MTF, device, check_env=True)
    emit("adapter_instance_does_not_shadow_load_kernel", "_load_kernel" not in vars(adapter),
         instance_attributes=sorted(vars(adapter)),
         detail="the class-level guard above cannot see a per-instance assignment, and "
                "adapter._load_kernel = ... would bypass it silently")
    try:
        kernel, load_exc = adapter._load_kernel(), None            # noqa: SLF001
    except Exception as exc:                                       # noqa: BLE001
        kernel, load_exc = None, {"type": type(exc).__name__, "message": str(exc)[:400],
                                  "cause": type(exc.__cause__).__name__ if exc.__cause__
                                  else None}
    if kernel is None:
        emit("absent_kernel_raises_rather_than_falling_back", True, raised=load_exc,
             detail="the property this arm exists to have. _load_kernel raised instead of "
                    "returning a substitute, so no run can reach compute_attention on a host "
                    "without the Neuron stack -- a silent fallback is how the stub got into "
                    "every existing figure. Desk-verifiable, since the desk GPU is exactly such "
                    "a host once the two env vars above are exported")
        return _write_arm(args, {"device": device, "device_resolution": device_resolution, "env": env, "kernel": None,
                                 "blocked_because": "neuron_kernel_unavailable",
                                 "raises_rather_than_falling_back": True},
                          "BLOCKED", 3)

    ident = _kernel_identity(kernel)
    emit("kernel_is_the_real_nkilib_attention_cte", _under_nkilib(ident), kernel=ident,
         required="a module under nkilib",
         detail="scored before any rel_err. REQ-086's definition of real hardware is which "
                "kernel ran, so this is the gate that makes the discrepancy below mean anything")
    if not _under_nkilib(ident):
        return _write_arm(args, {"device": device, "device_resolution": device_resolution, "env": env, "kernel": ident,
                                 "raises_rather_than_falling_back": False,
                                 "not_scored_because": "_load_kernel returned an object that is "
                                                       "not under nkilib; no rel_err computed"},
                          "FAIL", 1)

    # ---- the reference. Transport is not trust: verify the bytes before using them ---------
    manifest, manifest_why = {}, None
    try:
        with open(args.reference_manifest) as fh:
            art = json.load(fh)
        manifest = {os.path.basename(c["retained_tensors"]["path"]): c
                    for c in art["cells"] if c.get("retained_tensors")}
    except Exception as exc:                                       # noqa: BLE001
        manifest_why = f"{type(exc).__name__}: {exc}"[:200]
    staged = sorted(f for f in os.listdir(args.reference_tensors) if f.endswith(".pt")) \
        if args.reference_tensors and os.path.isdir(args.reference_tensors) else []
    verified, bad = [], []
    for name in staged:
        cell = manifest.get(name)
        got = _sha256(os.path.join(args.reference_tensors, name))
        if cell and got == cell["retained_tensors"]["sha256"]:
            verified.append(name)
        else:
            bad.append({"file": name, "sha256": got,
                        "expected": (cell or {}).get("retained_tensors", {}).get("sha256"),
                        "why": "not in the manifest" if not cell else "sha256 mismatch"})
    emit("staged_reference_outputs_present_and_verified", bool(verified) and not bad,
         reference_tensors=args.reference_tensors, manifest=args.reference_manifest,
         manifest_unreadable_because=manifest_why, files_verified=verified, files_rejected=bad,
         detail="owed step 2's tensors are the only admissible reference: a CPU SDPA on the "
                "trn2 host cannot discharge REQ-021's GPU comparison, and reseeding on another "
                "device attends different keys (replay()'s docstring). Staging moves bytes and "
                "confers no integrity, so each file is checked against the sha256 committed in "
                "c5_reference_floor_results.json. A run with nothing to compare against is the "
                "vacuous case and fails here rather than reporting a verdict on nothing")
    if not verified or bad:
        return _write_arm(args, {"device": device, "device_resolution": device_resolution, "env": env, "kernel": ident,
                                 "raises_rather_than_falling_back": False,
                                 "not_scored_because": "no verified staged reference"},
                          "FAIL", 1)

    # ---- and only now, the comparison -----------------------------------------------------
    # Shallow first, which sorted() on the filenames is not: "c5_floor_1124f" sorts before
    # "c5_floor_8f". The order is load-bearing rather than cosmetic -- an unhonoured tail reads
    # ~0.99 at 8 frames and ~0.014 at 1124, so the shallow cell is the sensitive detector, and
    # a preemption partway through must not be the reason it never ran. Every verified name is
    # in the manifest by construction: the verification loop above rejects anything that is not.
    cells, worst, worst_at, dtypes = [], 0.0, None, set()
    for name in sorted(verified, key=lambda f: (manifest[f]["frames"],
                                                manifest[f]["seqlen_q"])):
        blob = torch.load(os.path.join(args.reference_tensors, name), weights_only=False)
        ref = blob["outputs"].get(ARM_REFERENCE_MEMBER)
        row = {"file": name, "frames": blob["frames"], "seqlen_q": blob["seqlen_q"],
               "reference_device": blob["device"], "dtype": blob["dtype"],
               "geometry_matches": (blob["num_heads"] == NUM_HEADS
                                    and blob["head_dim"] == HEAD_DIM
                                    and blob["max_total_frames"] == MTF)}
        dtypes.add(blob["dtype"])
        try:
            if ref is None:
                raise RuntimeError(f"staged blob has no {ARM_REFERENCE_MEMBER} output")
            if not row["geometry_matches"]:
                raise RuntimeError("staged blob was produced at another geometry")
            # No dtype argument anywhere: the arm runs at the dtype the reference was staged
            # at. Casting here would compare a bf16 evaluation against an fp32 reference and
            # score it against an fp32-measured floor, which is the substitution the 2026-08-08
            # review's P1 caught in the other direction. A bf16 answer needs a bf16 reference.
            q = blob["inputs"]["q"].to(device)
            k_live = blob["inputs"]["k_live"].to(device)
            v_live = blob["inputs"]["v_live"].to(device)
            c5, adapter = build_with_adapter(HEAD_DIM, NUM_HEADS, MTF, device, check_env=True)
            row["input_device"] = _dev(q)
            replay(c5, k_live, v_live, NUM_HEADS, HEAD_DIM)
            k_seen, v_seen, valid, plan = c5.visible_group(0, 0)
            # The tensors attend_groups actually hands the kernel, so this is the load-bearing
            # witness of the three: `device` above is what was requested, this is what happened.
            row["kv_device"] = _dev(k_seen)
            n = int(valid)
            row["replay_is_bit_exact"] = bool(
                n == int(k_live.shape[0])
                and torch.equal(k_seen[:n].cpu(), k_live.cpu())
                and torch.equal(v_seen[:n].cpu(), v_live.cpu()))
            got = c5.compute_attention(0, 0, q)
            # BEFORE the .cpu() below, which is the only chance: that call is what makes the
            # comparison host-side, and after it every tensor here reads "cpu" and the evidence of
            # where the output materialised is gone.
            row["compute_device"] = _dev(got)
            # Compared on the host in fp32 so a second device's arithmetic cannot enter the
            # comparison itself.
            row["rel_err"] = rel_err(got.detach().cpu().float(), ref.float())
            row["bit_equal"] = torch.equal(got.detach().cpu().float(), ref.float())
            row["also_vs_adapter_einsum"] = (
                rel_err(got.detach().cpu().float(), blob["outputs"]["adapter_einsum"].float())
                if "adapter_einsum" in blob["outputs"] else None)
        except Exception as exc:                                   # noqa: BLE001 - a cell that
            row["failed"] = f"{type(exc).__name__}: {exc}"[:300]   # cannot run is data

        # The negative control gets its OWN try, and the separation is load-bearing.
        # `valid_len == plan.padded` is an input the real attention_cte has never been given on this
        # project, so a kernel assert here is a plausible FIRST-run outcome. Sharing the block above
        # left such a cell both scored and `failed`: every_staged_cell_ran_and_replayed_bit_exactly
        # reported PASS while listing the cell in its own cells_failed, and the control guard came
        # out 0/N with the exception text nowhere in it -- an operator at $8.5964/hr read a FAIL
        # verdict with no reason. A control that could not run is recorded as control_failed and
        # surfaces in the guard that needed it.
        if row.get("rel_err") is not None:
            # From the input, on the same buffers this cell just scored: valid_len = plan.padded
            # makes tile_valid_lens (neuron_attention.py:367) hand the kernel the whole tile, so it
            # attends the padding rows append never wrote. full_like keeps it a 0-d device tensor --
            # a Python int here would bake the length into the graph and cost a NEFF per frame
            # (tile_valid_lens' docstring).
            try:
                ctrl = adapter.attend_groups(
                    q[0].permute(1, 0, 2).contiguous(),
                    [(k_seen, v_seen, torch.full_like(valid, int(plan.padded)), plan)])
                row["control_valid_len"] = int(plan.padded)
                # The control carries the 100x margin, so a control that quietly ran somewhere
                # else would score that margin from a device the measurement never used.
                row["control_device"] = _dev(ctrl)
                row["control_rel_err"] = rel_err(
                    ctrl.permute(1, 0, 2).unsqueeze(0).detach().cpu().float(), ref.float())
            except Exception as exc:                                       # noqa: BLE001
                row["control_failed"] = f"{type(exc).__name__}: {exc}"[:300]
        if row.get("rel_err") is not None and row["rel_err"] > worst:
            worst, worst_at = row["rel_err"], {"frames": row["frames"],
                                               "seqlen_q": row["seqlen_q"], "file": name}
        cells.append(row)
        print(f"    {name}: frames={row['frames']:5d} sq={row['seqlen_q']} "
              + (f"rel_err={row['rel_err']:.3e} "
                 f"control={row.get('control_rel_err', float('nan')):.3e} "
                 f"replay_exact={row.get('replay_is_bit_exact')}"
                 if "rel_err" in row else f"FAILED {row.get('failed')}"))

    scored = [c for c in cells if c.get("rel_err") is not None]
    emit("every_staged_cell_ran_and_replayed_bit_exactly",
         bool(scored) and len(scored) == len(cells)
         and all(c.get("replay_is_bit_exact") for c in scored),
         cells_scored=f"{len(scored)}/{len(cells)}",
         cells_failed=[{"file": c["file"], "failed": c["failed"]} for c in cells
                       if "failed" in c],
         replay_exact=[c.get("replay_is_bit_exact") for c in cells],
         detail="append is a copy, so a replayed prefix that is not bit-equal to the staged "
                "keys means this run attended something the reference never saw, and the "
                "rel_err below would be a comparison of two different problems")

    # ---- the venue, SELF-OBSERVED: which device the compared tensors actually lived on ---------
    # The guards above establish which kernel OBJECT loaded. None of them establishes which DEVICE
    # ran it, and the artifact recorded `device` as a field -- an echo of the argument, which is a
    # claim the harness makes about itself and never checks. Same class as _harness_revision
    # refusing to record a caller-supplied SHA as self_observed_git: `--device` is a caller
    # assertion, a tensor's own .device is self-observation, and only the second one is evidence.
    offending = [{"file": c["file"], "frames": c["frames"],
                  "witnesses": {w: c[w] for w in DEVICE_WITNESSES if w in c},
                  "disagreeing": [w for w in DEVICE_WITNESSES if w in c
                                  and _device_type(c[w]) in NON_NEURON_DEVICE_TYPES]}
                 for c in scored
                 if [w for w in DEVICE_WITNESSES if w in c
                     and _device_type(c[w]) in NON_NEURON_DEVICE_TYPES]]
    emit("compared_tensors_ran_on_a_neuron_device", bool(scored) and not offending,
         device_requested=args.device, device_resolved=device,
         device_resolution=device_resolution, xla_runtime_importable=xla is not None,
         excluded_device_types=list(NON_NEURON_DEVICE_TYPES),
         witnesses_per_cell=[{"file": c["file"],
                              **{w: c[w] for w in DEVICE_WITNESSES if w in c}} for c in scored],
         cells_on_a_non_neuron_device=offending,
         detail="scored before any rel_err, for the reason the kernel-provenance guards are: a "
                "discrepancy from an unknown venue misleads rather than under-informs. It reads "
                "the tensors' own .device, captured inside the loop before the host-side .cpu() "
                "normalisation erases it, and it excludes rather than requires -- naming the "
                "string a Trainium device reports would fail a run that DID execute on Neuron, "
                "which is the one failure direction this arm must not add. IF THE KERNEL "
                "PROVENANCE GUARDS PASSED AND THESE WITNESSES READ cpu, the finding is that "
                "device provenance is not observable through Tensor.device on this stack -- "
                "attention_cte is a GenericKernel handed plain torch tensors, and a baremetal NKI "
                "path may ship host buffers to the device itself. That is a fact about "
                "observability and NOT a C5 parity failure; the first run to hit it should record "
                "it as the fact this repository has never established, not re-derive the bar")

    tol = args.tolerance

    # ---- the vacuity guard, INVERTED: agreement to the last bit is a failure here ----------
    bit_equal_at = [{"frames": c["frames"], "seqlen_q": c["seqlen_q"], "file": c["file"]}
                    for c in scored if c.get("bit_equal")]
    emit("neuron_and_gpu_do_not_agree_bit_exactly",
         bool(scored) and not bit_equal_at and worst > 0.0,
         worst_rel_err=f"{worst:.3e}", cells_bit_equal=bit_equal_at,
         detail="inverted on purpose, and scored whether or not a --tolerance was supplied: "
                "the tolerance emit below passes on worst <= tol, and 0.0 satisfies every "
                "tolerance anyone could derive, so a bit-exact run would otherwise be written "
                "MEASURED. Two devices with different reduction orders cannot agree exactly on "
                "CORRECT output, so max_abs_diff == 0.0 means both sides ran the same path -- "
                "and on this harness's history that path is install_cte_stub(). This and the "
                "kernel-provenance guards above are the same check from two directions")

    # ---- and the negative control, which on the real kernel must come from the INPUT -------
    controlled = [c for c in scored if c.get("control_rel_err") is not None]
    control_failures = [{"file": c["file"], "control_failed": c["control_failed"]}
                        for c in scored if "control_failed" in c]
    # The margin is against the MEASUREMENT, not against the caller's bar. Scoring it against `tol`
    # imposed a silent ceiling on whatever TASK-N21 derives: the deep controls are ~1.3e-2, so any
    # bar at or above that failed this guard on a run whose own worst error was 1.6e-06 -- a
    # correct run rejected for having a generous bar, and in the counter-intuitive direction.
    # Whether the bar is too generous to catch a broken mask is a real defect, but it is a
    # DIFFERENT one and it belongs to TASK-N21's bar, so it is named separately below.
    CONTROL_MARGIN = 100.0
    undiscriminating = [
        {"frames": c["frames"], "seqlen_q": c["seqlen_q"], "rel_err": c["rel_err"],
         "control_rel_err": c["control_rel_err"]} for c in controlled
        if not c["control_rel_err"] > CONTROL_MARGIN * c["rel_err"]]
    emit("tail_masking_is_load_bearing_on_the_real_kernel",
         bool(controlled) and len(controlled) == len(scored) and not undiscriminating,
         cells_controlled=f"{len(controlled)}/{len(scored)}", margin=CONTROL_MARGIN,
         by_cell={f"{c['frames']}f_sq{c['seqlen_q']}":
                  f"{c['rel_err']:.3e} -> {c['control_rel_err']:.3e}" for c in controlled},
         cells_that_did_not_discriminate=undiscriminating,
         control_failures=control_failures,
         detail="main()'s desk control is a stub built to ignore prior_used_len; the real "
                "kernel cannot be asked to ignore it, so the control is input-side -- "
                "attend_groups re-run on this cell's own buffers with valid_len = plan.padded, "
                "the whole tile, which makes the kernel attend the padding rows append never "
                "wrote. It must land at least 100x FURTHER from the reference than the measured "
                "cell did, at every scored depth. If it does not, the parity gate above "
                "discriminates nothing and a fallback, or a kernel that silently ignores "
                "prior_used_len, reads as a pass. Cells ran shallow first because the same "
                "breakage reads ~0.99 at 8 frames and ~0.014 at 1124. A control that could not "
                "run at all appears in control_failures: valid_len == the tile length is an input "
                "the real kernel has never been given here, so a kernel assert is a plausible "
                "first-run outcome and is a finding about that contract, not about C5 parity")

    # ---- and the other half of the same property: is the BAR tight enough to catch it? ------
    # Split out of the guard above deliberately. Both are required, but they fail for opposite
    # reasons and an operator has to be able to tell which: above, the harness cannot see a broken
    # mask; here, it can see it and the bar would forgive it. Emitted only when a --tolerance was
    # supplied, the same condition the parity emit itself carries -- with no bar there is no bar
    # to be too generous, and a guard that passes on an absent input is the vacuity this file's
    # other inversions exist to catch.
    if tol is not None:
        too_generous = [{"frames": c["frames"], "seqlen_q": c["seqlen_q"],
                         "control_rel_err": c["control_rel_err"]}
                        for c in controlled if not c["control_rel_err"] > tol]
        emit("the_tolerance_is_tight_enough_to_reject_a_broken_mask",
             bool(controlled) and not too_generous,
             tolerance=tol,
             tightest_control=(f"{min(c['control_rel_err'] for c in controlled):.3e}"
                               if controlled else None),
             cells_the_bar_would_forgive=too_generous,
             detail="the parity emit passes on worst <= tol, so a tolerance at or above the "
                    "control error would pass a run whose mask was doing nothing. That makes the "
                    "BAR the defect rather than the port, which is why it is not folded into the "
                    "control guard above: TASK-N21's Phase 4 bar is "
                    "max(reference_floor, 3 x noise_floor) and nothing in that formula has an "
                    "upper bound, yet the deep control is only ~1.3e-2. If this is the only "
                    "failure, the run measured correctly and the bar needs re-deriving. Requires "
                    "at least one control: with none, there is no evidence either way")

    # Coverage is recorded, not gated by default: TASK-N10's shallow-first rule means a first
    # trn2 run legitimately carries one cell. --require-full-geometry is what the final Phase 4
    # run passes, so a partial run can never be cited as the whole geometry either way.
    expected = len(manifest) or len(staged)
    full = len(scored) == expected and expected > 0
    if args.require_full_geometry:
        emit("arm_geometry_fully_covered", full, cells_scored=len(scored), expected=expected,
             detail="requested explicitly. The Phase 4 run that discharges REQ-028 needs every "
                    "staged cell; a shallow-first probe does not and must not claim to")

    if tol is not None:
        emit("neuron_vs_staged_gpu_reference_within_tolerance", bool(scored) and worst <= tol,
             worst_rel_err=f"{worst:.3e}", tolerance=tol, worst_at=worst_at,
             detail="the bar is the caller's, computed by TASK-N21 as "
                    "max(reference_floor, 3 x noise_floor). Nothing here derives it: the "
                    "harness's own TOL=1e-4 above is a generous parity tolerance and not a "
                    "floor. A discrepancy above the bar escalates per TASK-N21's step-5 rule "
                    "and is never used to move the floor -- the value being scored cannot "
                    "re-derive the bar that scores it")

    body = {"device": device, "device_requested": args.device,
            "device_resolution": device_resolution,
            "device_observed": sorted({c[w] for c in scored for w in DEVICE_WITNESSES if w in c}),
            "neuron_device_observed": bool(scored) and not offending,
            "xla_runtime_importable": xla is not None,
            "env": env, "kernel": ident,
            "raises_rather_than_falling_back": False,
            "raises_note": "false because the kernel WAS available here, so the property was "
                           "not exercised by this run; it is recorded true by the runs that "
                           "block with neuron_kernel_unavailable",
            "reference_tensors": args.reference_tensors,
            "reference_manifest": args.reference_manifest,
            "input_dtypes": sorted(dtypes),
            "cells_expected": expected, "cells_scored": len(scored),
            "full_geometry_covered": full,
            "tolerance": tol,
            "worst_rel_err": (worst if scored else None), "worst_rel_err_at": worst_at,
            "cells": cells}
    n_fail = sum(1 for r in RESULTS if not r["pass"])
    if n_fail:
        verdict, code = "FAIL", 1
    elif tol is None:
        body["not_scored_because"] = ("no --tolerance was supplied, so this run measured a "
                                      "discrepancy that no gate consumed. Exit 2, never 0")
        verdict, code = "MEASURED_NOT_SCORED", 2
    else:
        verdict, code = "MEASURED", 0
    if scored:
        print(f"\n    worst Neuron-vs-staged-GPU rel_err = {worst:.6e}  at {worst_at}")
    return _write_arm(args, body, verdict, code)


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
    ap.add_argument("--arm", choices=("stub", "neuron"), default="stub",
                    help="which kernel the run goes through. 'stub' is every existing "
                         "invocation and is unchanged; 'neuron' is TASK-N08 owed step 1's arm, "
                         "which leaves _load_kernel alone and lets the real nkilib import "
                         "happen (REQ-086's definition of real hardware for REQ-028)")
    ap.add_argument("--arm-out", default=os.path.join(os.path.dirname(__file__),
                                                     "c5_neuron_arm_results.json"))
    ap.add_argument("--reference-tensors", default=None,
                    help="the staged owed-step-2 directory the Neuron arm compares against; "
                         "its files are verified against --reference-manifest before use")
    ap.add_argument("--reference-manifest",
                    default=os.path.join(os.path.dirname(__file__),
                                         "c5_reference_floor_results.json"),
                    help="the committed floor artifact, whose cells[].retained_tensors.sha256 "
                         "is the manifest for the staged tensors")
    ap.add_argument("--tolerance", type=float, default=None,
                    help="the bar the Neuron arm scores against, supplied by the caller: "
                         "TASK-N21 computes max(reference_floor, 3 x noise_floor). Omit it and "
                         "the arm reports MEASURED_NOT_SCORED with exit 2 -- there is "
                         "deliberately no default, because this harness's own generous "
                         "TOL=1e-4 is not a derived floor")
    ap.add_argument("--require-full-geometry", action="store_true",
                    help="make full staged-cell coverage a guard rather than a recorded field; "
                         "the Phase 4 run that discharges REQ-028 passes this, a shallow-first "
                         "probe under TASK-N10's rule does not")
    ap.add_argument("--device", default=None,
                    help="device for the Neuron arm. Which string names a Trainium device is "
                         "not established by any run in this repository, so it is a parameter "
                         "here rather than a guess in the source. Omitted, the arm ASKS the "
                         "runtime -- torch_xla's xla_device() when it is importable and "
                         "PJRT_DEVICE says NEURON -- and falls back to cuda, else cpu, which is "
                         "what the desk raise-not-fallback check wants. Supplying it is a caller "
                         "assertion and buys no provenance: the compared tensors' own .device is "
                         "scored either way, so --device cpu on a trn2 host FAILS rather than "
                         "publishing a number")
    args = ap.parse_args()

    # Owed step 3 is a measurement, not a check. It writes its own artifact and deliberately
    # does not join the 7/7 parity verdict other documents cite by count. The same is true of
    # the Neuron arm, and for the same reason -- plus one of its own: it must not run any code
    # path that has touched install_cte_stub(), so it dispatches before the stub install below.
    if args.measure_reference_floor:
        return measure_reference_floor(args)
    if args.arm == "neuron":
        return run_neuron_arm(args)

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
