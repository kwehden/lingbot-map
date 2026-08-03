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
   is the uncomfortable part: ~0.97 at 8 frames but ~**0.012** at 1124 frames, because the
   live fraction grows with depth. A broken tail mask presents as 1% noise at production
   depth, so "the numbers look close" is not evidence here.

Run::

    python3 verify/neuron/check_c5_attention_parity.py
    python3 verify/neuron/check_c5_attention_parity.py --inject axes_mixed

Nothing in ``attention.py``/``camera_head.py`` is edited; the GPU reference is only a valid
oracle while that stays true.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from lingbot_map.heads.neuron_camera_cache import NeuronCameraCache  # noqa: E402
from lingbot_map.layers.neuron_attention import NeuronAttentionAdapter  # noqa: E402

RESULTS: list = []


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", choices=INJECTIONS, default=None)
    ap.add_argument("--honour-prior-used-len", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                 "c5_attention_parity_results.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    install_cte_stub(honour_prior_used_len=bool(args.honour_prior_used_len))
    H, D, MTF = 16, 128, 1124
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
    # costs rel_err ~0.97 at 8 frames but only ~0.012 at 1124, because the live fraction
    # grows. At production depth a broken tail mask looks like 1% noise, not a bug.
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
                    "with depth (~0.97 at 8 frames vs ~0.012 at 1124): at production depth a "
                    "broken tail mask presents as ~1% noise. ODQ7's trn2 half is what "
                    "confirms the real kernel honours it per tile inside a combine")

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
