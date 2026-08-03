"""End-to-end check that C4's wiring did not change what the model does.

spec/neuron-port/design.md C4: both flow-keyframe loop sites in ``gct_stream_window.py``
(``inference_streaming`` ~:556-600 and ``inference_windowed`` ~:1140-1162, design.md F6) now
call C3's ``keyframe_decision`` instead of inlining the flow computation and branch.

``check_keyframe_decision.py`` verifies C3 in isolation against the original function. That is
necessary but not sufficient: it does not exercise the *wiring* — the argument threading, the
``frames_since_kf`` bookkeeping, the ``is_first_streaming_frame`` condition (which differs
between the two sites: ``i == scale_frames`` vs ``kf_count == 0``), the ``hw`` derivation from
the depth tensor's shape, or the commit/rollback paths the decisions drive. A transcription
slip in any of those leaves C3 passing and the model wrong.

WHAT IS OBSERVED
----------------
The decision sequence, not just the predictions. ``inference_streaming`` returns no keyframe
observable, so this wraps the two cache hooks the decision selects between —
``_execute_deferred_eviction`` (keyframe) and ``_rollback_last_frame`` (non-keyframe) — and
records which fires per frame. That is instrumentation of the harness's model instance, not a
change to the code under test, and it yields the exact per-frame decision at BOTH sites. A
single flipped decision shows up as an integer difference, with no tolerance involved.

Predictions are compared too, but as a secondary gate, because a bit-exact cross-pass
comparison is only meaningful if the model is bit-exact run-to-run in the first place. Each
pass therefore runs the streaming loop twice and records whether its own two runs agree; the
prediction gate is skipped (and reported as skipped) when they do not. Asserting bit-exactness
against a baseline without that check would attribute framework nondeterminism to C4, or worse,
pass vacuously because both passes happened to be noisy in the same direction.

TWO REGIMES ARE REQUIRED, NOT ONE
---------------------------------
The default config (``--flow_threshold 8``) leaves the ``is_first_streaming_frame`` argument
completely unexercised, and this was found the only way it could be — by injecting a bug into
it and watching the suite pass anyway. At that threshold the first streaming frame's flow
already exceeds 8, so the frame is a keyframe by the flow term whether or not the first-frame
term exists. That argument is exactly the one whose expression DIFFERS between the two call
sites (``i == scale_frames`` vs ``kf_count == 0``), so leaving it uncovered would leave the
likeliest transcription slip invisible.

The second regime sets ``--flow_threshold 1e9``: the flow computation still runs (so C3 is
still on the critical path) but the flow term can never fire, leaving the first-frame and gap
terms as the only things deciding anything. Both regimes must be recorded and compared.

Protocol — restore by file copy, not ``git stash``, so a mid-run interruption cannot leave the
C4 edits buried on the stash stack:

    cp lingbot_map/models/gct_stream_window.py /tmp/gct_stream_window.C4.py
    git checkout -- lingbot_map/models/gct_stream_window.py     # pre-C4
    D="docker exec -w /home/kwehden/lingbot-tier1/lingbot-map lingbot-tier1 python"
    $D verify/neuron/check_c4_wiring.py --mode baseline --output_dir /tmp/c4_verify
    $D verify/neuron/check_c4_wiring.py --mode baseline --output_dir /tmp/c4_gaponly \
        --flow_threshold 1e9 --n_frames 12
    cp /tmp/gct_stream_window.C4.py lingbot_map/models/gct_stream_window.py   # restore C4
    # ...then the matching --mode compare runs, plus the --inject_bug runs below.

NOTE: ``/tmp`` here is the CONTAINER's /tmp — it is not part of the bind mount, so the outputs
are not visible from the host. Read them with ``docker exec``.

Synthetic images, not TUM: no TUM fixture is present on this desk, and the property under test
is "C4's wiring preserves behaviour", which any input producing a MIX of keyframe and
non-keyframe decisions exercises. The check asserts that mix occurred rather than assuming it —
an all-keyframe sequence would match trivially.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)


# ---------------------------------------------------------------------------------
# Instrumentation: recover the per-frame decision from the hooks it selects between.
# ---------------------------------------------------------------------------------
class DecisionRecorder:
    """Records 1 per committed (keyframe) frame and 2 per rolled-back (non-keyframe) frame.

    Wraps the bound methods on a live ``GCTStream``. Same encoding as the windowed loop's
    own ``frame_type`` (1=keyframe, 2=non-keyframe) so the two can be cross-checked.
    """

    def __init__(self, model):
        self.model = model
        self.seq: list[int] = []
        self._commit = model._execute_deferred_eviction
        self._rollback = model._rollback_last_frame
        model._execute_deferred_eviction = self._wrap(self._commit, 1)
        model._rollback_last_frame = self._wrap(self._rollback, 2)

    def _wrap(self, fn, code):
        def wrapped(*a, **kw):
            self.seq.append(code)
            return fn(*a, **kw)
        return wrapped

    def restore(self):
        self.model._execute_deferred_eviction = self._commit
        self.model._rollback_last_frame = self._rollback

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.restore()
        return False


def inject_bug(kind: str) -> None:
    """Replace C4's ``keyframe_decision`` with a subtly wrong variant.

    The discriminating-power guard. A compare pass that reports PASS is worthless unless the
    same comparison would report FAIL on a wiring bug, and the bugs worth worrying about here
    are quiet ones — an off-by-one in ``frames_since_kf``, a dropped
    ``is_first_streaming_frame``. Neither raises; both change which frames enter the cache.

    Run with ``--mode compare --inject_bug <kind>`` against a clean baseline: the suite must
    report FAIL. If it reports PASS, the clean 11/11 above was vacuous.

    MEASURED (24 frames, img 518, real checkpoint):
      * ``gap_off_by_one``  -> FAIL, 7/11 checks, first divergence at frame 18
      * ``threshold``       -> FAIL, 7/11 checks, first divergence at frame 12
      * ``first_frame``     -> PASS at ``--flow_threshold 8`` (the term is masked by the flow
        term there; see "TWO REGIMES" above), FAIL at ``--flow_threshold 1e9``.
    """
    from lingbot_map.models import gct_stream_window as gsw
    real = gsw.keyframe_decision

    if kind == "gap_off_by_one":
        def bugged(*a, max_non_keyframe_gap, **kw):
            # Fires the gap rule one frame early.
            return real(*a, max_non_keyframe_gap=max_non_keyframe_gap - 1, **kw)
    elif kind == "first_frame":
        def bugged(*a, is_first_streaming_frame, **kw):
            # Drops the forced keyframe on the first streaming frame.
            return real(*a, is_first_streaming_frame=False, **kw)
    elif kind == "threshold":
        def bugged(*a, flow_threshold, **kw):
            # Compares against a slightly different threshold.
            return real(*a, flow_threshold=flow_threshold * 0.9, **kw)
    else:
        raise ValueError(f"unknown --inject_bug kind: {kind}")

    gsw.keyframe_decision = bugged
    print(f"*** INJECTED BUG: {kind} -- this pass MUST report FAIL ***", flush=True)


def build_images(n_frames: int, size: int, device, seed: int = 0) -> torch.Tensor:
    """A deterministic pseudo-trajectory: a textured pattern that pans and dims.

    Fully reproducible across the two passes (fixed generator seed, integer crop offsets).
    Textured rather than smooth so the depth head produces non-degenerate output and the
    flow computation gets meaningful inputs.
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(3, size + 64, size + 64, generator=g)
    frames = []
    for i in range(n_frames):
        dy = (2 * i) % 64
        dx = (i * 13 // 10) % 64
        crop = base[:, dy:dy + size, dx:dx + size].clone()
        crop = crop * (0.9 + 0.1 * torch.cos(torch.tensor(i / 7.0)))
        frames.append(crop)
    return torch.stack(frames).unsqueeze(0).to(device)


def load_model(device, args):
    from lingbot_map.models.gct_stream_window import GCTStream

    model = GCTStream(
        img_size=args.image_size,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=1024,
        kv_cache_sliding_window=args.sliding_window,
        kv_cache_scale_frames=args.scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=True,          # flashinfer is not installed in the desk container
        camera_num_iterations=4,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.to(device).eval(), len(missing), len(unexpected)


def _cpu(pred, keys=("pose_enc", "depth")):
    return {k: pred[k].detach().float().cpu() for k in keys
            if k in pred and torch.is_tensor(pred[k])}


def run(mode: str, args) -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    if args.inject_bug:
        inject_bug(args.inject_bug)

    model, n_missing, n_unexpected = load_model(dev, args)
    images = build_images(args.n_frames, args.image_size, dev)
    print(f"[{mode}] device={dev} images={tuple(images.shape)} "
          f"missing_keys={n_missing} unexpected_keys={n_unexpected}", flush=True)

    record: dict = {
        "mode": mode, "device": str(dev),
        "n_missing_keys": n_missing, "n_unexpected_keys": n_unexpected,
        "config": {k: v for k, v in vars(args).items()},
    }

    def streaming_pass():
        model.clean_kv_cache()
        with DecisionRecorder(model) as rec:
            pred = model.inference_streaming(
                images, num_scale_frames=args.scale_frames,
                flow_threshold=args.flow_threshold,
                max_non_keyframe_gap=args.max_gap,
            )
        return _cpu(pred), list(rec.seq)

    with torch.no_grad():
        # --- site 1: inference_streaming, flow mode. Run twice: the second run is the
        #     run-to-run determinism self-check that licenses the prediction gate below.
        s_pred, s_seq = streaming_pass()
        print(f"[{mode}] streaming decisions: {len(s_seq)} recorded", flush=True)
        s_pred2, s_seq2 = streaming_pass()

        record["streaming"] = s_pred
        record["streaming_seq"] = s_seq
        record["self_repeat_identical"] = bool(
            s_seq == s_seq2
            and s_pred.keys() == s_pred2.keys()
            and all(torch.equal(s_pred[k], s_pred2[k]) for k in s_pred)
        )
        print(f"[{mode}] run-to-run deterministic: {record['self_repeat_identical']}",
              flush=True)

        # --- site 2: inference_windowed, flow mode ---
        model.clean_kv_cache()
        with DecisionRecorder(model) as rec:
            pred_w = model.inference_windowed(
                images, window_size=args.window_size, overlap_size=args.overlap,
                num_scale_frames=args.scale_frames,
                flow_threshold=args.flow_threshold,
                max_non_keyframe_gap=args.max_gap,
            )
        record["windowed"] = _cpu(pred_w)
        record["windowed_seq"] = list(rec.seq)
        # frame_type is the windowed loop's OWN record of the same decisions: 0=scale,
        # 1=keyframe, 2=non-keyframe. Kept as an independent second observable.
        if "frame_type" in pred_w and torch.is_tensor(pred_w["frame_type"]):
            record["frame_type"] = pred_w["frame_type"].int().cpu()
        print(f"[{mode}] windowed decisions: {len(record['windowed_seq'])} recorded",
              flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    blob = os.path.join(args.output_dir, f"c4_{mode}.pt")
    torch.save(record, blob)
    print(f"[{mode}] wrote {blob}", flush=True)

    if mode == "baseline":
        for site in ("streaming", "windowed"):
            seq = record[f"{site}_seq"]
            n_kf, n_non = seq.count(1), seq.count(2)
            print(f"baseline {site} mix: {n_kf} keyframes, {n_non} non-keyframes")
            if n_kf == 0 or n_non == 0:
                print(f"WARNING: baseline {site} has no keyframe/non-keyframe mix; the "
                      f"compare pass would be vacuous there. Adjust --flow_threshold.")
        return 0

    return compare(record, args)


def compare(record: dict, args) -> int:
    base = torch.load(os.path.join(args.baseline_dir, "c4_baseline.pt"),
                      map_location="cpu", weights_only=False)
    results: list[dict] = []

    def emit(name, ok, **kw):
        results.append({"check": name, "pass": bool(ok), **kw})

    # ---- PRIMARY GATE: the decisions themselves, at both sites ----------------------
    for site in ("streaming", "windowed"):
        b, c = base.get(f"{site}_seq"), record.get(f"{site}_seq")
        if b is None or c is None:
            emit(f"{site}_decision_sequence_identical", False, detail="sequence not recorded")
            continue
        n_kf, n_non = b.count(1), b.count(2)
        first_div = next((i for i, (x, y) in enumerate(zip(b, c)) if x != y), None)
        emit(f"{site}_decision_sequence_identical",
             b == c,
             n_frames=len(b), n_keyframes=n_kf, n_non_keyframes=n_non,
             baseline_len=len(b), compare_len=len(c), first_divergence=first_div,
             detail="per-frame keyframe decision recovered from the commit/rollback hooks; "
                    "exact integers, no tolerance")
        emit(f"{site}_sequence_contains_both_outcomes", n_kf > 0 and n_non > 0,
             n_keyframes=n_kf, n_non_keyframes=n_non,
             detail="if either is 0 the identity check above is vacuous")

    # The windowed loop's own frame_type must agree with the hook trace, and with baseline.
    ft_b, ft_c = base.get("frame_type"), record.get("frame_type")
    if ft_b is not None and ft_c is not None:
        emit("windowed_frame_type_identical", bool(torch.equal(ft_b, ft_c)),
             n_differences=int((ft_b != ft_c).sum()), n_frames=int(ft_b.numel()),
             detail="independent second observable: the loop's own 0/1/2 record")
        trace = [t for t in record.get("windowed_seq", [])]
        from_ft = [int(v) for v in ft_c.flatten().tolist() if v in (1, 2)]
        emit("frame_type_agrees_with_hook_trace", from_ft == trace,
             n_from_frame_type=len(from_ft), n_from_hooks=len(trace),
             detail="cross-validates the instrumentation itself: if the recorder missed or "
                    "double-counted a call, these two counts diverge")

    # ---- SECONDARY GATE: predictions, only if the model is run-to-run deterministic --
    det = bool(base.get("self_repeat_identical")) and bool(record.get("self_repeat_identical"))
    emit("model_is_run_to_run_deterministic", det,
         baseline=bool(base.get("self_repeat_identical")),
         compare=bool(record.get("self_repeat_identical")),
         detail="licenses the bit-exact prediction comparison below; if False those checks "
                "are reported as SKIP because any diff would be framework noise, not C4")

    for site in ("streaming", "windowed"):
        for key in ("pose_enc", "depth"):
            a = base.get(site, {}).get(key)
            b = record.get(site, {}).get(key)
            name = f"{site}_{key}_bit_exact"
            if a is None and b is None:
                continue
            if a is None or b is None:
                emit(name, False, detail=f"present in only one pass "
                                         f"(baseline={a is not None}, compare={b is not None})")
                continue
            if a.shape != b.shape:
                emit(name, False, detail=f"shape {tuple(a.shape)} vs {tuple(b.shape)}")
                continue
            identical = bool(torch.equal(a, b))
            maxdiff = 0.0 if identical else float((a - b).abs().max())
            if not det:
                results.append({"check": name, "pass": True, "skipped": True,
                                "max_abs_diff": maxdiff,
                                "detail": "SKIP: model not run-to-run deterministic"})
                continue
            emit(name, identical, max_abs_diff=maxdiff, shape=tuple(a.shape),
                 detail="bit-exact" if identical else "DIVERGED")

    n_fail = sum(1 for r in results if not r["pass"])
    for r in results:
        tag = "SKIP" if r.get("skipped") else ("PASS" if r["pass"] else "FAIL")
        extra = "  ".join(f"{k}={v}" for k, v in r.items()
                          if k not in ("check", "pass", "skipped"))
        print(f"[{tag}] {r['check']}  {extra}", flush=True)

    out = {
        "suite": "neuron C4 wiring: real inference loops, pre-C4 baseline vs post-C4",
        "primary_gate": "per-frame keyframe decision sequence at both loop sites",
        "injected_bug": args.inject_bug,
        "expected_verdict": "FAIL" if args.inject_bug else "PASS",
        "n_checks": len(results), "n_fail": n_fail,
        "verdict": "PASS" if n_fail == 0 else "FAIL",
        "baseline_config": base.get("config"), "compare_config": record.get("config"),
        "results": results,
    }
    path = os.path.join(args.output_dir, "c4_wiring_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n=== {out['verdict']}: {len(results) - n_fail}/{len(results)} checks passed")
    print(f"=== wrote {path}")
    return 1 if n_fail else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("baseline", "compare"), required=True)
    # The container mounts the host's /home/kwehden/lingbot-tier1 at the same path, but
    # /local/home/... (the host symlink) is NOT inside the mount. Use the mounted path.
    p.add_argument("--checkpoint",
                   default="/home/kwehden/lingbot-tier1/models/lingbot-map-long.pt")
    p.add_argument("--output_dir", default="/tmp/c4_verify")
    p.add_argument("--baseline_dir", default="/tmp/c4_verify")
    # 518 is not a free choice: the checkpoint's patch_embed.pos_embed is [1, 1370, 1024],
    # i.e. (518/14)^2 + 1 patches. A smaller img_size fails to load with a size mismatch
    # even under strict=False, and loading the real weights is the point of this check.
    p.add_argument("--image_size", type=int, default=518)
    p.add_argument("--n_frames", type=int, default=24)
    p.add_argument("--scale_frames", type=int, default=4)
    p.add_argument("--sliding_window", type=int, default=8)
    p.add_argument("--window_size", type=int, default=12)
    p.add_argument("--overlap", type=int, default=4)
    p.add_argument("--flow_threshold", type=float, default=8.0)
    p.add_argument("--max_gap", type=int, default=6)
    p.add_argument("--inject_bug", choices=("gap_off_by_one", "first_frame", "threshold"),
                   default=None,
                   help="Discriminating-power guard: corrupt the decision and confirm the "
                        "suite FAILS. A compare pass that cannot fail proves nothing.")
    args = p.parse_args()
    return run(args.mode, args)


if __name__ == "__main__":
    sys.exit(main())
