"""End-to-end check that routing the camera head through C5 does not change what it does.

spec/neuron-port/design.md C5: the camera head's per-frame ``torch.cat``
(``attention.py:285-286``/``:298-299``) becomes an indexed write at a device cursor, so the key
extent the graph sees is a compile-time constant. The seam is opt-in — ``CausalAttention.forward``
takes ``neuron_cache``/``neuron_slot``, and ``CameraCausalHead.install_neuron_cache`` supplies
them — and this file drives the real ``GCTStream`` with and without it.

``check_c5_camera_cache.py`` (34) verifies cache CONTENTS against the real head and
``check_c5_attention_parity.py`` (7) verifies attention OUTPUT against SDPA. Both are unit-level:
they construct a cache and call its methods. Neither exercises the *wiring* — the slot mapping
``(iteration, block)``, the ``skip_append`` handoff, the reshape the seam must reproduce, or the
lifetime across ``clean_kv_cache``. A transcription slip in any of those leaves 41 checks green
and the model wrong.

WHY THIS IS STRUCTURALLY STRONGER THAN check_c4_wiring.py
--------------------------------------------------------
C4 replaced code in place, so its baseline had to be captured from a ``git checkout``ed copy of
the pre-C4 file, in a separate process, and compared across runs. That protocol is sound but
fragile: the two passes cannot see each other, so run-to-run nondeterminism has to be
separately established before the comparison means anything.

Here the stock path and the C5 path both exist in ONE process, selected by one argument. Both
passes run back to back on the same model instance and the same inputs, so:

  * there is no baseline file to go stale, and no checkout to forget to restore;
  * the comparison is bit-exact by construction (``torch.equal``), not tolerance-based;
  * a run-to-run determinism self-check is still performed, because bit-exactness across two
    *configurations* is only meaningful if the model is bit-exact across two *runs*. It is
    reported, and the prediction gate is skipped rather than silently trusted if it fails.

WHAT IS OBSERVED
----------------
1. ``predictions_bit_exact`` — pose_enc and depth from ``inference_streaming``, stock vs C5.
   ``torch.equal``, no tolerance. This is the right gate here (unlike the attention parity
   suite, where the tile/merge legitimately reassociates an fp32 reduction): with no adapter
   installed the seam changes only WHERE keys are stored, and SDPA then sees the same keys in
   the same order, so any difference is a defect rather than arithmetic.
2. ``windowed_predictions_bit_exact`` — the same over ``inference_windowed``, which is the
   loop that calls ``clean_kv_cache`` once per WINDOW, mid-run. Streaming alone would never
   exercise the reset, and the reset is the single most dangerous part of the seam.
3. ``cache_survives_window_boundary`` — the installed instance is still installed after a
   windowed run. ``clean_kv_cache`` does ``del self.kv_cache``, so an earlier design that put
   the cache in ``head.kv_cache`` was silently orphaned: the next forward rebuilt the
   list-of-dicts and resumed the ``torch.cat`` path with no error and no log line. This check
   is what makes that failure loud, and ``--inject_bug orphan_on_clean`` proves it can fail.
4. ``window_boundary_resets_cursor`` — a boundary must zero the cursors. If it does not, the
   next window's visible set carries the previous window's keys (measured: GPU 3 keys vs C5 8,
   attention rel_err 0.90). Counted by wrapping ``reset``.
5. ``skip_append_not_honoured_before_first_append`` — the GPU's flag lives in dicts that do not
   exist until the first causal forward, and the driver's setter is guarded by
   ``kv_cache is not None`` (``gct_stream_window.py:365``), so ``_set_skip_append(True)`` before
   frame 0 is a silent no-op on the GPU while a host bool would persist. That asymmetry would
   be a permanent off-by-n. The seam takes ``skip_append`` from the dict per call, so this is
   true by construction; the check pins it so a future refactor to the cache's own bool fails
   here rather than in production.
6. ``decision_mix_is_nontrivial`` — asserts the run actually produced both keyframes and
   non-keyframes.
7. ``skip_append_regime_*`` — the SECOND regime, and it is required, not optional. See below.

TWO REGIMES ARE REQUIRED, AND THE FIRST DRAFT OF THIS FILE HAD ONLY ONE
-----------------------------------------------------------------------
Found the only way it could be found: by injecting a bug into the ``skip_append`` handoff and
watching the suite pass anyway. ``--inject_bug skip_append_ignored`` (commit every frame) and
``--inject_bug drop_pending_frame`` (exclude the uncommitted frame from the visible set) both
scored 9/9 against the flow-keyframe regime.

The cause is a real property of the driver, worth recording because it is counter-intuitive:
**in flow-keyframe mode ``_set_skip_append`` is never called at all.** The loop defers eviction,
runs the forward unconditionally, and only then decides — committing via
``_execute_deferred_eviction`` or undoing via ``_rollback_last_frame``
(``gct_stream_window.py:591-596``). ``_set_skip_append`` appears only in the FIXED-INTERVAL
branch (``:602``). So on the flow path ``skip_append`` is permanently ``False``, and both
injections above were no-ops rather than undetected bugs.

Note ``decision_mix_is_nontrivial`` does not protect against this: it counts commit-vs-rollback,
which the flow path does produce a healthy mix of (7 keyframes / 13 non-keyframes here). Those
are a different mechanism from the flag, and conflating them is exactly what made the one-regime
version look adequate.

Regime 2 therefore runs ``keyframe_interval=3`` with ``flow_threshold=0``, which selects the
fixed-interval branch and does set the flag. Both regimes run in every invocation; there is no
way to run only the reassuring one.

Also worth knowing for the same reason: on the camera head ``_rollback_last_frame`` is a no-op —
``CameraCausalHead`` has no ``rollback_last_frame`` and the driver's call is guarded by
``hasattr`` (``:397``). That is Gap A, preserved deliberately (design.md), not an oversight
here.

Run::

    D="docker exec -w /home/kwehden/lingbot-tier1/lingbot-map lingbot-tier1 python"
    $D verify/neuron/check_c5_wiring.py
    $D verify/neuron/check_c5_wiring.py --inject_bug slot_collapse    # must FAIL

No adapter is installed: ``nkilib`` is absent from the desk container, and installing it would
move the key extent to the padded constant and make the bit-exact comparison impossible. That
half is what ``check_c5_attention_parity.py`` gates numerically and what V3/V4(b) confirm on
trn2. Synthetic images for the same reason ``check_c4_wiring.py`` uses them: no TUM fixture is
present, and the property under test is "the seam preserves behaviour", which any input
producing a mix of decisions exercises.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)

RESULTS: list = []


def emit(check: str, ok: bool, **kw) -> None:
    RESULTS.append({"check": check, "pass": bool(ok), **kw})
    print(f"  [{'PASS' if ok else 'FAIL'}] {check}"
          + ("".join(f"\n           {k}={v}" for k, v in kw.items()) if kw else ""),
          flush=True)


# ---------------------------------------------------------------------------------
# Injections. Each corrupts the WIRING, not the cache, and each must make some check
# above fail. A compare pass that cannot fail proves nothing.
# ---------------------------------------------------------------------------------
INJECTIONS = ("slot_collapse", "orphan_on_clean", "no_reset_on_clean",
              "skip_append_ignored", "drop_pending_frame")


def inject_bug(kind: str) -> None:
    from lingbot_map.heads import camera_head as ch
    from lingbot_map.heads.neuron_camera_cache import NeuronCameraCache

    if kind == "slot_collapse":
        # Ignore the block index: all trunk_depth blocks share one slot. The 16 streams are
        # independent caches, so this cross-contaminates every block within an iteration.
        #
        # This one is caught TWICE, and the capacity assert fires first: 4 blocks writing one
        # slot exhausts it 4x sooner, so at the default --max_total_frames it raises before
        # any prediction is compared. That is a genuine detection but not the one being
        # demonstrated, so run it with a larger capacity to see the contamination itself:
        #   --inject_bug slot_collapse --max_total_frames 256
        real_slot = NeuronCameraCache.slot
        NeuronCameraCache.slot = lambda self, i, j: real_slot(self, i, 0)
    elif kind == "orphan_on_clean":
        # The pre-fix design: hold the cache where clean_kv_cache destroys it. The next
        # forward silently rebuilds the dicts and resumes torch.cat -- no error, no log.
        real_clean = ch.CameraCausalHead.clean_kv_cache

        def orphaning_clean(self):
            real_clean(self)
            self.neuron_cache = None
        ch.CameraCausalHead.clean_kv_cache = orphaning_clean
    elif kind == "no_reset_on_clean":
        # Install the cache but never reset it at a window boundary: the next window attends
        # the previous window's keys.
        def no_reset_clean(self):
            del self.kv_cache
            self.kv_cache = None
            self.frame_idx = 0
        ch.CameraCausalHead.clean_kv_cache = no_reset_clean
    elif kind == "skip_append_ignored":
        # Treat every frame as a keyframe: non-keyframes get committed, so the speculative
        # frame is never overwritten and depth runs ahead of the GPU's by one per non-keyframe.
        real_append = NeuronCameraCache.append

        def always_commit(self, i, j, k, v):
            self.set_skip_append(False, iter_idx=i)
            return real_append(self, i, j, k, v)
        NeuronCameraCache.append = always_commit
    elif kind == "drop_pending_frame":
        # The subtle one: store correctly but exclude the uncommitted frame from the visible
        # set. This frame's own attention is unaffected on the GPU path too, so it diverges
        # only via what the NEXT frame sees -- exactly the "independently wrong-able and both
        # silent" halves append()'s docstring warns about.
        real_visible = NeuronCameraCache.visible_kv

        def visible_without_pending(self, i, j):
            k, v, _ = real_visible(self, i, j)
            s = self.slot(i, j)
            return k, v, (self._cursor[s] * self.frame_seqlen).to(torch.int32)
        NeuronCameraCache.visible_kv = visible_without_pending
    else:
        raise ValueError(kind)
    print(f"[inject] {kind}", flush=True)


# ---------------------------------------------------------------------------------
class ResetCounter:
    """Count reset() calls on the installed cache without editing it."""

    def __init__(self, cache):
        self.cache = cache
        self.n = 0
        self._real = cache.reset

    def __enter__(self):
        def counting_reset():
            self.n += 1
            return self._real()
        self.cache.reset = counting_reset
        return self

    def __exit__(self, *exc):
        self.cache.reset = self._real
        return False


class FirstAppendWatcher:
    """Record whether any append saw skip_append=True before the first append committed.

    On the GPU, ``_set_skip_append`` before the dicts exist is a no-op, so frame 0 is always
    stored. If the seam ever honours an early skip, C5 stores 0 frames where the GPU stores
    all of them and never re-synchronises.

    Also counts how many appends actually saw the flag set, which is what proves the regime
    exercises the flag at all — see the module docstring on why one regime was not enough.
    """

    def __init__(self, cache):
        self.cache = cache
        self.n_appends = 0
        self.n_skip_appends = 0
        self.early_skip = False
        self._real = type(cache).append

    def __enter__(self):
        watcher = self

        def watched(inner_self, i, j, k, v):
            skipping = inner_self.get_skip_append(i)
            if watcher.n_appends == 0 and skipping:
                watcher.early_skip = True
            watcher.n_appends += 1
            if skipping:
                watcher.n_skip_appends += 1
            return watcher._real(inner_self, i, j, k, v)
        type(self.cache).append = watched
        return self

    def __exit__(self, *exc):
        type(self.cache).append = self._real
        return False


class DecisionRecorder:
    """1 per committed (keyframe) frame, 2 per rolled-back (non-keyframe) frame.

    Same encoding and same mechanism as check_c4_wiring.py: wrap the two cache hooks the
    keyframe decision selects between, so the per-frame decision is observable without
    changing the code under test.
    """

    def __init__(self, model):
        self.model = model
        self.seq: list[int] = []
        self._real_commit = model._execute_deferred_eviction
        self._real_rollback = model._rollback_last_frame

    def __enter__(self):
        def commit():
            self.seq.append(1)
            return self._real_commit()

        def rollback():
            self.seq.append(2)
            return self._real_rollback()
        self.model._execute_deferred_eviction = commit
        self.model._rollback_last_frame = rollback
        return self

    def __exit__(self, *exc):
        self.model._execute_deferred_eviction = self._real_commit
        self.model._rollback_last_frame = self._real_rollback
        return False


def build_images(n_frames, size, device, seed=0):
    """Same generator as check_c4_wiring.py, so the two suites drive identical inputs."""
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


def _same(a: dict, b: dict) -> tuple:
    """Bit-exact over shared keys; returns (all_equal, per-key max abs diff)."""
    keys = sorted(set(a) & set(b))
    diffs, ok = {}, True
    for k in keys:
        if a[k].shape != b[k].shape:
            diffs[k] = f"shape {tuple(a[k].shape)} vs {tuple(b[k].shape)}"
            ok = False
            continue
        eq = torch.equal(a[k], b[k])
        diffs[k] = 0.0 if eq else (a[k] - b[k]).abs().max().item()
        ok = ok and eq
    return (ok and bool(keys)), diffs


def make_cache(model, device, args):
    from lingbot_map.heads.neuron_camera_cache import NeuronCameraCache

    head = model.camera_head
    return NeuronCameraCache(
        num_iterations=head.num_iterations,
        trunk_depth=head.trunk_depth,
        num_heads=head.num_heads,
        head_dim=head.trunk[0].attn.head_dim,
        device=device,
        max_total_frames=args.max_total_frames,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default="/home/kwehden/lingbot-tier1/models/lingbot-map-long.pt")
    # 518 is not a free choice: the checkpoint's patch_embed.pos_embed is [1, 1370, 1024],
    # i.e. (518/14)^2 + 1 patches. See check_c4_wiring.py.
    p.add_argument("--image_size", type=int, default=518)
    p.add_argument("--n_frames", type=int, default=24)
    p.add_argument("--scale_frames", type=int, default=4)
    p.add_argument("--sliding_window", type=int, default=8)
    p.add_argument("--window_size", type=int, default=12)
    p.add_argument("--overlap", type=int, default=4)
    p.add_argument("--flow_threshold", type=float, default=8.0)
    p.add_argument("--max_gap", type=int, default=6)
    # Regime 2. >1 selects the fixed-interval branch, which is the only place
    # _set_skip_append is called (gct_stream_window.py:602).
    p.add_argument("--keyframe_interval", type=int, default=3)
    p.add_argument("--max_total_frames", type=int, default=64)
    p.add_argument("--inject_bug", choices=INJECTIONS, default=None,
                   help="Discriminating-power guard: corrupt the WIRING and confirm this "
                        "suite FAILS. A compare pass that cannot fail proves nothing.")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                 "c5_wiring_results.json"))
    args = p.parse_args()

    if args.inject_bug:
        inject_bug(args.inject_bug)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model, n_missing, n_unexpected = load_model(dev, args)
    images = build_images(args.n_frames, args.image_size, dev)
    print(f"\n=== C5 wiring (device={dev} images={tuple(images.shape)} "
          f"missing={n_missing} unexpected={n_unexpected} inject={args.inject_bug}) ===\n",
          flush=True)

    def streaming_pass():
        model.clean_kv_cache()
        with DecisionRecorder(model) as rec:
            pred = model.inference_streaming(
                images, num_scale_frames=args.scale_frames,
                flow_threshold=args.flow_threshold,
                max_non_keyframe_gap=args.max_gap,
            )
        return _cpu(pred), list(rec.seq)

    def interval_pass():
        """Regime 2: the FIXED-INTERVAL branch, the only one that sets _skip_append.

        flow_threshold=0 disables the flow branch (``use_flow_keyframe`` is False), so the
        loop takes ``gct_stream_window.py:598-602`` and calls ``_set_skip_append(True)`` on
        every non-keyframe. Without this regime the entire skip_append half of the seam is
        untested and two injections pass — see the module docstring.
        """
        model.clean_kv_cache()
        pred = model.inference_streaming(
            images, num_scale_frames=args.scale_frames,
            keyframe_interval=args.keyframe_interval,
            flow_threshold=0.0,
        )
        return _cpu(pred)

    def windowed_pass():
        model.clean_kv_cache()
        pred = model.inference_windowed(
            images, window_size=args.window_size, overlap_size=args.overlap,
            num_scale_frames=args.scale_frames,
            flow_threshold=args.flow_threshold,
            max_non_keyframe_gap=args.max_gap,
        )
        return _cpu(pred)

    with torch.no_grad():
        # ---- stock path, twice: the second run is the run-to-run determinism self-check
        #      that licenses every bit-exact comparison below.
        model.camera_head.install_neuron_cache(None)
        stock_pred, stock_seq = streaming_pass()
        stock_pred2, stock_seq2 = streaming_pass()
        det_ok, det_diffs = _same(stock_pred, stock_pred2)
        emit("model_is_deterministic_run_to_run", det_ok and stock_seq == stock_seq2,
             diffs=det_diffs, decisions_match=stock_seq == stock_seq2,
             detail="Licenses the bit-exact gates below. Without this, a passing comparison "
                    "could be two identically-noisy runs and a failing one could be "
                    "framework nondeterminism attributed to C5")
        stock_win = windowed_pass()
        stock_interval = interval_pass()

        n_kf = sum(1 for d in stock_seq if d == 1)
        n_nkf = sum(1 for d in stock_seq if d == 2)
        emit("decision_mix_is_nontrivial", n_kf > 0 and n_nkf > 0,
             n_keyframe=n_kf, n_non_keyframe=n_nkf,
             detail="Commit-vs-rollback mix on the flow path. NOTE this does NOT imply the "
                    "skip_append flag was exercised -- flow mode never sets it "
                    "(gct_stream_window.py:591-596 commits or rolls back instead). That is "
                    "what regime 2 below is for, and believing otherwise is what let two "
                    "injections pass a 9/9 run")

        # ---- C5 path ------------------------------------------------------------------
        cache = make_cache(model, dev, args)
        model.camera_head.install_neuron_cache(cache)
        emit("cache_installed", model.camera_head.neuron_cache is cache,
             slots=cache.num_slots, plan=str(cache.plan),
             detail="16 independent streams at production; slot = iteration*trunk_depth+block")

        with FirstAppendWatcher(cache) as watch:
            c5_pred, c5_seq = streaming_pass()
        emit("skip_append_not_honoured_before_first_append", not watch.early_skip,
             n_appends=watch.n_appends, early_skip=watch.early_skip,
             detail="The GPU's flag lives in dicts that do not exist until the first causal "
                    "forward and its setter is guarded by kv_cache is not None "
                    "(gct_stream_window.py:365), so an early set is a silent GPU no-op. "
                    "Honouring it here would be a permanent off-by-n")

        pred_ok, pred_diffs = _same(stock_pred, c5_pred)
        emit("predictions_bit_exact", pred_ok and det_ok, diffs=pred_diffs,
             gate="torch.equal", determinism_established=det_ok,
             detail="With no adapter the seam changes only WHERE keys are stored; SDPA then "
                    "sees the same keys in the same order, so bit-exact is the correct gate "
                    "(unlike check_c5_attention_parity.py, where the tile/merge legitimately "
                    "reassociates an fp32 reduction)")
        emit("decision_sequence_identical", stock_seq == c5_seq,
             n_stock=len(stock_seq), n_c5=len(c5_seq),
             first_divergence=next((i for i, (a, b) in enumerate(zip(stock_seq, c5_seq))
                                    if a != b), None),
             detail="One flipped keyframe changes which frames enter the cache, and every "
                    "later frame then attends a different key set (FM8, REQ-025)")

        # ---- the windowed loop: this is the one that calls clean_kv_cache mid-run -------
        with ResetCounter(cache) as rc:
            c5_win = windowed_pass()
        win_ok, win_diffs = _same(stock_win, c5_win)
        emit("windowed_predictions_bit_exact", win_ok, diffs=win_diffs,
             detail="inference_windowed calls clean_kv_cache once per WINDOW, mid-run "
                    "(gct_stream_window.py:1094/:1213). Streaming alone never exercises the "
                    "reset, and the reset is the most dangerous part of the seam")
        emit("window_boundary_resets_cursor", rc.n > 0, n_resets=rc.n,
             detail="If a boundary does not reset, the next window's visible set carries the "
                    "previous window's keys (measured: GPU 3 keys vs C5 8, rel_err 0.90)")
        emit("cache_survives_window_boundary", model.camera_head.neuron_cache is cache,
             still_installed=model.camera_head.neuron_cache is cache,
             detail="clean_kv_cache does `del self.kv_cache`, so holding the cache there got "
                    "it silently orphaned: the next forward rebuilt the list-of-dicts and "
                    "resumed torch.cat with no error and no log line. Hence the separate "
                    "head.neuron_cache attribute")

        # ---- REGIME 2: fixed-interval, the only branch that sets _skip_append -----------
        model.camera_head.install_neuron_cache(cache)
        with FirstAppendWatcher(cache) as watch2:
            c5_interval = interval_pass()
        emit("skip_append_regime_is_exercised", watch2.n_skip_appends > 0,
             n_appends=watch2.n_appends, n_with_flag_set=watch2.n_skip_appends,
             keyframe_interval=args.keyframe_interval,
             detail="Regime 1 (flow) NEVER sets the flag, so this count is the only evidence "
                    "the non-keyframe store path ran at all. If it is 0 the two checks below "
                    "are vacuous and skip_append_ignored/drop_pending_frame pass a green run")
        int_ok, int_diffs = _same(stock_interval, c5_interval)
        emit("interval_predictions_bit_exact", int_ok, diffs=int_diffs,
             detail="The non-keyframe store must be right too: the write lands at the cursor, "
                    "valid_len counts it, and the cursor does NOT advance, so the next "
                    "keyframe overwrites that row (design.md :712-713). Getting the visible "
                    "set right but the commit wrong is invisible on the frame itself and "
                    "diverges one frame later")

    n_fail = sum(1 for r in RESULTS if not r["pass"])
    verdict = "PASS" if n_fail == 0 else "FAIL"
    with open(args.out, "w") as fh:
        json.dump({"suite": "C5 wiring (real GCTStream, stock vs C5 in one process)",
                   "device": str(dev), "injected_bug": args.inject_bug,
                   "config": {k: v for k, v in vars(args).items()},
                   "n_checks": len(RESULTS), "n_fail": n_fail, "verdict": verdict,
                   "results": RESULTS}, fh, indent=2)
    print(f"\n===== {verdict}: {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed "
          f"(inject={args.inject_bug})\n", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
