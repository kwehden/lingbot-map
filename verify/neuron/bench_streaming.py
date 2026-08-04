"""V8: per-frame wall-clock for GCT streaming inference, cold and warm frames separated.

spec/neuron-port/design.md V8 is the objective the whole initiative exists to produce: a
per-frame number on Neuron against a *matched* GPU baseline, on the same fixture at the same
frame count. This file is that measurement. It is ONE script that runs on both devices, which
is the only way "matched" means anything -- two scripts that each print milliseconds are not a
comparison, they are two numbers with a hopeful subtraction between them.

WHY COLD AND WARM MUST BE SEPARATED, AND WHY THAT IS NOT A FORMALITY
--------------------------------------------------------------------
On Neuron the first execution of a shape pays NEFF compilation. Phase 0 measured 8.3 s for a
single trivial graph (design.md). Fold that into a mean over 100 frames and you have added
83 ms/frame of one-time cost to a steady-state number -- enough to invert the comparison's
sign. Fold it into a mean over 10,000 frames and you have hidden it. Neither is the truth, and
which one you get depends on a frame count nobody chose for that reason.

So this script reports:

  * ``cold_ms``      -- every frame up to ``--warmup`` individually, never averaged into warm.
  * ``warm_*``       -- median/p10/p90/mean over the remaining frames.
  * ``compile_ms``   -- an explicit estimate: (cold total) - (warm median x n_cold).

``compile_ms`` is a derived quantity and labelled as one. It is reported because a reader who
sees only "warm median = X" cannot tell whether the port pays 8 s or 800 s to get there, and on
a streaming perception workload the answer decides whether the port is usable at all. It is
measured against the FIRST WARM DECILE rather than the warm median, because per-frame cost is
depth-dependent and a global median would charge the cold frames for cache depth they did not
have -- the first desk run of this script reported -145 ms of "compile" before that was fixed.
A negative value is left unclamped and means "no measurable one-time cost at this scale".

WHY THE MEDIAN, NOT THE MEAN
----------------------------
The mean is the wrong statistic for this workload and reporting it alone would be misleading in
a specific, predictable direction. Per-frame cost is not stationary: the KV cache grows, and on
GPU it grows by ``torch.cat``, so late frames are genuinely slower than early ones. A mean over
a 613-frame sequence therefore reports a cost that occurs at no frame. Both are recorded (mean
is in the JSON) but the headline is the median, and ``warm_p10``/``warm_p90`` plus the
``drift_pct`` field make the growth visible rather than averaged away.

``drift_pct`` is the point of the whole exercise on the Neuron side: C1/C5 replace the growing
``torch.cat`` with a fixed-capacity indexed write, so the Neuron trace should be FLAT where the
GPU trace rises. A port that matched the GPU's median but reproduced its drift would have
failed at its actual purpose while passing a median-only gate.

WHAT MAKES THIS COMPARABLE ACROSS DEVICES
-----------------------------------------
1. ``time.perf_counter`` around a *synchronised* region, not CUDA events. CUDA events are more
   precise but do not exist on Neuron, and a harness that measures the two devices with two
   different clocks has a systematic difference built into it before the first frame runs.
   ``sync()`` dispatches to ``torch.cuda.synchronize`` or ``xm.mark_step``+materialise as
   appropriate; on both devices the timed region ends only once the result is real.
2. Host-to-device transfer is EXCLUDED (moved before the start stamp), because it measures the
   desk's PCIe and the trn2 instance's, not the model.
3. The fixture is the real TUM ``freiburg1_desk`` sequence (REQ-065), preprocessed identically
   by the package's own ``load_and_preprocess_images``. ``--synthetic`` exists for smoke runs
   and is recorded in the JSON so a synthetic number can never be mistaken for a fixture one.
4. Frame count, image size, ``sliding_window``, ``scale_frames``, ``camera_num_iterations`` and
   dtype are all recorded in the output. A number without its configuration is not a result.

REQ-079 CARVE-OUT, ENFORCED RATHER THAN REMEMBERED
--------------------------------------------------
REQ-079 forbids claiming any performance benefit for a configuration that needs the kernel's
dense additive ``position_bias`` path unless that path was actually measured. The dense path is
reached only via ``sliding_window_size > 0`` on the camera head (design.md F4/ODQ3). So this
script records ``dense_mask_path`` in every result, and ``report.py``-style consumers can gate
on it. Passing ``--camera_sliding_window`` sets it True and prints a banner saying the run
carries the carve-out. Nobody has to remember the rule; the artifact carries it.

Run (desk GPU baseline)::

    D="docker exec -w /home/kwehden/lingbot-tier1/lingbot-map lingbot-tier1 python"
    $D verify/neuron/bench_streaming.py --frames 128 --out verify/neuron/bench_gpu_a10g.json

Run (trn2, identical arguments)::

    python verify/neuron/bench_streaming.py --frames 128 --out bench_trn2.json

Then::

    $D verify/neuron/bench_streaming.py --compare bench_gpu_a10g.json bench_trn2.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import statistics
import sys
import time

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)

_TUM_DEFAULT = "/home/kwehden/lingbot-tier1/tum_data/rgbd_dataset_freiburg1_desk/rgb"


# ---------------------------------------------------------------------------------
# Device abstraction. Deliberately tiny: the only thing the timing loop needs from a
# device is "make pending work real, now".
# ---------------------------------------------------------------------------------
def pick_device(requested: str):
    """Return (torch.device, kind) where kind is one of cuda/xla/cpu."""
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif _xla() is not None:
            requested = "xla"
        else:
            requested = "cpu"
    if requested == "xla":
        xm = _xla()
        if xm is None:
            raise RuntimeError("--device xla requested but torch_xla is not importable")
        return xm.xla_device(), "xla"
    return torch.device(requested), requested


def _xla():
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        return xm
    except Exception:
        return None


def make_sync(kind: str, device):
    """Return a callable that blocks until all queued work on `device` has completed.

    This is the load-bearing part of cross-device comparability. On CUDA, kernels are
    queued asynchronously, so a perf_counter delta without a sync measures dispatch, not
    execution. On XLA the graph is not even built until something forces it. Both would
    silently report an absurdly fast number.
    """
    if kind == "cuda":
        def sync(_=None):
            torch.cuda.synchronize(device)
        return sync
    if kind == "xla":
        xm = _xla()

        def sync(tensor=None):
            xm.mark_step()
            # mark_step queues execution; touching a value is what waits for it.
            if tensor is not None:
                _touch(tensor)
            else:
                xm.wait_device_ops()
        return sync

    def sync(_=None):
        return None
    return sync


def _touch(obj):
    """Force materialisation of any tensor reachable in a prediction dict."""
    if torch.is_tensor(obj):
        obj.detach().float().sum().item()
    elif isinstance(obj, dict):
        for v in obj.values():
            _touch(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _touch(v)


# ---------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------
def load_fixture(rgb_dir: str, n_frames: int, image_size: int):
    """Real TUM frames through the package's own preprocessing (REQ-065)."""
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"no *.png under {rgb_dir}")
    if len(paths) < n_frames:
        raise ValueError(
            f"fixture has {len(paths)} frames but --frames {n_frames} was requested. "
            f"Silently looping the sequence would make the KV cache see repeated content and "
            f"is not the same measurement -- pass --frames {len(paths)} or fewer.")
    imgs = load_and_preprocess_images(
        paths[:n_frames], mode="crop", image_size=image_size, patch_size=14)
    return imgs.unsqueeze(0), len(paths)


def build_synthetic(n_frames: int, image_size: int):
    """Smoke-run inputs. Same generator as check_c4_wiring/check_c5_wiring."""
    g = torch.Generator().manual_seed(0)
    base = torch.rand(3, image_size + 64, image_size + 64, generator=g)
    frames = []
    for i in range(n_frames):
        dy, dx = (2 * i) % 64, (i * 13 // 10) % 64
        crop = base[:, dy:dy + image_size, dx:dx + image_size].clone()
        frames.append(crop * (0.9 + 0.1 * torch.cos(torch.tensor(i / 7.0))))
    return torch.stack(frames).unsqueeze(0)


def load_model(args, device):
    from lingbot_map.models.gct_stream_window import GCTStream

    model = GCTStream(
        img_size=args.image_size,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.sliding_window,
        kv_cache_scale_frames=args.scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
    )
    n_missing = n_unexpected = -1
    if args.checkpoint and args.checkpoint != "none":
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_missing, n_unexpected = len(missing), len(unexpected)
    return model.to(device).eval(), n_missing, n_unexpected


def compile_gpu_model(model, mode: str) -> None:
    """Apply the GPU path's own production compilation before timing it.

    THIS IS A FAIRNESS REQUIREMENT, NOT AN OPTIMISATION. The Neuron side necessarily runs
    compiled -- a NEFF is the only way it runs at all. If the GPU baseline runs eager, the
    comparison credits the port with a speedup that is partly just "compiled vs not", and
    REQ-079's spirit (no unmeasured performance claim) is violated by the baseline rather
    than by the subject.

    Targets mirror ``demo.py:compile_model`` / ``gct_profile.py:compile_model``, which is
    what the GPU path actually ships with. ``point_head`` is left alone: gct_profile drops
    it to save ~5.9 ms/frame, but dropping a head on one side of a comparison and not the
    other is exactly the kind of asymmetry this function exists to avoid.

    ON THE MODE, AND WHY THE DEFAULT IS NOT ``reduce-overhead``
    ----------------------------------------------------------
    demo.py and gct_profile.py both use ``mode="reduce-overhead"``, which enables CUDA graph
    trees. Those reuse output buffers between invocations, and this model retains attention
    outputs across frames *in the KV cache*, so a later frame reads a buffer a subsequent
    capture has overwritten:

        RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a
        subsequent run  (raised from block.py:99)

    ``torch.compiler.cudagraph_mark_step_begin()`` before every forward -- which this harness
    does, and which is the documented remedy -- is NOT sufficient here, because the offending
    read is of a tensor the cache kept from an *earlier* step, not of the previous step's
    return value. gct_profile.py does not hit this because it profiles ``gct_stream``, whose
    cache does not retain compiled-region outputs the same way.

    So the default is plain ``torch.compile`` (inductor fusion, no graph capture). That is
    still a compiled baseline and it is the strongest one that runs correctly here; the
    measurement is recorded with its mode so nobody has to guess which was used.
    ``--compile_mode reduce-overhead`` is available if a future container fixes this.
    """
    agg = model.aggregator
    kw = {} if mode in ("", "default") else {"mode": mode}
    for i, b in enumerate(agg.frame_blocks):
        agg.frame_blocks[i] = torch.compile(b, **kw)
    for i, b in enumerate(agg.patch_embed.blocks):
        agg.patch_embed.blocks[i] = torch.compile(b, **kw)
    for b in agg.global_blocks:
        if hasattr(b, "attn_pre"):
            b.attn_pre = torch.compile(b.attn_pre, **kw)
        if hasattr(b, "ffn_residual"):
            b.ffn_residual = torch.compile(b.ffn_residual, **kw)
        b.attn.proj = torch.compile(b.attn.proj, **kw)


def install_c5(model, device, args):
    """Route the camera head through C5, exactly as check_c5_wiring.py does."""
    from lingbot_map.heads.neuron_camera_cache import NeuronCameraCache

    head = model.camera_head
    cache = NeuronCameraCache(
        num_iterations=head.num_iterations,
        trunk_depth=head.trunk_depth,
        num_heads=head.num_heads,
        head_dim=head.trunk[0].attn.head_dim,
        device=device,
        max_total_frames=args.max_frame_num + 100,
    )
    head.install_neuron_cache(cache)
    return cache


# ---------------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------------
@torch.no_grad()
def run(model, images, args, device, sync):
    """Drive the streaming loop by hand, timing each frame in isolation.

    This reimplements the loop rather than calling ``inference_streaming`` for one reason:
    ``inference_streaming`` accumulates every frame's predictions and returns them in one
    dict, so there is no seam at which to stamp a per-frame time. The loop below is the
    fixed-interval branch of ``gct_stream_window.py:598-612`` -- the same branch, including
    the ``_set_skip_append`` handoff, so what is timed is what production runs.
    """
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    autocast = (torch.amp.autocast("cuda", dtype=dtype)
                if (args.dtype != "fp32" and device.type == "cuda")
                else _null())

    S = images.shape[1]
    scale_frames = min(args.scale_frames, S)
    kf_int = max(int(args.keyframe_interval), 1)

    model.clean_kv_cache()
    sync()

    # -- Scale prefill. Timed separately and NEVER folded into per-frame: it is a
    #    different shape (num_frame_per_block=scale_frames), so on Neuron it is a
    #    different NEFF, and on GPU it is a different amount of work.
    scale_batch = images[:, :scale_frames].to(device=device, dtype=dtype)
    sync()
    _mark_step()
    t0 = time.perf_counter()
    with autocast:
        out = model.forward(scale_batch, num_frame_for_scale=scale_frames,
                            num_frame_per_block=scale_frames, causal_inference=True)
    sync(out)
    prefill_ms = (time.perf_counter() - t0) * 1e3
    del scale_batch, out
    print(f"  prefill ({scale_frames} scale frames): {prefill_ms:.1f} ms", flush=True)

    per_frame_ms: list[float] = []
    for i in range(scale_frames, S):
        is_keyframe = (kf_int <= 1) or ((i - scale_frames) % kf_int == 0)
        if not is_keyframe:
            model._set_skip_append(True)

        # Transfer OUTSIDE the timed region: this measures the host link, not the model.
        frame = images[:, i:i + 1].to(device=device, dtype=dtype)
        sync()
        _mark_step()

        t0 = time.perf_counter()
        with autocast:
            out = model.forward(frame, num_frame_for_scale=scale_frames,
                                num_frame_per_block=1, causal_inference=True)
        sync(out)
        ms = (time.perf_counter() - t0) * 1e3

        if not is_keyframe:
            model._set_skip_append(False)
        per_frame_ms.append(ms)
        del frame, out

        if args.progress and (i - scale_frames) % args.progress == 0:
            print(f"    frame {i:>5d}: {ms:8.2f} ms", flush=True)

    return per_frame_ms, prefill_ms, scale_frames


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _mark_step() -> None:
    """Tell CUDA-graph trees a new invocation is starting.

    Required under ``--compile``: ``mode="reduce-overhead"`` uses CUDA graph trees, which
    reuse output buffers between invocations. Reading a previous frame's output after the
    next capture raises "accessing tensor output of CUDAGraphs that has been overwritten"
    -- which is exactly what this loop does, since it calls ``sync(out)`` to materialise
    each frame before stamping the time. ``demo.py`` and ``gct_profile.py`` both call this
    before every forward for the same reason. It is placed OUTSIDE the timed region because
    it is harness bookkeeping, not model work.
    """
    torch.compiler.cudagraph_mark_step_begin()


def summarize(per_frame_ms, prefill_ms, scale_frames, n_cold):
    """Split cold from warm and report the median, not just the mean. See the docstring."""
    n = len(per_frame_ms)
    n_cold = min(n_cold, n)
    cold = per_frame_ms[:n_cold]
    warm = per_frame_ms[n_cold:]
    if not warm:
        raise ValueError(
            f"--warmup {n_cold} consumed all {n} streaming frames; there is no warm "
            f"population left to report. Raise --frames or lower --warmup.")

    warm_median = statistics.median(warm)
    srt = sorted(warm)
    p10 = srt[max(0, int(0.10 * len(srt)) - 1)]
    p90 = srt[min(len(srt) - 1, int(0.90 * len(srt)))]

    # Drift: does per-frame cost grow with cache depth? First vs last decile of warm.
    dec = max(1, len(warm) // 10)
    first_dec = statistics.median(warm[:dec])
    last_dec = statistics.median(warm[-dec:])
    drift_pct = (last_dec - first_dec) / first_dec * 100.0 if first_dec else 0.0

    # One-time cost estimate: what the cold frames cost ABOVE the frames immediately
    # after them. The reference is the FIRST warm decile, not the warm median, and that
    # choice is load-bearing rather than stylistic. Per-frame cost grows with cache depth
    # (see drift_pct), so subtracting a *global* median charges the cold frames for depth
    # they never had: the first desk smoke run reported -145 ms of "compile" that way,
    # which is not a compile cost, it is the depth trend leaking into the estimator.
    # Using the depth-adjacent reference isolates the one-time component.
    #
    # A NEGATIVE value still means something and is not clamped: it says the cold frames
    # were cheaper than their neighbours, i.e. there is no measurable one-time cost at
    # this scale. That is the expected GPU answer, and on Neuron it would be the signal
    # that compilation was cached (or never happened -- worth checking against V4(b)).
    compile_ms_est = sum(cold) - first_dec * len(cold)

    return {
        "n_streaming_frames": n,
        "n_scale_frames": scale_frames,
        "prefill_ms": prefill_ms,
        "n_cold": len(cold),
        "cold_ms": [round(x, 3) for x in cold],
        "compile_ms_estimate": compile_ms_est,
        "warm_median_ms": warm_median,
        "warm_mean_ms": statistics.fmean(warm),
        "warm_p10_ms": p10,
        "warm_p90_ms": p90,
        "warm_min_ms": srt[0],
        "warm_max_ms": srt[-1],
        "warm_fps_median": 1000.0 / warm_median if warm_median else 0.0,
        "drift_first_decile_ms": first_dec,
        "drift_last_decile_ms": last_dec,
        "drift_pct": drift_pct,
        "warm_ms": [round(x, 3) for x in warm],
    }


def print_summary(s, label):
    print(f"\n  === {label} ===")
    print(f"    prefill            : {s['prefill_ms']:9.2f} ms "
          f"({s['n_scale_frames']} scale frames, separate shape -- not per-frame)")
    print(f"    cold frames ({s['n_cold']})    : "
          + ", ".join(f"{x:.1f}" for x in s["cold_ms"][:8])
          + (" ..." if len(s["cold_ms"]) > 8 else ""))
    print(f"    one-time estimate  : {s['compile_ms_estimate']:9.2f} ms "
          f"(derived: cold total - first-warm-decile x n_cold; "
          f"{'no measurable one-time cost' if s['compile_ms_estimate'] <= 0 else 'see docstring'})")
    print(f"    warm median        : {s['warm_median_ms']:9.2f} ms  "
          f"-> {s['warm_fps_median']:6.2f} FPS   <-- headline")
    print(f"    warm p10 / p90     : {s['warm_p10_ms']:9.2f} / {s['warm_p90_ms']:.2f} ms")
    print(f"    warm mean          : {s['warm_mean_ms']:9.2f} ms "
          f"(reported for completeness; median is the headline -- cost is not stationary)")
    print(f"    drift 1st->last dec: {s['drift_first_decile_ms']:.2f} -> "
          f"{s['drift_last_decile_ms']:.2f} ms  ({s['drift_pct']:+.1f}%)")


def compare(paths):
    """Side-by-side of two or more result files, with the matching preconditions checked.

    A comparison across runs whose configuration differs is not a result, so the
    configuration keys that would invalidate it are checked and any mismatch is printed as
    a NOT COMPARABLE line rather than quietly folded into a ratio.
    """
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append((os.path.basename(p), json.load(f)))

    # input_hw, not image_size: crop preprocessing makes the effective shape differ from
    # the requested one (TUM -> 392x518, synthetic -> 518x518), so --image_size agreeing
    # is not evidence the two runs did the same amount of work.
    must_match = ("frames", "input_hw", "scale_frames", "sliding_window",
                  "camera_num_iterations", "keyframe_interval", "dtype", "fixture")
    base_cfg = runs[0][1]["config"]
    mismatched = []
    for name, r in runs[1:]:
        for k in must_match:
            if r["config"].get(k) != base_cfg.get(k):
                mismatched.append((name, k, base_cfg.get(k), r["config"].get(k)))

    print(f"\n{'=' * 78}\n  V8 comparison\n{'=' * 78}")
    if mismatched:
        print("  *** NOT COMPARABLE -- configuration differs: ***")
        for name, k, a, b in mismatched:
            print(f"      {name}: {k} = {b!r}, baseline has {a!r}")
        print("  Ratios below are printed but MUST NOT be reported as a speedup.\n")
    else:
        print("  configuration matches on: " + ", ".join(must_match) + "\n")

    if any(r["config"].get("dense_mask_path") for _, r in runs):
        print("  *** REQ-079 CARVE-OUT: at least one run used the dense position_bias path.")
        print("      No performance claim may ride on this run for the -1 configuration.\n")

    names = [n for n, _ in runs]
    col = max(14, max(len(n) for n in names) + 2)
    rows = [("warm median ms", "warm_median_ms"), ("warm FPS", "warm_fps_median"),
            ("warm p10 ms", "warm_p10_ms"), ("warm p90 ms", "warm_p90_ms"),
            ("warm mean ms", "warm_mean_ms"), ("prefill ms", "prefill_ms"),
            ("compile est ms", "compile_ms_estimate"), ("drift %", "drift_pct")]
    print(f"  {'metric':<22s}" + "".join(f"{n:>{col}s}" for n in names))
    print("  " + "-" * (22 + col * len(names)))
    for label, key in rows:
        vals = "".join(f"{r['summary'][key]:>{col}.2f}" for _, r in runs)
        print(f"  {label:<22s}{vals}")

    # The compile flag is deliberately NOT in must_match: a compiled GPU baseline vs a
    # NEFF-compiled Neuron run is the correct pairing, and requiring the flags to be equal
    # would forbid it. What must never happen is an EAGER GPU baseline being called slower
    # than a compiled Neuron run, so that specific asymmetry is named.
    cuda_eager = [n for n, r in runs
                  if r["config"].get("device_kind") == "cuda"
                  and not r["config"].get("torch_compiled")]
    has_neuron = any(r["config"].get("device_kind") == "xla" for _, r in runs)
    if cuda_eager and has_neuron:
        print("  *** BASELINE IS EAGER: " + ", ".join(cuda_eager) + " ran without "
              "torch.compile,\n      while the Neuron side necessarily ran compiled. Any "
              "speedup below is partly\n      'compiled vs not' and is NOT a port result. "
              "Re-run the GPU side with --compile.\n")

    # prefill_ms and compile_ms_estimate are NOT comparable across runs unless every run
    # had an equally warm compilation cache, and nothing in the config records that.
    #
    # This is not hypothetical. The first V8 pair showed stock prefill 17,015 ms vs C5
    # 5,781 ms and looked like C5 making prefill 3x faster. It was the inductor cache
    # warming between the two runs: re-running stock afterwards gave 5,819 ms, i.e. the
    # same as C5. Whoever reads the table next should not have to know that story.
    if any(r["config"].get("torch_compiled") for _, r in runs) and len(runs) > 1:
        print("  NOTE: 'prefill ms' and 'compile est ms' are NOT comparable across runs "
              "here.\n        A compiled run populates a persistent inductor/NEFF cache, so "
              "whichever run\n        went FIRST pays for both. (Measured: stock prefill "
              "17.0 s cold vs 5.8 s warm --\n        the same shape, the same code.) Compare "
              "these only between runs you know\n        started from the same cache state. "
              "The warm medians are unaffected.\n")

    if len(runs) == 2 and not mismatched:
        a, b = runs[0][1]["summary"], runs[1][1]["summary"]
        ratio = a["warm_median_ms"] / b["warm_median_ms"] if b["warm_median_ms"] else 0.0
        licensed = not (cuda_eager and has_neuron)
        print(f"\n  {names[1]} warm median is {ratio:.2f}x "
              f"{'FASTER' if ratio > 1 else 'SLOWER'} than {names[0]}")
        print(f"  (medians, warm only; prefill and one-time cost excluded by construction)")
        if not licensed:
            print(f"  UNLICENSED: see the eager-baseline warning above. Do not report this.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare", nargs="+", metavar="RESULTS.json",
                   help="Compare existing result files instead of running.")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "xla", "cpu"))
    p.add_argument("--checkpoint",
                   default="/home/kwehden/lingbot-tier1/models/lingbot-map-long.pt",
                   help="'none' for random weights (architecture-only timing).")
    p.add_argument("--rgb_dir", default=_TUM_DEFAULT,
                   help="TUM freiburg1_desk rgb/ directory (REQ-065 fixture).")
    p.add_argument("--synthetic", action="store_true",
                   help="Smoke inputs instead of the fixture. Recorded in the output so a "
                        "synthetic number cannot be mistaken for a fixture one.")
    p.add_argument("--frames", type=int, default=128)
    # 518 is not free: the checkpoint's patch_embed.pos_embed is [1, 1370, 1024],
    # i.e. (518/14)^2 + 1. See check_c4_wiring.py.
    p.add_argument("--image_size", type=int, default=518)
    p.add_argument("--scale_frames", type=int, default=8)
    p.add_argument("--sliding_window", type=int, default=64)
    p.add_argument("--max_frame_num", type=int, default=1024)
    p.add_argument("--camera_num_iterations", type=int, default=4)
    p.add_argument("--keyframe_interval", type=int, default=1)
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    p.add_argument("--use_sdpa", type=int, default=1,
                   help="1 = SDPA (flashinfer is absent from the desk container).")
    p.add_argument("--warmup", type=int, default=4,
                   help="Frames treated as COLD. Never averaged into the warm statistics; "
                        "reported individually. On Neuron this is where NEFF compile lands.")
    p.add_argument("--install_c5", action="store_true",
                   help="Route the camera head through NeuronCameraCache (C5).")
    p.add_argument("--compile", action="store_true",
                   help="Apply the GPU path's production torch.compile before timing. "
                        "FAIRNESS, not optimisation: Neuron necessarily runs compiled, so an "
                        "eager GPU baseline credits the port with 'compiled vs not'. Recorded "
                        "in the output; compare() refuses to call it a speedup if the two "
                        "runs disagree on this flag.")
    p.add_argument("--compile_mode", default="default",
                   choices=("default", "reduce-overhead", "max-autotune"),
                   help="See compile_gpu_model's docstring: 'reduce-overhead' (what demo.py "
                        "ships) crashes on this model under a retained KV cache, so the "
                        "default here is plain inductor.")
    p.add_argument("--camera_sliding_window", type=int, default=-1,
                   help="Camera-head sliding_window_size. Anything > 0 takes the dense "
                        "additive position_bias path and triggers the REQ-079 carve-out.")
    p.add_argument("--progress", type=int, default=0,
                   help="Print every Nth frame's time (0 = silent).")
    p.add_argument("--label", default=None)
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                 "bench_streaming_results.json"))
    args = p.parse_args()

    if args.compare:
        return compare(args.compare)

    device, kind = pick_device(args.device)
    sync = make_sync(kind, device)
    dense = args.camera_sliding_window > 0
    if dense:
        print("*** REQ-079: camera_sliding_window > 0 selects the dense additive "
              "position_bias path.\n    This run's numbers carry the carve-out and may "
              "not back a claim for the -1 config.\n")

    torch.manual_seed(0)
    if args.synthetic:
        images, n_avail = build_synthetic(args.frames, args.image_size), args.frames
        fixture = "synthetic"
    else:
        images, n_avail = load_fixture(args.rgb_dir, args.frames, args.image_size)
        fixture = os.path.basename(os.path.dirname(args.rgb_dir.rstrip("/")))

    model, n_missing, n_unexpected = load_model(args, device)
    if args.install_c5:
        install_c5(model, device, args)
    if args.compile:
        if kind != "cuda":
            raise SystemExit(
                "--compile applies the CUDA path's torch.compile targets and is meaningless "
                "on Neuron, where compilation is the NEFF. Drop it for the trn2 run; the "
                "comparison records the flag on both sides so the asymmetry stays visible.")
        compile_gpu_model(model, args.compile_mode)

    label = args.label or (
        f"{kind}:{'C5' if args.install_c5 else 'stock'}"
        f"{':compiled' if args.compile else ''}:{args.dtype}")
    dev_name = (torch.cuda.get_device_name(device) if kind == "cuda"
                else (str(device) if kind == "xla" else platform.processor() or "cpu"))

    print(f"\n=== V8 streaming benchmark ===")
    print(f"  device     : {dev_name} ({kind})")
    print(f"  fixture    : {fixture} ({args.frames} of {n_avail} frames available)")
    print(f"  images     : {tuple(images.shape)}  dtype={args.dtype}")
    print(f"  weights    : {'random' if n_missing < 0 else f'ckpt (missing={n_missing} unexpected={n_unexpected})'}")
    print(f"  camera path: {'C5 NeuronCameraCache' if args.install_c5 else 'stock torch.cat'}")
    print(f"  cold frames: {args.warmup} (reported separately, never averaged into warm)\n",
          flush=True)

    per_frame_ms, prefill_ms, scale_frames = run(model, images, args, device, sync)
    s = summarize(per_frame_ms, prefill_ms, scale_frames, args.warmup)
    print_summary(s, label)

    result = {
        "label": label,
        "config": {
            "device_kind": kind, "device_name": dev_name,
            "frames": args.frames, "image_size": args.image_size,
            "scale_frames": args.scale_frames, "sliding_window": args.sliding_window,
            "max_frame_num": args.max_frame_num,
            "camera_num_iterations": args.camera_num_iterations,
            "keyframe_interval": args.keyframe_interval, "dtype": args.dtype,
            "use_sdpa": bool(args.use_sdpa), "fixture": fixture,
            # The EFFECTIVE shape, not just --image_size. `mode="crop"` preprocessing
            # yields 392x518 from the TUM 480x640 aspect while --synthetic gives 518x518,
            # so two runs can agree on --image_size and still be timing different amounts
            # of work. This is the field compare() actually gates on.
            "input_hw": list(images.shape[-2:]),
            "checkpoint_loaded": n_missing >= 0,
            "n_missing_keys": n_missing, "n_unexpected_keys": n_unexpected,
            "install_c5": bool(args.install_c5),
            "torch_compiled": bool(args.compile),
            "compile_mode": args.compile_mode if args.compile else None,
            "camera_sliding_window": args.camera_sliding_window,
            # REQ-079: carried in the artifact so the rule does not depend on memory.
            "dense_mask_path": dense,
            "warmup": args.warmup,
            "torch": torch.__version__,
        },
        "summary": s,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  wrote {args.out}")
    if fixture == "synthetic":
        print("  NOTE: synthetic inputs. Not a V8 result -- REQ-065 requires the TUM fixture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
