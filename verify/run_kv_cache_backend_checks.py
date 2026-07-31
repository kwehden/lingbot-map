"""Tier 2 verification harness for the KVCacheBackend refactor (TASK-025).

Lives under verify/, not lingbot_map/ -- per spec/design.md's explicit note that this
is verification-harness code outside the refactor's own file-list/Simplicity Budget.

Invoked by LingBotMapSSI/skypilot/kv_cache_backend_verify_job.yaml's run: block:
    python verify/run_kv_cache_backend_checks.py --mode baseline \
        --checkpoint /tmp/models/lingbot-map-long.pt \
        --tum_root /tmp/tum_data --scene rgbd_dataset_freiburg1_desk \
        --output_dir "$OUTPUT_DIR"
    python verify/run_kv_cache_backend_checks.py --mode compare \
        --checkpoint /tmp/models/lingbot-map-long.pt \
        --tum_root /tmp/tum_data --scene rgbd_dataset_freiburg1_desk \
        --baseline_dir /tmp/baseline --output_dir "$OUTPUT_DIR"

Implements spec/design.md's Verification Strategy table, restricted to the items that
need a real checkpoint + the real TUM fixture (items 1, 2, 6, 7, 8 -- item 3's keyframe-
decision-sequence check and item 9's torch.compile check are TASK-023/TASK-020's own
scope and run separately, not duplicated here; item 4's no-FlashInfer-dependency check
and item 5's grep check need no real checkpoint/fixture at all and already run on Tier 1
via TASK-017/TASK-018):

  1. demo.py-shaped end-to-end parity: streaming + windowed, both backends.
  2. benchmark harness parity: LingbotMapMethod.process_scene, both modes, both backends.
  6. get_kv_cache_info() byte-identical, both backends (including the FlashInfer-side
     preserved bug).
  7. Gap A preserved: CameraCausalHead.rollback_last_frame does not exist.
  8. Gap B preserved: SDPA _defer_eviction leak still reproduces.

baseline mode must run against the UNMODIFIED pre-refactor code -- this script imports
lingbot_map/benchmark lazily (inside functions, not at module scope) precisely so it is
importable and runnable against a pre-refactor checkout too.

Writes a single JSON results file ($OUTPUT_DIR/results.json: per-check pass/fail + noise
floors in baseline mode, or per-check pass/fail + diffs in compare mode) and raw
prediction tensors (baseline mode only, via torch.save, so compare mode has something to
diff against). Exits nonzero iff any check in results.json failed.
"""

import argparse
import json
import sys
from pathlib import Path


# Per TASK-003's derived defaults (spec/design.md Decision 4): rtol/atol for bf16 paths
# vs. the tighter force_fp32 path. Compare mode gates at max(default, 3 x measured
# baseline noise floor) -- the noise floor itself is measured and recorded by this
# script's own baseline-mode run, immediately below.
_DEFAULT_TOL = {
    "bf16": {"rtol": 1e-3, "atol": 1e-5},
    "fp32": {"rtol": 1e-5, "atol": 1e-6},
}

_OUTPUT_KEYS = ["pose_enc", "depth", "depth_conf", "world_points", "world_points_conf"]

# inference_windowed emits three MORE outputs that _OUTPUT_KEYS does not name, and that
# earlier revisions of this harness therefore never diffed: the cross-window alignment
# metadata attached by _merge_and_align (gct_stream_window.py:951-955).
#
# These are the numerics MOST exposed to a KV-cache refactor in windowed mode -- they are
# derived from _pairwise_alignment over each window's overlap region, so they depend on
# per-window cache state directly, whereas pose_enc/depth are per-frame. Leaving them
# ungated meant windowed parity could pass while cross-window alignment silently drifted.
#
# alignment_mode is a STRING ("scaled"), not a tensor -- _values_close below dispatches on
# type rather than assuming tensors, so it is compared by equality.
#
# Deliberately NOT included: `images`. Both inference paths echo the input images back
# (gct_stream_window.py:659) as a visualization payload; it is model *input*, not a
# prediction, so a diff there could only ever report a harness bug, at ~1.6GB of tensor
# comparison per config.
_ALIGNMENT_KEYS = ["chunk_scales", "chunk_transforms", "alignment_mode"]

# Discrete per-frame decision outputs, present in BOTH inference paths and likewise never
# previously diffed. Found by inspecting a real staged baseline artifact rather than by
# reading the return annotations -- both are emitted but neither appears in _OUTPUT_KEYS.
#
#   is_keyframe  [B, S] bool  -- whether each frame's KV was retained in cache
#   frame_type   [B, S] uint8 -- scale-frame vs. sliding-window frame classification
#
# These matter more than their size suggests: they ARE the keyframe-decision sequence, and
# the refactor moves exactly the machinery that produces them (_set_skip_append,
# _defer_eviction, rollback_last_frame, execute_deferred_eviction). A refactor that changed
# which frames get cached would alter these while potentially leaving pose_enc/depth within
# tolerance -- a silent behavioral change in the cache policy itself.
#
# This is design.md's Verification Strategy item 3 (keyframe-decision-sequence identity),
# which the module docstring defers to TASK-023 as a separate run. The staged baseline
# already contains these tensors, so gating them here costs nothing and closes item 3 for
# the two inference entry points this harness drives. It does NOT subsume all of TASK-023,
# which also covers flow-threshold-driven decisions this harness never exercises.
#
# Compared EXACTLY, not at rtol/atol: a keyframe decision is a discrete choice, so "close"
# is not a meaningful relaxation -- one flipped frame is a real divergence.
_EXACT_KEYS = ["is_keyframe", "frame_type"]

# What compare mode actually diffs. Streaming configs do not emit the alignment keys, so
# for them these land in the absent-in-both branch (parity, close=True) -- which also
# usefully guards the reverse direction: if a refactor ever started emitting alignment
# metadata from the streaming path, present-in-one-only would flag it.
_COMPARED_KEYS = _OUTPUT_KEYS + _ALIGNMENT_KEYS + _EXACT_KEYS

# (mode, use_sdpa) combinations item 1/2 both require -- FlashInfer is the default
# (use_sdpa=False); force_fp32 is FlashInfer-only per context.md's Glossary, so it is
# only exercised as a third configuration, not crossed with mode.
_CONFIGS = [
    ("streaming", False),
    ("streaming", True),
    ("windowed", False),
    ("windowed", True),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_demo_images(tum_root: Path, scene: str, max_frames: int = None):
    """Load a frame list from the TUM scene as a preprocessed [S, 3, H, W] tensor,
    matching demo.py's own load_images() shape exactly (same crop/resize/patch-align
    preprocessing via lingbot_map.utils.load_fn.load_and_preprocess_images).

    Deliberately bypasses benchmark.datasets.tum.TumDataset (which associates GT poses
    this script doesn't need for a pure prediction-parity check) and just reads rgb.txt
    directly for the frame path list.
    """
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    rgb_txt = tum_root / scene / "rgb.txt"
    lines = [
        line.strip() for line in rgb_txt.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if max_frames is not None:
        lines = lines[:max_frames]

    paths = [str(tum_root / scene / line.split()[1]) for line in lines]
    images = load_and_preprocess_images(paths, mode="crop", image_size=518, patch_size=14)
    return images


def _build_model(mode: str, use_sdpa: bool, checkpoint: str, device: str):
    """Instantiate + load a GCTStream model, mirroring benchmark/methods/lingbot_map.py's
    own _load_model() exactly (same defaults), so this script's models match what the
    real benchmark harness constructs.
    """
    import torch

    if mode == "windowed":
        from lingbot_map.models.gct_stream_window import GCTStream
    else:
        from lingbot_map.models.gct_stream import GCTStream

    model = GCTStream(
        img_size=518,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=1024,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=8,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=use_sdpa,
    )
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Report load fidelity explicitly. strict=False silently tolerates a checkpoint that
    # does not match the architecture, which for a BASELINE would mean staging predictions
    # from a partly-randomly-initialized model and diffing everything else against them --
    # a false reference nobody would notice. Print the counts (mirroring
    # benchmark/methods/lingbot_map.py's own diagnostics) and fail loudly on wholesale
    # mismatch rather than producing a quietly worthless baseline.
    tag = f"{mode}/{'sdpa' if use_sdpa else 'flashinfer'}"
    print(f"[_build_model {tag}] checkpoint={checkpoint} "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    total = len(state_dict)
    if len(missing) > 0.5 * max(total, 1):
        raise RuntimeError(
            f"Checkpoint load looks wholesale-mismatched for {tag}: {len(missing)} missing "
            f"keys against a {total}-entry state_dict. Refusing to produce a baseline from "
            f"a mostly-uninitialized model. Verify the checkpoint matches this architecture "
            f"(TASK-001 only ever validated it under use_sdpa=True)."
        )
    return model.to(device).eval()


def _run_inference(model, mode: str, images, device=None, force_fp32: bool = False):
    """Run one inference pass, mirroring gct_profile.py's dtype handling exactly.

    gct_profile.py (the codebase's own bf16-vs-fp32 accuracy harness) is the reference:
    the model's weights stay fp32 as loaded, the INPUT images are cast to the run's dtype,
    bf16 runs use autocast and fp32 runs use a null context, and for the FlashInfer
    backend specifically fp32 additionally requires
    `aggregator.kv_cache_force_fp32 = True` -- FlashInfer's FA2 kernel only supports
    fp16/bf16, so that flag switches it to a dense gather + fp32 SDPA path
    (flashinfer_cache.py's compute_attention).

    Getting this wrong is what failed baseline job 4: disabling autocast without also
    casting the inputs left fp32 activations meeting bf16 intermediates and raised
    "mat1 and mat2 must have the same dtype, but got BFloat16 and Float" in attn.proj.
    """
    import contextlib

    import torch

    if device is None:
        device = next(model.parameters()).device

    dtype = torch.float32 if force_fp32 else (
        torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )
    images = images.to(device=device, dtype=dtype)

    if force_fp32:
        # FlashInfer-only knob; harmless no-op attribute on the SDPA path, but this
        # configuration is only ever run against FlashInfer (see _CONFIGS comment).
        model.aggregator.kv_cache_force_fp32 = True

    autocast_ctx = (
        contextlib.nullcontext() if force_fp32
        else torch.amp.autocast("cuda", dtype=dtype)
    )
    with torch.no_grad(), autocast_ctx:
        if mode == "streaming":
            predictions = model.inference_streaming(
                images, num_scale_frames=8, keyframe_interval=1,
                output_device=torch.device("cpu"),
            )
        else:
            predictions = model.inference_windowed(
                images, window_size=16, overlap_size=4, num_scale_frames=8,
                keyframe_interval=1, output_device=torch.device("cpu"),
            )
    return predictions


def _run_benchmark_harness(mode: str, use_sdpa: bool, checkpoint: str, device: str,
                            tum_root: Path, scene: str):
    """Item 2: run through the actual benchmark.methods.lingbot_map.LingbotMapMethod,
    not demo.py-shaped direct calls -- exercises _resolve_keyframe_interval's "auto" path
    and output_device=torch.device("cpu") offloading.

    Import convention matches run.py/prepare.py's own: the OUTER benchmark/ directory
    (containing run.py, datasets/, configs/) goes on sys.path, so `datasets.tum` and
    `methods.lingbot_map` resolve top-level, while `benchmark.core.*` resolves via the
    nested benchmark/benchmark/ framework package (confirmed by TASK-002's own finding).
    """
    outer_benchmark = _repo_root() / "benchmark"
    if str(outer_benchmark) not in sys.path:
        sys.path.insert(0, str(outer_benchmark))
    from methods.lingbot_map import LingbotMapMethod
    from datasets.tum import TumDataset
    from benchmark.core.storage import BSSArtifact
    import prepare as prepare_mod

    method = LingbotMapMethod(
        checkpoint=checkpoint, device=device, mode=mode, use_sdpa=use_sdpa,
        enable_3d_rope=True, keyframe_interval="auto", logger=_null_logger(),
    )

    # BSSLoader expects a real BSSArtifact (gt/ directory with rgb/ files already
    # staged by prepare.py); rather than re-implement its resize/loading logic here,
    # drive prepare.py's own prepare_scene() once against a real BSSArtifact.
    workspace = tum_root.parent / f"_verify_workspace_{scene}"
    gt_artifact = BSSArtifact(workspace / "tum" / scene / "gt")
    if not gt_artifact.is_complete():
        dataset = TumDataset(str(tum_root), scenes=[scene])
        prepare_mod.prepare_scene(gt_artifact, scene, dataset, logger=_null_logger())

    output = method.process_scene(gt_artifact)
    return output


class _null_logger:
    """Drop-in no-op logger for functions that expect a logging.Logger interface."""
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass


def _check_get_kv_cache_info(device: str) -> dict:
    """Item 6: get_kv_cache_info() byte-identity across a few synthetic cache states,
    both backends. No checkpoint needed -- these are Python ints/floats, not model
    predictions.

    *** The `get_kv_cache_info` half of this check is VACUOUS FOR FLASHINFER. ***
    get_kv_cache_info reads only `aggregator.kv_cache` (gct_stream_window.py:430-448),
    the SDPA dict. FlashInfer state lives in `aggregator.kv_cache_manager`, which that
    method never touches, and `_init_kv_cache` populates the k_i/v_i keys only under
    `if self.use_sdpa:` (stream.py:185-193). For use_sdpa=False the dict is `{}` -- not
    None, so the early return does not fire -- giving num_cached=0 and 0.0 MB. Both
    recorded FlashInfer states are therefore {0, 0.0}, before AND after inference.
    (Confirmed on hardware by baseline job 16, which recorded exactly that.)

    That zero-return is Decision 3's documented, deliberately-preserved bug, so the
    `states` half of this check ASSERTS THE BUG rather than measuring cache stats. On its
    own it is a tautology for FlashInfer: comparing {0,0.0} to {0,0.0} passes even if a
    refactor completely breaks FlashInfer cache accounting.

    *** `flashinfer_cache_stats` is the non-vacuous companion gate. ***
    `FlashInferKVCacheManager.get_cache_stats` (flashinfer_cache.py:303 pre-refactor,
    :324 post-refactor) DOES return live per-block occupancy -- frame_count, scale_pages,
    live_pages, free_pages, special_tokens. It exists with an identical signature and
    identical return keys on both refs, so its output is directly diffable across the
    refactor. Recorded here for every block, after inference, so compare mode has a real
    equality gate on FlashInfer accounting instead of only the zero-return assertion.

    This is a harness-side probe: it calls an existing public read-only method and does
    not modify lingbot_map/, so it is compatible with the requirement that these runs
    exercise unmodified source. The SDPA leg deliberately does NOT get the same treatment
    -- pre-refactor SDPA has no get_cache_stats at all (it is a bare dict), so there is no
    cross-ref-comparable call to make; its get_kv_cache_info numbers are already live and
    non-vacuous (job 16: 24 blocks / 25.78 MB), which is why only FlashInfer needed this.

    Note also the SDPA leg runs fp32 (autocast disabled below is CPU-only, but the model
    is built without .to(dtype)), while the memory estimate hardcodes 2 bytes/element, so
    cache_memory_mb is a stable-but-wrong constant. Fine for a byte-identity check.

    On Tier 1 the use_sdpa=False iteration RAISES rather than skipping, since FlashInfer
    cannot construct there -- this function is Tier-2-only despite recording both legs.
    """
    import torch
    from lingbot_map.models.gct_stream_window import GCTStream

    results = {}
    for use_sdpa in (True, False):
        model = GCTStream(
            img_size=98, patch_size=14, enable_3d_rope=True, max_frame_num=64,
            use_sdpa=use_sdpa, kv_cache_sliding_window=64, kv_cache_scale_frames=1,
            camera_num_iterations=1,
        ).to(device).eval()
        states = [model.get_kv_cache_info()]
        images = torch.rand(1, 5, 3, 98, 98, device=device)
        # Must run under the SAME autocast the production path uses
        # (_run_inference below, and demo.py). The model here is built in fp32, but
        # FlashInferKVCache coerces its storage dtype to bf16 for an fp32 request
        # (flashinfer_cache.py:125-127, since the FA2 kernel is fp16/bf16-only) and its
        # paged branch returns kernel output in that storage dtype without casting back
        # to q.dtype the way the fp32-gather branch does (:383 vs :417-421). Without
        # autocast the bf16 attention output then hits an fp32 attn.proj and raises
        # "mat1 and mat2 must have the same dtype, but got BFloat16 and Float".
        # This does not perturb the recorded values, but NOT because bf16 matches the
        # memory estimate's assumption -- see the "vacuous for FlashInfer" note in the
        # docstring above. Autocast does not stop the bf16 return; it makes the consumer
        # (attn.proj, an autocast-eligible nn.Linear) accept it.
        # enabled=False on CPU because FlashInfer cannot construct on CPU at all
        # (flashinfer_cache.py raises when unavailable), so the use_sdpa=False leg cannot
        # reach the mismatch there, and CPU bf16 autocast would perturb the SDPA leg's
        # arithmetic for no benefit. Uses .startswith so a "cuda:1"-style device string
        # does not silently disable autocast and reintroduce the crash.
        with torch.no_grad(), torch.amp.autocast(
            device, dtype=torch.bfloat16, enabled=str(device).startswith("cuda")
        ):
            model.inference_streaming(images, num_scale_frames=1, keyframe_interval=1)
        states.append(model.get_kv_cache_info())
        key = "sdpa" if use_sdpa else "flashinfer"
        results[key] = states

        # Real FlashInfer occupancy, to escape the tautology documented above. Guarded
        # rather than assumed: the manager is lazily constructed (stream.py:207-215), so a
        # future change that stopped constructing it must surface as an explicit
        # unavailable marker -- which will FAIL compare mode's equality gate against a
        # baseline that recorded real numbers -- rather than silently vanish from results.
        if not use_sdpa:
            mgr = getattr(model.aggregator, "kv_cache_manager", None)
            if mgr is None or not hasattr(mgr, "get_cache_stats"):
                results["flashinfer_cache_stats"] = {"unavailable": True}
            else:
                results["flashinfer_cache_stats"] = {
                    str(i): mgr.get_cache_stats(block_idx=i)
                    for i in range(mgr.num_blocks)
                }
    return results


def _check_gap_a() -> bool:
    """Item 7: CameraCausalHead.rollback_last_frame must not exist."""
    from lingbot_map.models.gct_stream_window import GCTStream
    model = GCTStream(embed_dim=64, use_sdpa=True)
    return not hasattr(model.camera_head, "rollback_last_frame")


def _check_gap_b() -> bool:
    """Item 8: SDPA _defer_eviction leak must still reproduce (assert the bug).

    Recipe matches TASK-015's own validated repro exactly (small conv-embed synthetic
    model, inference_streaming with flow_threshold high enough to force every
    post-scale frame to be judged a non-keyframe and rolled back).
    """
    import torch
    from lingbot_map.models.gct_stream_window import GCTStream

    model = GCTStream(
        img_size=42, patch_size=14, embed_dim=64, patch_embed='conv',
        enable_camera=True, enable_point=False, enable_local_point=False,
        enable_depth=False, use_sdpa=True,
        kv_cache_sliding_window=1, kv_cache_scale_frames=1,
    )
    images = torch.rand(1, 6, 3, 42, 42)
    with torch.no_grad():
        model.inference_streaming(
            images, num_scale_frames=1, keyframe_interval=1,
            flow_threshold=1e9, max_non_keyframe_gap=1000,
        )
    return model.aggregator.kv_cache.get("k_0_special") is not None


def _free_cuda() -> None:
    """Release cached CUDA memory between full-fixture model instances -- this script
    builds several 613-frame-scale models back to back in one process (4 configs x 2
    phases in baseline mode alone), which OOMs a single A10G (22GB) unless callers `del`
    their own model/tensor references before calling this.
    """
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _values_close(va, vb, tol: dict, exact: bool = False) -> dict:
    """Compare one prediction value, dispatching on type.

    Most values are tensors, but inference_windowed's `alignment_mode` is a plain string,
    and a tensor-only implementation would raise AttributeError on `.float()` the moment
    the alignment keys were added to the compared set. Non-tensor values are compared by
    equality -- for a mode flag, exact equality is the correct gate anyway: a refactor
    that silently switched windowed inference from "scaled" to any other alignment mode is
    a divergence no numeric tolerance should absorb.

    `exact=True` (used for _EXACT_KEYS) demands element-wise equality instead of allclose.
    Necessary for the bool/uint8 decision tensors: allclose on a bool tensor would compare
    True against 1.0009 as "close", which is meaningless for a discrete decision, and the
    count of differing frames is the useful diagnostic rather than a max-abs magnitude.
    """
    import torch

    if not (torch.is_tensor(va) and torch.is_tensor(vb)):
        return {"close": bool(va == vb), "non_tensor": True,
                "baseline_value": str(va), "compare_value": str(vb)}

    if exact:
        if va.shape != vb.shape:
            return {"close": False, "shape_mismatch": True, "exact": True,
                    "baseline_shape": list(va.shape), "compare_shape": list(vb.shape)}
        ne = (va != vb)
        n_diff = int(ne.sum().item())
        detail = {"close": n_diff == 0, "exact": True, "num_differing": n_diff,
                  "num_elements": int(va.numel())}
        if n_diff:
            # Name the first few offending frame indices -- for a decision sequence, WHICH
            # frames flipped is the whole diagnostic, and a bare count would send the
            # reader back to the raw tensors.
            detail["first_differing_indices"] = (
                ne.nonzero(as_tuple=False)[:10].tolist())
        return detail

    # Shape mismatch is a real divergence, and allclose would raise on non-broadcastable
    # shapes rather than report. chunk_scales/chunk_transforms are shaped by the *window
    # count*, so a refactor that changed windowing would land here -- exactly the class of
    # bug this gate is being added to catch.
    if va.shape != vb.shape:
        return {"close": False, "shape_mismatch": True,
                "baseline_shape": list(va.shape), "compare_shape": list(vb.shape)}

    ta, tb = va.float(), vb.float()
    diff = (ta - tb).abs()
    denom = tb.abs()
    # Decision 4 step 2 asks for max absolute AND max relative difference per key; earlier
    # revisions recorded only max_abs. Relative is computed only where the denominator is
    # nonzero -- 0/0 is parity, not infinite error.
    nonzero = denom > 0
    max_rel = (diff[nonzero] / denom[nonzero]).max().item() if nonzero.any() else 0.0
    return {
        "close": bool(torch.allclose(ta, tb, rtol=tol["rtol"], atol=tol["atol"])),
        "max_abs_diff": diff.max().item(),
        "max_rel_diff": max_rel,
    }


def _tensor_dict_close(a: dict, b: dict, tol: dict, keys=None) -> dict:
    """Per-key parity between two prediction dicts.

    _OUTPUT_KEYS lists every key the model *can* emit, but which are actually present
    depends on the model config: world_points/world_points_conf only exist when
    enable_point=True, and the production pipeline (benchmark/methods/lingbot_map.py and
    demo.py) leaves enable_point at its False default, so those two keys are legitimately
    absent from both runs. Likewise the _ALIGNMENT_KEYS are windowed-only. A key absent
    from BOTH is parity -- same before and after the refactor -- so it counts as
    close=True. A key present in one run but not the other IS a real divergence (the
    refactor added or dropped an output) and stays close=False, so callers'
    `all(v["close"] for ...)` aggregation still catches that case.

    `keys` defaults to _COMPARED_KEYS (named outputs + windowed alignment metadata).
    """
    diffs = {}
    for key in (keys if keys is not None else _COMPARED_KEYS):
        in_a, in_b = key in a, key in b
        if not in_a and not in_b:
            diffs[key] = {"close": True, "absent_both": True}
            continue
        if in_a != in_b:
            diffs[key] = {
                "close": False,
                "present_baseline": in_a,
                "present_compare": in_b,
            }
            continue
        diffs[key] = _values_close(a[key], b[key], tol, exact=key in _EXACT_KEYS)
    return diffs


def run_baseline(args) -> dict:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tum_root = Path(args.tum_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"mode": "baseline", "checks": {}}

    # Every stage below costs real Tier 2 GPU minutes, so each one is flushed to
    # results.json the moment it completes rather than only at the end. Baseline job 10
    # computed all five noise floors correctly, then died in the item-2 harness loop on a
    # missing `plyfile` import -- and because results.json was written only after every
    # stage, 33 minutes of correct GPU work was discarded. Checkpointing per stage means a
    # late-stage failure now costs only the stages that had not yet finished. `results` is
    # also printed, so baseline_run.log carries the numbers even if the S3 sync itself is
    # what fails.
    def _checkpoint(stage: str) -> None:
        (output_dir / "results.json").write_text(json.dumps(results, indent=2))
        print(f"[run_baseline] checkpointed results.json after stage: {stage}", flush=True)

    images = _load_demo_images(tum_root, args.scene)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    # Item 1: demo.py-shaped end-to-end, all 4 mode x backend combinations, plus a
    # noise-floor measurement (run twice back-to-back) per TASK-003's procedure.
    noise_floors = {}
    for mode, use_sdpa in _CONFIGS:
        model = _build_model(mode, use_sdpa, args.checkpoint, device)
        run_a = _run_inference(model, mode, images)
        run_b = _run_inference(model, mode, images)
        tag = f"{mode}_{'sdpa' if use_sdpa else 'flashinfer'}"
        tol = _DEFAULT_TOL["bf16"]
        diff = _tensor_dict_close(run_a, run_b, tol)
        noise_floors[tag] = {
            k: v["max_abs_diff"] for k, v in diff.items() if "max_abs_diff" in v
        }
        print(f"[run_baseline] noise_floor[{tag}] = {noise_floors[tag]}", flush=True)
        torch.save(run_a, predictions_dir / f"demo_{tag}.pt")
        results["checks"]["noise_floors"] = {"pass": True, "values": noise_floors}
        _checkpoint(f"noise_floor:{tag}")
        del model, run_a, run_b
        _free_cuda()

    # force_fp32 FlashInfer noise floor (item 1's third configuration).
    model = _build_model("streaming", False, args.checkpoint, device)
    run_a = _run_inference(model, "streaming", images, device, force_fp32=True)
    run_b = _run_inference(model, "streaming", images, device, force_fp32=True)
    diff = _tensor_dict_close(run_a, run_b, _DEFAULT_TOL["fp32"])
    noise_floors["streaming_flashinfer_fp32"] = {
        k: v["max_abs_diff"] for k, v in diff.items() if "max_abs_diff" in v
    }
    print("[run_baseline] noise_floor[streaming_flashinfer_fp32] = "
          f"{noise_floors['streaming_flashinfer_fp32']}", flush=True)
    torch.save(run_a, predictions_dir / "demo_streaming_flashinfer_fp32.pt")
    results["checks"]["noise_floors"] = {"pass": True, "values": noise_floors}
    _checkpoint("noise_floor:streaming_flashinfer_fp32")
    del model, run_a, run_b
    _free_cuda()

    # Item 2: benchmark harness parity, both modes x backends.
    harness_recorded = []
    for mode, use_sdpa in _CONFIGS:
        tag = f"{mode}_{'sdpa' if use_sdpa else 'flashinfer'}"
        output = _run_benchmark_harness(mode, use_sdpa, args.checkpoint, device,
                                         tum_root, args.scene)
        torch.save(output, predictions_dir / f"harness_{tag}.pt")
        harness_recorded.append(tag)
        results["checks"]["harness_recorded"] = {
            "pass": True, "recorded": list(harness_recorded)
        }
        _checkpoint(f"harness:{tag}")
        del output
        _free_cuda()

    # Item 6: get_kv_cache_info byte-identity states (recorded, not compared yet --
    # compare mode diffs against these).
    kv_info_states = _check_get_kv_cache_info(device)
    kv_info_states = {
        k: [{kk: vv for kk, vv in state.items()} for state in v]
        for k, v in kv_info_states.items()
    }
    results["checks"]["kv_cache_info_states"] = {"pass": True, "values": kv_info_states}
    _checkpoint("kv_cache_info_states")

    # Items 7/8: Gap A/B preserved, recorded as booleans (must remain the same value
    # in compare mode).
    # Flushed separately: sharing one checkpoint means a failure in gap B discards a
    # completed gap A, which is the same loss mode the per-stage flush exists to prevent.
    results["checks"]["gap_a_preserved"] = {"pass": True, "value": _check_gap_a()}
    _checkpoint("gap_a_preserved")
    results["checks"]["gap_b_preserved"] = {"pass": True, "value": _check_gap_b()}
    _checkpoint("gap_b_preserved")

    _checkpoint("complete")
    return results


def run_compare(args) -> dict:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tum_root = Path(args.tum_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = Path(args.baseline_dir)

    baseline_results = json.loads((baseline_dir / "results.json").read_text())
    baseline_predictions_dir = baseline_dir / "predictions"

    results = {"mode": "compare", "checks": {}}
    all_pass = True

    # Compare mode gets the same per-stage flush baseline mode has. It did not before: the
    # checkpointing added after job 10 lost 33 GPU-minutes was applied only to run_baseline,
    # while run_compare still wrote results.json exactly once, at the very end -- so any
    # late failure discarded every completed comparison. That is the identical loss mode,
    # and compare mode is the run whose *findings* (not just measurements) would be lost.
    #
    # `pass` is written on every flush so a partial results.json is still interpretable,
    # with `complete: False` marking that not all stages ran. A reader must not mistake a
    # partial all_pass=True for a clean run.
    def _flush(stage: str, complete: bool = False) -> None:
        results["pass"] = all_pass
        results["complete"] = complete
        (output_dir / "results.json").write_text(json.dumps(results, indent=2))
        print(f"[run_compare] checkpointed results.json after stage: {stage} "
              f"(all_pass so far: {all_pass})", flush=True)

    # Cheap stages FIRST. Items 6/7/8 build only tiny synthetic models (98px and 42px) and
    # cost seconds; items 1/2 cost ~40 GPU-minutes. Running the cheap ones last is how job
    # 15 died: it burned the full inference budget and then crashed in the item-6 stage,
    # losing everything. Front-loading them means a defect in that code path surfaces in the
    # first minute, and a spot preemption during the expensive stages still leaves the cheap
    # verdicts already flushed to S3.
    #
    # Ordering is safe: these three stages share no state with items 1/2 -- separate models,
    # no checkpoint, no fixture -- so the only coupling is GPU memory, and each frees its
    # own before returning.

    # Item 6: get_kv_cache_info byte-identity vs baseline's recorded states.
    #
    # Compared per sub-key rather than with a single `==` on the whole dict. This harness
    # now records `flashinfer_cache_stats` alongside the two `get_kv_cache_info` state
    # lists, and the staged baseline (job 16) predates that probe, so a whole-dict equality
    # would report FAIL purely because the compare side carries an extra key -- a spurious
    # failure that says nothing about the refactor. Sub-key comparison also localizes a
    # genuine failure to the backend that caused it.
    kv_info_states = _check_get_kv_cache_info(device)

    # A dedicated pre-refactor probe run (--mode kv_info_probe against the baseline ref)
    # can supply baseline values for keys the staged baseline never recorded. Falls back to
    # the staged baseline's own values when absent.
    baseline_kv_states = dict(
        baseline_results["checks"]["kv_cache_info_states"]["values"])
    probe_source = "staged_baseline"
    if args.kv_info_baseline:
        probe_path = Path(args.kv_info_baseline)
        if probe_path.is_file():
            probe = json.loads(probe_path.read_text())
            probe_values = probe.get("kv_cache_info_states", probe)
            for k, v in probe_values.items():
                baseline_kv_states.setdefault(k, v)
            probe_source = f"{probe_path.name}+staged_baseline"
        else:
            probe_source = "staged_baseline (probe file absent)"

    kv_detail = {}
    kv_pass = True
    for subkey, compare_value in kv_info_states.items():
        if subkey not in baseline_kv_states:
            # Recorded for the NEXT baseline to compare against, but explicitly not
            # counted as a pass -- there is nothing to compare it to this run. Not counted
            # as a failure either: an absent baseline is a coverage gap, not evidence of a
            # regression, and failing here would make an unrelated refactor look broken.
            kv_detail[subkey] = {
                "pass": None, "no_baseline": True, "compare": compare_value,
                "note": ("no baseline recorded for this sub-check; value stored as a "
                         "forward baseline. NOT verified this run."),
            }
            continue
        match = compare_value == baseline_kv_states[subkey]
        kv_pass = kv_pass and match
        kv_detail[subkey] = {
            "pass": bool(match),
            "baseline": baseline_kv_states[subkey],
            "compare": compare_value,
        }
    all_pass = all_pass and kv_pass
    results["checks"]["kv_cache_info_byte_identical"] = {
        "pass": bool(kv_pass),
        "baseline_source": probe_source,
        "per_subcheck": kv_detail,
        # Surfaced at the top level so a reader of results.json cannot mistake a passing
        # `pass` for full coverage when some sub-checks had no baseline.
        "unverified_subchecks": [
            k for k, v in kv_detail.items() if v.get("no_baseline")],
    }
    _flush("kv_cache_info_byte_identical")
    _free_cuda()

    # Items 7/8: Gap A/B preservation must match baseline's recorded value exactly.
    gap_a = _check_gap_a()
    gap_a_pass = gap_a == baseline_results["checks"]["gap_a_preserved"]["value"]
    all_pass = all_pass and gap_a_pass
    results["checks"]["gap_a_preserved"] = {"pass": bool(gap_a_pass), "value": gap_a}
    _flush("gap_a_preserved")

    gap_b = _check_gap_b()
    gap_b_pass = gap_b == baseline_results["checks"]["gap_b_preserved"]["value"]
    all_pass = all_pass and gap_b_pass
    results["checks"]["gap_b_preserved"] = {"pass": bool(gap_b_pass), "value": gap_b}
    _flush("gap_b_preserved")
    _free_cuda()

    images = _load_demo_images(tum_root, args.scene)

    # Item 1: diff demo.py-shaped predictions against staged baseline at
    # max(default, 3 x measured noise floor).
    demo_check = {}
    for mode, use_sdpa in _CONFIGS:
        model = _build_model(mode, use_sdpa, args.checkpoint, device)
        run = _run_inference(model, mode, images)
        tag = f"{mode}_{'sdpa' if use_sdpa else 'flashinfer'}"
        baseline_pred = torch.load(
            baseline_predictions_dir / f"demo_{tag}.pt", weights_only=False)
        noise = baseline_results["checks"]["noise_floors"]["values"].get(tag, {})
        tol = {
            "rtol": _DEFAULT_TOL["bf16"]["rtol"],
            "atol": max(
                _DEFAULT_TOL["bf16"]["atol"],
                3 * max(noise.values()) if noise else _DEFAULT_TOL["bf16"]["atol"],
            ),
        }
        diff = _tensor_dict_close(baseline_pred, run, tol)
        passed = all(v.get("close", False) for v in diff.values())
        all_pass = all_pass and passed
        demo_check[tag] = {"pass": passed, "diff": diff}
        results["checks"]["demo_parity"] = demo_check
        _flush(f"demo_parity:{tag}")
        del model, run, baseline_pred
        _free_cuda()
    results["checks"]["demo_parity"] = demo_check

    # force_fp32 configuration.
    model = _build_model("streaming", False, args.checkpoint, device)
    run = _run_inference(model, "streaming", images, device, force_fp32=True)
    baseline_pred = torch.load(
        baseline_predictions_dir / "demo_streaming_flashinfer_fp32.pt", weights_only=False)
    noise = baseline_results["checks"]["noise_floors"]["values"].get(
        "streaming_flashinfer_fp32", {})
    tol = {
        "rtol": _DEFAULT_TOL["fp32"]["rtol"],
        "atol": max(
            _DEFAULT_TOL["fp32"]["atol"],
            3 * max(noise.values()) if noise else _DEFAULT_TOL["fp32"]["atol"],
        ),
    }
    diff = _tensor_dict_close(baseline_pred, run, tol)
    passed = all(v.get("close", False) for v in diff.values())
    all_pass = all_pass and passed
    results["checks"]["demo_parity"]["streaming_flashinfer_fp32"] = {
        "pass": passed, "diff": diff,
    }
    _flush("demo_parity:streaming_flashinfer_fp32")
    del model, run, baseline_pred
    _free_cuda()

    # Item 2: benchmark harness parity.
    harness_check = {}
    for mode, use_sdpa in _CONFIGS:
        tag = f"{mode}_{'sdpa' if use_sdpa else 'flashinfer'}"
        output = _run_benchmark_harness(mode, use_sdpa, args.checkpoint, device,
                                         tum_root, args.scene)
        # process_scene() returns numpy arrays inside a nested dict, which torch 2.6+'s
        # weights_only=True default refuses to unpickle -- these are our own artifacts
        # written by this script's baseline run, so opt out explicitly.
        baseline_output = torch.load(
            baseline_predictions_dir / f"harness_{tag}.pt", weights_only=False)
        # This is a bf16 harness run, so it gets the bf16 tolerance widened by this
        # config's own measured noise floor -- NOT whatever `tol` the fp32 block above
        # happened to leave bound.
        harness_noise = baseline_results["checks"]["noise_floors"]["values"].get(tag, {})
        harness_tol = {
            "rtol": _DEFAULT_TOL["bf16"]["rtol"],
            "atol": max(
                _DEFAULT_TOL["bf16"]["atol"],
                3 * max(harness_noise.values()) if harness_noise
                else _DEFAULT_TOL["bf16"]["atol"],
            ),
        }
        # LingbotMapMethod.process_scene() returns {'frame': {...lists...}, 'global': {}};
        # compare depth lists element-wise at the same tolerance as demo_parity.
        b_depth = baseline_output["frame"]["depth"]
        c_depth = output["frame"]["depth"]
        depth_close = len(b_depth) == len(c_depth) and all(
            abs(bd - cd).max()
            <= harness_tol["atol"] + harness_tol["rtol"] * abs(bd).max()
            for bd, cd in zip(b_depth, c_depth)
        )
        harness_check[tag] = {"pass": bool(depth_close), "tol": harness_tol}
        all_pass = all_pass and depth_close
        results["checks"]["harness_parity"] = harness_check
        _flush(f"harness_parity:{tag}")
        del output, baseline_output
        _free_cuda()
    results["checks"]["harness_parity"] = harness_check

    # Items 6/7/8 already ran, at the TOP of this function -- see the "Cheap stages FIRST"
    # note there. They are deliberately not repeated here.

    _flush("complete", complete=True)
    return results


def run_kv_info_probe(args) -> dict:
    """Record ONLY the item-6 cache-state block, against whatever ref is checked out.

    Exists so the new `flashinfer_cache_stats` gate can get a pre-refactor baseline without
    re-running the full ~40-minute baseline: this path builds only tiny synthetic 98px
    models and needs no checkpoint or TUM fixture, so it costs GPU seconds, not minutes.
    Run it against the baseline ref, keep the JSON, and feed it to compare mode via
    --kv_info_baseline.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values = _check_get_kv_cache_info(device)
    payload = {"mode": "kv_info_probe", "kv_cache_info_states": values}
    (output_dir / "kv_info_probe.json").write_text(json.dumps(payload, indent=2))
    print(f"[run_kv_info_probe] {json.dumps(values, indent=2)}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",
                        choices=["baseline", "compare", "kv_info_probe"], required=True)
    # Not required for kv_info_probe: that mode builds only tiny synthetic models and
    # touches neither the checkpoint nor the fixture.
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tum_root", default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_dir", default=None,
                         help="Required for --mode compare")
    parser.add_argument("--kv_info_baseline", default=None,
                         help="Optional kv_info_probe.json from the baseline ref, "
                              "supplying baseline values for item-6 sub-checks the "
                              "staged baseline predates (e.g. flashinfer_cache_stats).")
    args = parser.parse_args()

    if args.mode == "compare" and not args.baseline_dir:
        parser.error("--baseline_dir is required for --mode compare")
    if args.mode in ("baseline", "compare"):
        missing = [n for n in ("checkpoint", "tum_root", "scene")
                   if getattr(args, n) is None]
        if missing:
            parser.error(
                f"--mode {args.mode} requires: {', '.join('--' + m for m in missing)}")

    if args.mode == "kv_info_probe":
        run_kv_info_probe(args)
        sys.exit(0)
    elif args.mode == "baseline":
        results = run_baseline(args)
        # Baseline mode has no pass/fail semantics of its own (per TASK-027's own
        # framing) -- it always "succeeds" if it completes and records numbers.
        sys.exit(0)
    else:
        results = run_compare(args)
        sys.exit(0 if results["pass"] else 1)


if __name__ == "__main__":
    main()
