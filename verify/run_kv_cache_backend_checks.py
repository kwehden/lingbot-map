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

`--mode self_test` needs none of that -- no torch, no checkpoint, no fixture, no GPU. It
exercises the tolerance derivation (_derive_tol, spec/design.md Decision 4 step 2 / step 5)
against hand-built noise floors, including the legacy abs-only artifact format, and exits
nonzero if any assertion fails:
    python verify/run_kv_cache_backend_checks.py --mode self_test --output_dir /tmp/st

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
# baseline noise floor), per half -- see _derive_tol. The noise floor itself is measured and
# recorded by this script's own baseline-mode run, immediately below.
_DEFAULT_TOL = {
    "bf16": {"rtol": 1e-3, "atol": 1e-5},
    "fp32": {"rtol": 1e-5, "atol": 1e-6},
}

# The point past which widening rtol stops producing a gate. rtol = 1.0 is "within 100% of
# the baseline", which no comparison can fail, so a measured relative floor whose 3x reaches
# it is refused rather than widened from -- see _derive_tol's fourth section.
_RTOL_WIDENING_CEILING = 1.0

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


def _derive_tol(defaults: dict, floor_entry: dict) -> dict:
    """Decision 4 step 5's gate -- max(default, 3 x measured noise floor) -- applied to the
    matching half of the floor per tolerance: `atol` widens from the ABSOLUTE floor, `rtol`
    from the RELATIVE one.

    `floor_entry` is one configuration's entry out of baseline mode's recorded
    `noise_floors`: {output_key: {"max_abs_diff": float, "max_rel_diff": float}}. An empty
    entry (no baseline recorded for that configuration) yields the defaults unchanged.

    Pure -- no torch, no I/O, no globals -- so `--mode self_test` can exercise every branch
    on CPU without a checkpoint, a fixture or a GPU.

    *** Why this is not one max() over one number. ***
    Earlier revisions widened `atol` by `3 x max(noise.values())` and left `rtol` pinned at
    its default, which the tolerance artifact flagged as diverging from Decision 4. The
    obvious repair -- feed the same max() into `rtol` too -- is a UNIT ERROR: `noise_floors`
    recorded max_abs_diff only, so for any key with an absolute floor above
    3.3e-4 that would push the bf16 `rtol` past its own 1e-3 default and LOOSEN the gate,
    while appearing to follow the spec. atol and rtol are widened from different measured
    quantities or not at all.

    *** Legacy artifacts: refuse, and say so. ***
    A baseline written before this change records a bare float per key (the absolute floor;
    _values_close computed the relative half and noise_floors discarded it, and it is not
    recoverable after the fact -- only the first of the two back-to-back runs is saved).
    There is then no relative floor to widen from, so `rtol` stays at the step-4 default and
    the refusal plus the keys that caused it are recorded in the returned dict, which
    callers write into results.json. A run against a legacy baseline therefore states which
    quantity it declined to widen from instead of silently substituting the wrong one.

    ONE legacy case is not a refusal, because the missing half is entailed rather than
    guessed: an absolute floor of exactly 0.0 means the two back-to-back runs matched
    element for element, so max|a-b| / |b| is 0.0 as well. That is what all five recorded
    configurations measured, so the recorded verdict derives as the plain defaults with
    nothing refused. A legacy entry with a NONZERO absolute floor carries no such
    entailment, and that is the one that refuses.

    *** A floor is only usable if it is finite, non-negative, and small enough to gate. ***
    Widening had no upper bound and no finiteness check, and the widened quantity is the
    thing that decides a pass. `_values_close` excludes only `denom == 0`, not denom close
    to zero, so a single denormal baseline element yields an enormous `max_rel_diff`; a
    relative floor of `1e11` derived `rtol = 3e11`, and one of `inf` derived `rtol = inf`,
    at which `torch.allclose` passes ANY two tensors. The gate would have been recorded as
    a PASS while being incapable of failing -- and `inf`/`nan` also make results.json
    non-strict JSON (`Infinity`, `NaN`), which breaks the `jq` roll-ups that read it.
    Before the relative half existed `rtol` was pinned, so this failure mode is one this
    change introduced, and the run that first writes a nonzero-floor baseline is exactly
    the run that would arm it.

    So a floor that is non-finite or negative is not measurement, it is a broken
    measurement, and it is refused rather than widened from -- as is any relative floor
    whose 3x widening would reach `_RTOL_WIDENING_CEILING`, a relative tolerance of 1.0,
    i.e. "within 100% of the baseline", which no comparison can fail. Every refusal holds
    the half at its step-4 default, which can only make the gate STRICTER: the safe
    direction is a run that fails and is looked at, never one that passes on a tolerance
    nobody chose. `atol` gets the finiteness and sign checks but no ceiling, because its
    scale is the data's and this file has no scale-free bound to impose on it; what it gets
    instead is `atol_widening_factor`, so the loosening is visible in the artifact.
    """
    abs_floors = []
    rel_floors = []
    abs_only_keys = []
    unusable = []

    def _usable(value, key, half):
        """A floor we can widen from, or a record of why not. Order matters: the `!=`
        self-comparison is the nan test, and nan fails every inequality silently."""
        if value != value or value in (float("inf"), float("-inf")):
            unusable.append({"key": key, "half": half, "value": repr(value),
                             "why": "not finite"})
            return False
        if value < 0.0:
            unusable.append({"key": key, "half": half, "value": repr(value),
                             "why": "negative -- a max of absolute differences cannot be"})
            return False
        return True

    for key in sorted(floor_entry):
        entry = floor_entry[key]
        if isinstance(entry, dict):
            abs_floor = entry.get("max_abs_diff")
            rel_floor = entry.get("max_rel_diff")
        else:
            abs_floor, rel_floor = entry, None
        if abs_floor is not None:
            abs_floor = float(abs_floor)
            if _usable(abs_floor, key, "max_abs_diff"):
                abs_floors.append(abs_floor)
            else:
                abs_floor = None          # and so it entails nothing about the other half
        if rel_floor is None and abs_floor == 0.0:
            rel_floor = 0.0
        if rel_floor is not None:
            rel_floor = float(rel_floor)
            if _usable(rel_floor, key, "max_rel_diff"):
                rel_floors.append(rel_floor)
            else:
                rel_floor = None
        if rel_floor is None:
            abs_only_keys.append(key)

    tol = {
        "rtol": defaults["rtol"],
        "atol": defaults["atol"],
        "abs_floor_max": max(abs_floors) if abs_floors else None,
        "rel_floor_max": None,
        "rtol_widened_from_relative_floor": False,
        "atol_widening_factor": 1.0,
    }
    if unusable:
        tol["unusable_floors"] = unusable
        tol["unusable_floors_reason"] = (
            "these recorded floors are not finite non-negative numbers, so they are a "
            "broken measurement rather than a measured noise floor and nothing is widened "
            "from them. The half stays at its Decision 4 step-4 default, which can only "
            "make the gate stricter. Re-measure; a non-finite relative floor usually means "
            "_values_close divided by a near-zero baseline element."
        )
    if abs_floors:
        tol["atol"] = max(defaults["atol"], 3 * max(abs_floors))
        tol["atol_widening_factor"] = tol["atol"] / defaults["atol"]
    if abs_only_keys:
        tol["rtol_widening_refused"] = True
        tol["rtol_refusal_keys"] = abs_only_keys
        tol["rtol_refusal_reason"] = (
            "baseline noise_floors records an absolute floor only for these keys, so no "
            "relative floor exists to widen rtol from. rtol held at the Decision 4 step-4 "
            "default rather than widened by 3 x an absolute difference, which is a unit "
            "error that would loosen the gate. Re-measure the baseline to widen rtol."
        )
    elif rel_floors:
        tol["rel_floor_max"] = max(rel_floors)
        widened = 3 * max(rel_floors)
        if widened >= _RTOL_WIDENING_CEILING:
            tol["rtol_widening_refused"] = True
            tol["rtol_refusal_keys"] = [k for k in sorted(floor_entry)
                                        if _rel_of(floor_entry[k]) == max(rel_floors)]
            tol["rtol_refusal_reason"] = (
                "3 x this relative floor is {:g}, at or above a relative tolerance of "
                "{:g} -- 'within {:g}% of the baseline', which no comparison can fail. A "
                "measured floor that large is a broken measurement, not a loose gate: "
                "rtol is held at the step-4 default so the run FAILS and is looked at."
                .format(widened, _RTOL_WIDENING_CEILING, 100 * _RTOL_WIDENING_CEILING)
            )
        else:
            tol["rtol"] = max(defaults["rtol"], widened)
            tol["rtol_widened_from_relative_floor"] = tol["rtol"] > defaults["rtol"]
    return tol


def _rel_of(entry):
    """The relative half of one noise_floors entry, or None for the legacy bare float."""
    return entry.get("max_rel_diff") if isinstance(entry, dict) else None


def _is_strict_json(obj) -> bool:
    """Whether this would survive into results.json as JSON a strict parser accepts.

    `json.dumps` emits bare `Infinity`/`NaN` by default -- valid Python, not valid JSON -- and
    the roll-ups over results.json are `jq`. A derived tolerance that cannot be serialised
    strictly is a defect in the tolerance, so this reports rather than raises.
    """
    try:
        return "Infinity" not in json.dumps(obj, allow_nan=False) \
            and "NaN" not in json.dumps(obj, allow_nan=False)
    except (ValueError, TypeError):
        return False


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
        # BOTH halves of Decision 4 step 2's floor, per key. Recording only max_abs_diff
        # (as earlier revisions did) discards the relative floor irrecoverably -- only
        # run_a is saved, so it cannot be recomputed later -- and leaves compare mode with
        # nothing dimensionally valid to widen rtol from. See _derive_tol.
        noise_floors[tag] = {
            k: {"max_abs_diff": v["max_abs_diff"], "max_rel_diff": v["max_rel_diff"]}
            for k, v in diff.items() if "max_abs_diff" in v
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
        k: {"max_abs_diff": v["max_abs_diff"], "max_rel_diff": v["max_rel_diff"]}
        for k, v in diff.items() if "max_abs_diff" in v
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
            # A probe recorded from THIS checkout is the refactor comparing against itself
            # -- guaranteed to pass and worth nothing. The probe records its own
            # module_root precisely so this is detectable; refuse rather than silently
            # manufacture a self-referential baseline.
            probe_root = probe.get("module_root")
            if probe_root:
                import lingbot_map
                here = str(Path(lingbot_map.__file__).resolve().parent.parent)
                if str(Path(probe_root).resolve()) == here:
                    raise RuntimeError(
                        f"--kv_info_baseline was recorded from the SAME checkout this "
                        f"compare run is using ({here!r}). That would compare the refactor "
                        f"against itself and pass unconditionally. Record the probe from a "
                        f"clone of the baseline ref, with PYTHONPATH set to that clone."
                    )
            probe_values = probe.get("kv_cache_info_states", probe)
            for k, v in probe_values.items():
                baseline_kv_states.setdefault(k, v)
            probe_source = f"{probe_path.name}({probe_root})+staged_baseline"
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
        tol = _derive_tol(_DEFAULT_TOL["bf16"], noise)
        diff = _tensor_dict_close(baseline_pred, run, tol)
        passed = all(v.get("close", False) for v in diff.values())
        all_pass = all_pass and passed
        # `tol` carries the derivation's own provenance (which floor widened what, and any
        # refusal against a legacy abs-only baseline), so the gate this config was scored
        # at is readable from results.json instead of inferred from the code revision.
        demo_check[tag] = {"pass": passed, "tol": tol, "diff": diff}
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
    tol = _derive_tol(_DEFAULT_TOL["fp32"], noise)
    diff = _tensor_dict_close(baseline_pred, run, tol)
    passed = all(v.get("close", False) for v in diff.values())
    all_pass = all_pass and passed
    results["checks"]["demo_parity"]["streaming_flashinfer_fp32"] = {
        "pass": passed, "tol": tol, "diff": diff,
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
        harness_tol = _derive_tol(_DEFAULT_TOL["bf16"], harness_noise)
        # LingbotMapMethod.process_scene() returns {'frame': {...lists of ndarray...},
        # 'global': {}}. Compare every PREDICTED per-frame key element-wise, at the same
        # tolerance as demo_parity.
        #
        # Earlier revisions compared only 'depth', which left the most important output of a
        # reconstruction model ungated: 'pose' is 613 camera poses per config, and
        # 'confidence' 613 confidence maps, all of which this refactor's KV-cache changes
        # could perturb. Verified against the real staged artifact
        # (harness_windowed_flashinfer.pt): frame carries rgb, depth, pose, intrinsics,
        # confidence -- each a 613-element list of ndarrays.
        #
        # 'rgb' is excluded: it is the input imagery echoed back, not a prediction, so a
        # difference there could only indicate a harness/loader bug, and it is the largest
        # payload by far (613 x 378 x 518 x 3). 'intrinsics' IS included -- it is small, and
        # while it should be a pure function of the fixture, that is an assumption worth
        # gating rather than trusting.
        _HARNESS_KEYS = ("depth", "pose", "intrinsics", "confidence")
        per_key = {}
        for hk in _HARNESS_KEYS:
            b_list = baseline_output["frame"].get(hk)
            c_list = output["frame"].get(hk)
            if b_list is None and c_list is None:
                per_key[hk] = {"close": True, "absent_both": True}
                continue
            if (b_list is None) != (c_list is None):
                per_key[hk] = {"close": False, "present_baseline": b_list is not None,
                               "present_compare": c_list is not None}
                continue
            if len(b_list) != len(c_list):
                per_key[hk] = {"close": False, "length_mismatch": True,
                               "baseline_len": len(b_list), "compare_len": len(c_list)}
                continue
            # Track the worst per-frame deviation so a FAIL says how far off it was, and a
            # PASS records the margin, instead of only a boolean.
            worst = 0.0
            worst_frame = -1
            bad = 0
            for i, (bd, cd) in enumerate(zip(b_list, c_list)):
                if getattr(bd, "shape", None) != getattr(cd, "shape", None):
                    bad += 1
                    if worst_frame < 0:
                        worst_frame = i
                    continue
                d = abs(bd - cd).max()
                lim = harness_tol["atol"] + harness_tol["rtol"] * abs(bd).max()
                if d > lim:
                    bad += 1
                if d > worst:
                    worst, worst_frame = float(d), i
            per_key[hk] = {"close": bad == 0, "num_frames": len(b_list),
                           "num_frames_over_tol": bad,
                           "max_abs_diff": worst, "worst_frame": worst_frame}

        harness_pass = all(v["close"] for v in per_key.values())
        harness_check[tag] = {"pass": bool(harness_pass), "tol": harness_tol,
                              "per_key": per_key}
        all_pass = all_pass and harness_pass
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
    import lingbot_map

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Record and optionally ASSERT which lingbot_map actually got imported.
    #
    # This matters more than it looks. The Tier 2 worker `pip install -e`s the checkout at
    # ~/lingbot-map, so `import lingbot_map` resolves through that editable install. Running
    # this script from inside a *different* clone does NOT override it, and neither does
    # PYTHONPATH -- verified empirically, not assumed:
    #
    #   `pip install -e` drops a .pth that registers an `_EditableFinder` on sys.meta_path,
    #   and meta_path finders are consulted BEFORE the sys.path-based PathFinder. So the
    #   editable install beats PYTHONPATH and cwd both. (sys.path[0] for a script is also
    #   the script's own directory, <clone>/verify, not the clone root -- but that is the
    #   lesser of the two reasons.)
    #
    # A probe intended to measure the BASELINE ref would therefore silently import and
    # measure the REFACTOR code and record it as a baseline -- a false reference of exactly
    # the kind that makes every downstream comparison meaningless while looking healthy.
    #
    # To actually select the code under measurement, re-point the editable install:
    #     pip install -e <baseline-clone> --no-deps   # --no-deps: do not touch the pins
    #     <run this probe>
    #     pip install -e <refactor-checkout> --no-deps
    # and pass --expect_module_root so that selection is VERIFIED rather than assumed.
    module_root = str(Path(lingbot_map.__file__).resolve().parent.parent)
    print(f"[run_kv_info_probe] imported lingbot_map from: {module_root}", flush=True)
    if args.expect_module_root:
        expected = str(Path(args.expect_module_root).resolve())
        if module_root != expected:
            raise RuntimeError(
                f"Refusing to record a probe from the wrong checkout: imported "
                f"lingbot_map from {module_root!r} but --expect_module_root is "
                f"{expected!r}. Note PYTHONPATH CANNOT fix this -- an editable install's "
                f"meta_path finder wins over sys.path. Re-point the editable install "
                f"(`pip install -e {expected} --no-deps`) instead. Recording this as a "
                f"baseline would silently compare the refactor against itself."
            )

    values = _check_get_kv_cache_info(device)
    payload = {"mode": "kv_info_probe", "module_root": module_root,
               "kv_cache_info_states": values}
    (output_dir / "kv_info_probe.json").write_text(json.dumps(payload, indent=2))
    print(f"[run_kv_info_probe] {json.dumps(values, indent=2)}", flush=True)
    return payload


def run_self_test(args) -> dict:
    """Exercise every branch of _derive_tol on CPU: no torch, no checkpoint, no fixture, no
    GPU, seconds not minutes.

    The tolerance derivation is the one piece of this harness that decides what counts as a
    pass, and it is otherwise only ever executed inside a ~40-minute Tier 2 run, where a
    regression in it would present as a passing gate rather than as an error. These
    assertions are what make the derivation checkable without spending that run.
    """
    # Deliberately a duplicated literal, NOT a reference to _DEFAULT_TOL: this is the gate
    # compare job 21 was actually scored at, per verify/verification_tolerance_baseline.md's
    # derived-tolerance table. Pinning it here means an edit to _DEFAULT_TOL fails this test
    # loudly instead of silently rescoring the one result this refactor is trusted on.
    recorded_gate = {
        "bf16": {"rtol": 1e-3, "atol": 1e-5},
        "fp32": {"rtol": 1e-5, "atol": 1e-6},
    }
    checks = []

    def _expect(name: str, ok: bool, detail: dict) -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})
        print("[self_test] {}: {}".format("PASS" if ok else "FAIL", name), flush=True)
        if not ok:
            print("[self_test]   detail: {}".format(json.dumps(detail)), flush=True)

    keys = ("pose_enc", "depth", "depth_conf")

    # 1. A 0.0 floor -- what all five configurations actually measured -- must reduce to the
    #    defaults exactly, so this change is a no-op against the recorded verdict. Same for
    #    a configuration with no baseline entry at all.
    #    The legacy-shaped zero floor is included here, not with the refusal cases below,
    #    because it is the EXACT shape of the staged job-16 baseline (bare floats, all 0.0):
    #    this is the assertion that makes this change a provable no-op against the recorded
    #    verdict, refusal record included.
    zero_floor = {k: {"max_abs_diff": 0.0, "max_rel_diff": 0.0} for k in keys}
    legacy_zero_floor = {k: 0.0 for k in keys}
    for path in ("bf16", "fp32"):
        defaults, recorded = _DEFAULT_TOL[path], recorded_gate[path]
        for label, floor in (("zero_floor", zero_floor),
                             ("legacy_zero_floor", legacy_zero_floor),
                             ("no_baseline_entry", {})):
            tol = _derive_tol(defaults, floor)
            _expect(
                "{}_{}_reduces_to_recorded_gate".format(path, label),
                (tol["rtol"] == defaults["rtol"] == recorded["rtol"]
                 and tol["atol"] == defaults["atol"] == recorded["atol"]
                 and not tol["rtol_widened_from_relative_floor"]
                 and "rtol_widening_refused" not in tol),
                {"derived": {"rtol": repr(tol["rtol"]), "atol": repr(tol["atol"])},
                 "recorded_gate": {"rtol": repr(recorded["rtol"]),
                                   "atol": repr(recorded["atol"])}},
            )

    # 2. A nonzero RELATIVE floor widens rtol -- the half of Decision 4 step 5 the previous
    #    revision never applied -- and leaves atol at its default.
    rel = 4e-4
    tol = _derive_tol(_DEFAULT_TOL["bf16"],
                      {"pose_enc": {"max_abs_diff": 0.0, "max_rel_diff": rel},
                       "depth": {"max_abs_diff": 0.0, "max_rel_diff": 0.0}})
    _expect(
        "relative_floor_widens_rtol_only",
        (tol["rtol"] == 3 * rel
         and tol["rtol"] > _DEFAULT_TOL["bf16"]["rtol"]
         and tol["atol"] == _DEFAULT_TOL["bf16"]["atol"]
         and tol["rtol_widened_from_relative_floor"]
         and tol["rel_floor_max"] == rel),
        {"tol": {k: repr(v) for k, v in tol.items()}, "expected_rtol": repr(3 * rel)},
    )

    # 3. A nonzero ABSOLUTE floor widens atol and MUST NOT touch rtol. The floor is chosen
    #    so 3 x it exceeds the bf16 rtol default: the naive single-max() widening would have
    #    loosened rtol by 50% here, which is the defect this helper exists to prevent.
    ab = 5e-4
    tol = _derive_tol(_DEFAULT_TOL["bf16"],
                      {"pose_enc": {"max_abs_diff": ab, "max_rel_diff": 0.0},
                       "depth": {"max_abs_diff": 0.0, "max_rel_diff": 0.0}})
    _expect(
        "absolute_floor_widens_atol_never_rtol",
        (tol["atol"] == 3 * ab
         and tol["rtol"] == _DEFAULT_TOL["bf16"]["rtol"] == recorded_gate["bf16"]["rtol"]
         and not tol["rtol_widened_from_relative_floor"]
         and 3 * ab > _DEFAULT_TOL["bf16"]["rtol"]),
        {"tol": {k: repr(v) for k, v in tol.items()},
         "naive_rtol_would_have_been": repr(3 * ab)},
    )

    # 4. A legacy abs-only baseline (bare float per key -- the format of every artifact
    #    written before this change) with a NONZERO floor must refuse to widen rtol and
    #    record why, rather than widening from the wrong quantity. Only the nonzero key is
    #    named: a 0.0 absolute floor entails a 0.0 relative floor (see _derive_tol).
    legacy = {"pose_enc": ab, "depth": 0.0, "depth_conf": 0.0}
    tol = _derive_tol(_DEFAULT_TOL["bf16"], legacy)
    _expect(
        "legacy_abs_only_baseline_records_rtol_refusal",
        (tol.get("rtol_widening_refused") is True
         and tol["rtol"] == _DEFAULT_TOL["bf16"]["rtol"]
         and tol["atol"] == 3 * ab
         and tol["rel_floor_max"] is None
         and tol.get("rtol_refusal_keys") == ["pose_enc"]
         and "unit error" in tol.get("rtol_refusal_reason", "")),
        {"tol": {k: repr(v) for k, v in tol.items()}},
    )

    # Same refusal for a half-migrated entry: per-key dict, absolute half only.
    tol = _derive_tol(_DEFAULT_TOL["bf16"], {"pose_enc": {"max_abs_diff": ab}})
    _expect(
        "per_key_dict_without_relative_half_records_rtol_refusal",
        (tol.get("rtol_widening_refused") is True
         and tol["rtol"] == _DEFAULT_TOL["bf16"]["rtol"]
         and tol["atol"] == 3 * ab),
        {"tol": {k: repr(v) for k, v in tol.items()}},
    )

    # 5. Both halves nonzero in the SAME call. The four cases above each move one half, so
    #    every one of them passes against a helper that can only widen one at a time.
    tol = _derive_tol(_DEFAULT_TOL["bf16"],
                      {"pose_enc": {"max_abs_diff": ab, "max_rel_diff": rel},
                       "depth": {"max_abs_diff": 0.0, "max_rel_diff": 0.0}})
    _expect(
        "both_halves_widen_independently_in_one_call",
        (tol["atol"] == 3 * ab and tol["rtol"] == 3 * rel
         and tol["abs_floor_max"] == ab and tol["rel_floor_max"] == rel
         and tol["rtol_widened_from_relative_floor"]
         and "rtol_widening_refused" not in tol),
        {"tol": {k: repr(v) for k, v in tol.items()},
         "expected": {"atol": repr(3 * ab), "rtol": repr(3 * rel)}},
    )

    # 6. THE UNFAILABLE GATE. An unbounded rtol is not a loose gate, it is the absence of
    #    one: torch.allclose(rtol=inf) passes any two tensors and records a PASS. rtol was
    #    pinned before the relative half existed, so this is a failure mode the relative
    #    half introduced, and the run that first writes a nonzero-floor baseline is the run
    #    that arms it. Each of these must hold rtol at the default and say why.
    for label, bad_rel, expect_unusable in (("huge", 1e11, False),
                                            ("infinite", float("inf"), True),
                                            ("nan", float("nan"), True),
                                            ("negative", -1e-3, True)):
        tol = _derive_tol(_DEFAULT_TOL["bf16"],
                          {"pose_enc": {"max_abs_diff": 0.0, "max_rel_diff": bad_rel},
                           "depth": {"max_abs_diff": 0.0, "max_rel_diff": 0.0}})
        # Caught rather than raised: a non-finite that reaches results.json is a FAIL with a
        # reason, not a traceback out of the middle of the suite.
        strict_json = _is_strict_json(tol)
        _expect(
            "a_{}_relative_floor_cannot_widen_rtol".format(label),
            (tol["rtol"] == _DEFAULT_TOL["bf16"]["rtol"] == recorded_gate["bf16"]["rtol"]
             and not tol["rtol_widened_from_relative_floor"]
             and tol.get("rtol_widening_refused") is True
             and ("unusable_floors" in tol) is expect_unusable
             and strict_json),
            {"tol": {k: repr(v) for k, v in tol.items()}, "floor": repr(bad_rel),
             "strict_json": strict_json},
        )
    # ...and the same on the absolute half, which has no ceiling but must still reject a
    #    floor that is not a finite non-negative number.
    for label, bad_abs in (("infinite", float("inf")), ("nan", float("nan")),
                           ("negative", -1.0)):
        tol = _derive_tol(_DEFAULT_TOL["bf16"], {"pose_enc": {"max_abs_diff": bad_abs,
                                                             "max_rel_diff": 0.0}})
        _expect(
            "a_{}_absolute_floor_cannot_widen_atol".format(label),
            (tol["atol"] == _DEFAULT_TOL["bf16"]["atol"] == recorded_gate["bf16"]["atol"]
             and tol["atol_widening_factor"] == 1.0
             and tol["abs_floor_max"] is None
             and [u["half"] for u in tol.get("unusable_floors", [])] == ["max_abs_diff"]
             and _is_strict_json(tol)),
            {"tol": {k: repr(v) for k, v in tol.items()}, "floor": repr(bad_abs),
             "strict_json": _is_strict_json(tol)},
        )
    # And the boundary itself, from both sides, so the ceiling is a value and not a mood.
    just_under = (_RTOL_WIDENING_CEILING / 3.0) * (1 - 1e-9)
    tol_under = _derive_tol(_DEFAULT_TOL["bf16"], {"pose_enc": {"max_abs_diff": 0.0,
                                                               "max_rel_diff": just_under}})
    tol_over = _derive_tol(_DEFAULT_TOL["bf16"],
                           {"pose_enc": {"max_abs_diff": 0.0,
                                         "max_rel_diff": _RTOL_WIDENING_CEILING / 3.0}})
    _expect(
        "the_rtol_ceiling_discriminates_at_its_own_boundary",
        (tol_under["rtol"] == 3 * just_under
         and "rtol_widening_refused" not in tol_under
         and tol_over["rtol"] == _DEFAULT_TOL["bf16"]["rtol"]
         and tol_over.get("rtol_widening_refused") is True),
        {"under": repr(tol_under["rtol"]), "over": repr(tol_over["rtol"]),
         "ceiling": repr(_RTOL_WIDENING_CEILING)},
    )

    all_pass = all(c["pass"] for c in checks)
    # Provenance, in the style run_kv_info_probe records its module_root: which file and
    # which constants this verdict is about. Deliberately NOT a git revision -- the Tier 2
    # container has no git, so a harness that tried to stamp one would report nothing or
    # crash; the file path plus the asserted constants are what is actually available here.
    results = {"mode": "self_test", "pass": all_pass, "complete": True,
               "harness_file": str(Path(__file__).resolve()),
               "defaults_asserted": _DEFAULT_TOL,
               "recorded_gate_asserted": recorded_gate,
               "num_checks": len(checks), "checks": checks}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    print("[self_test] {}/{} checks passed -> {}".format(
        sum(1 for c in checks if c["pass"]), len(checks),
        "PASS" if all_pass else "FAIL"), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",
                        choices=["baseline", "compare", "kv_info_probe", "self_test"],
                        required=True)
    # Not required for kv_info_probe (builds only tiny synthetic models) or self_test
    # (builds nothing at all): neither mode touches the checkpoint or the fixture.
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
    parser.add_argument("--expect_module_root", default=None,
                         help="For --mode kv_info_probe: assert that `import lingbot_map` "
                              "resolves inside this directory, so a probe cannot silently "
                              "measure a different checkout than intended. Select the "
                              "checkout by re-pointing the editable install, NOT with "
                              "PYTHONPATH (which an editable install overrides).")
    args = parser.parse_args()

    if args.mode == "compare" and not args.baseline_dir:
        parser.error("--baseline_dir is required for --mode compare")
    if args.mode in ("baseline", "compare"):
        missing = [n for n in ("checkpoint", "tum_root", "scene")
                   if getattr(args, n) is None]
        if missing:
            parser.error(
                f"--mode {args.mode} requires: {', '.join('--' + m for m in missing)}")

    if args.mode == "self_test":
        sys.exit(0 if run_self_test(args)["pass"] else 1)
    elif args.mode == "kv_info_probe":
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
