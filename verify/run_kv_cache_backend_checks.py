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
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


def _run_inference(model, mode: str, images, device=None, force_fp32: bool = False):
    import torch

    if device is None:
        device = next(model.parameters()).device
    images = images.to(device)
    dtype = torch.float32 if force_fp32 else (
        torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=not force_fp32):
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
    predictions. On Tier 2 (this script's real target), FlashInfer is installed and
    use_sdpa=False constructs a real manager, so both backends are exercised for real
    here (unlike Tier 1, which can only run the SDPA half -- see TASK-016's own note).
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
        with torch.no_grad():
            model.inference_streaming(images, num_scale_frames=1, keyframe_interval=1)
        states.append(model.get_kv_cache_info())
        key = "sdpa" if use_sdpa else "flashinfer"
        results[key] = states
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


def _tensor_dict_close(a: dict, b: dict, tol: dict) -> dict:
    """Per-key parity between two prediction dicts.

    _OUTPUT_KEYS lists every key the model *can* emit, but which are actually present
    depends on the model config: world_points/world_points_conf only exist when
    enable_point=True, and the production pipeline (benchmark/methods/lingbot_map.py and
    demo.py) leaves enable_point at its False default, so those two keys are legitimately
    absent from both runs. A key absent from BOTH is parity -- same before and after the
    refactor -- so it counts as close=True. A key present in one run but not the other IS
    a real divergence (the refactor added or dropped an output) and stays close=False, so
    callers' `all(v["close"] for ...)` aggregation still catches that case.
    """
    import torch
    diffs = {}
    for key in _OUTPUT_KEYS:
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
        ta, tb = a[key].float(), b[key].float()
        close = torch.allclose(ta, tb, rtol=tol["rtol"], atol=tol["atol"])
        max_abs = (ta - tb).abs().max().item()
        diffs[key] = {"close": bool(close), "max_abs_diff": max_abs}
    return diffs


def run_baseline(args) -> dict:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tum_root = Path(args.tum_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"mode": "baseline", "checks": {}}

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
        torch.save(run_a, predictions_dir / f"demo_{tag}.pt")
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
    torch.save(run_a, predictions_dir / "demo_streaming_flashinfer_fp32.pt")
    results["checks"]["noise_floors"] = {"pass": True, "values": noise_floors}
    del model, run_a, run_b
    _free_cuda()

    # Item 2: benchmark harness parity, both modes x backends.
    for mode, use_sdpa in _CONFIGS:
        tag = f"{mode}_{'sdpa' if use_sdpa else 'flashinfer'}"
        output = _run_benchmark_harness(mode, use_sdpa, args.checkpoint, device,
                                         tum_root, args.scene)
        torch.save(output, predictions_dir / f"harness_{tag}.pt")
        del output
        _free_cuda()
    results["checks"]["harness_recorded"] = {"pass": True}

    # Item 6: get_kv_cache_info byte-identity states (recorded, not compared yet --
    # compare mode diffs against these).
    kv_info_states = _check_get_kv_cache_info(device)
    kv_info_states = {
        k: [{kk: vv for kk, vv in state.items()} for state in v]
        for k, v in kv_info_states.items()
    }
    results["checks"]["kv_cache_info_states"] = {"pass": True, "values": kv_info_states}

    # Items 7/8: Gap A/B preserved, recorded as booleans (must remain the same value
    # in compare mode).
    results["checks"]["gap_a_preserved"] = {"pass": True, "value": _check_gap_a()}
    results["checks"]["gap_b_preserved"] = {"pass": True, "value": _check_gap_b()}

    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
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
        del output, baseline_output
        _free_cuda()
    results["checks"]["harness_parity"] = harness_check

    # Item 6: get_kv_cache_info byte-identity vs baseline's recorded states.
    kv_info_states = _check_get_kv_cache_info(device)
    baseline_kv_states = baseline_results["checks"]["kv_cache_info_states"]["values"]
    kv_pass = kv_info_states == baseline_kv_states
    all_pass = all_pass and kv_pass
    results["checks"]["kv_cache_info_byte_identical"] = {
        "pass": bool(kv_pass), "baseline": baseline_kv_states, "compare": kv_info_states,
    }

    # Items 7/8: Gap A/B preservation must match baseline's recorded value exactly.
    gap_a = _check_gap_a()
    gap_a_pass = gap_a == baseline_results["checks"]["gap_a_preserved"]["value"]
    all_pass = all_pass and gap_a_pass
    results["checks"]["gap_a_preserved"] = {"pass": bool(gap_a_pass), "value": gap_a}

    gap_b = _check_gap_b()
    gap_b_pass = gap_b == baseline_results["checks"]["gap_b_preserved"]["value"]
    all_pass = all_pass and gap_b_pass
    results["checks"]["gap_b_preserved"] = {"pass": bool(gap_b_pass), "value": gap_b}

    results["pass"] = all_pass
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "compare"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tum_root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_dir", default=None,
                         help="Required for --mode compare")
    args = parser.parse_args()

    if args.mode == "compare" and not args.baseline_dir:
        parser.error("--baseline_dir is required for --mode compare")

    if args.mode == "baseline":
        results = run_baseline(args)
        # Baseline mode has no pass/fail semantics of its own (per TASK-027's own
        # framing) -- it always "succeeds" if it completes and records numbers.
        sys.exit(0)
    else:
        results = run_compare(args)
        sys.exit(0 if results["pass"] else 1)


if __name__ == "__main__":
    main()
