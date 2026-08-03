"""Desk-side verification of C3 (``keyframe_decision``) against the GPU original.

spec/neuron-port/design.md V4(a): the branchless keyframe decision must produce a keyframe
sequence **identical element for element** to the GPU's — not close, identical (FM8, REQ-025).
A single flipped decision changes which frames enter the KV cache, so every subsequent frame
attends a different key set. There is no tolerance to spend.

The reference is the REAL ``gct_stream_window._compute_flow_magnitude`` plus the real inline
branch logic transcribed from ``:575-587``, so this compares against the shipping code rather
than against a restatement of it.

Two guards against a vacuous pass, per the verification-gate lesson:
  * the recorded sequence must contain at least one keyframe AND one non-keyframe;
  * a deliberately wrong variant (fp16 flow accumulation) must FAIL, proving the comparison
    can detect a divergence at all.

Run:
    docker exec lingbot-tier1 python \
        /home/kwehden/lingbot-tier1/lingbot-map/verify/neuron/check_keyframe_decision.py
"""

from __future__ import annotations

import json
import os
import sys

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)

from lingbot_map.models.gct_stream_window import _compute_flow_magnitude  # noqa: E402
from lingbot_map.models.keyframe import flow_magnitude, keyframe_decision  # noqa: E402

RESULTS: list[dict] = []


def emit(name: str, ok: bool, **kw) -> None:
    RESULTS.append({"check": name, "pass": bool(ok), **kw})
    extra = "  ".join(f"{k}={v}" for k, v in kw.items())
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {extra}", flush=True)


# ---------------------------------------------------------------------------------
# A synthetic camera trajectory. Not a stand-in for freiburg1_desk — it is a driver
# that produces a MIX of keyframe and non-keyframe decisions, which is what V4(a)
# needs to be non-vacuous. The flow math under test is identical either way; what a
# real sequence would add is realistic *margins*, which this reports so the fp32-vs-
# float64 threshold question is answered by measurement (see keyframe.py).
# ---------------------------------------------------------------------------------
def make_pose_enc(t: torch.Tensor, dev, dtype=torch.float32) -> torch.Tensor:
    """[1,1,9] pose encoding: translation, quaternion, fov. Small rotation about y."""
    ang = 0.02 * t
    q = torch.tensor(
        [0.0, torch.sin(ang / 2).item(), 0.0, torch.cos(ang / 2).item()],
        dtype=dtype, device=dev,
    )
    trans = torch.tensor([0.03 * t.item(), 0.005 * t.item(), 0.02 * t.item()],
                         dtype=dtype, device=dev)
    fov = torch.tensor([0.8, 0.6], dtype=dtype, device=dev)
    return torch.cat([trans, q, fov]).view(1, 1, 9)


def make_depth(t: torch.Tensor, dev, H=28, W=28, dtype=torch.float32) -> torch.Tensor:
    g = torch.linspace(0.5, 4.0, H * W, device=dev, dtype=dtype).view(1, 1, H, W, 1)
    return g + 0.01 * t.to(dtype)


def main() -> int:
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emit("device", True, device=str(dev),
         gpu=(torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"))

    H = W = 28
    HW = (H, W)
    STRIDE = 8
    FLOW_THRESHOLD = 3.0
    MAX_GAP = 8
    SCALE_FRAMES = 8
    N = 60

    # ---- 1. flow_magnitude matches _compute_flow_magnitude exactly -----------------
    max_flow_diff = 0.0
    n_compared = 0
    for i in range(N):
        t = torch.tensor(float(i), device=dev)
        cur = make_pose_enc(t, dev)
        kf = make_pose_enc(torch.tensor(float(max(0, i - 3))), dev)
        d = make_depth(t, dev, H, W)
        ref = _compute_flow_magnitude(cur, kf, d, HW, stride=STRIDE)   # Python float
        got = flow_magnitude(cur, kf, d, HW, stride=STRIDE)            # 0-d tensor
        assert torch.is_tensor(got) and got.dim() == 0
        max_flow_diff = max(max_flow_diff, abs(float(got.item()) - ref))
        n_compared += 1
    emit("flow_magnitude_matches_original_bitwise", max_flow_diff == 0.0,
         max_abs_diff=max_flow_diff, n=n_compared,
         detail="same ops in the same order -> exact, not merely close")

    # ---- 2. the branchless empty-valid path returns 0 without nan -------------------
    # All-zero depth makes every pixel invalid, which is the `if valid_count < 1` case.
    zero_d = torch.zeros(1, 1, H, W, 1, device=dev)
    t0 = torch.tensor(1.0, device=dev)
    ref0 = _compute_flow_magnitude(make_pose_enc(t0, dev), make_pose_enc(t0 * 0, dev),
                                   zero_d, HW, stride=STRIDE)
    got0 = flow_magnitude(make_pose_enc(t0, dev), make_pose_enc(t0 * 0, dev),
                          zero_d, HW, stride=STRIDE)
    emit("empty_valid_mask_returns_zero_not_nan",
         float(got0.item()) == 0.0 == ref0 and torch.isfinite(got0).item(),
         ref=ref0, got=float(got0.item()),
         detail="torch.where replaces `if valid_count < 1: return 0.0`; the clamped divisor "
                "keeps the discarded branch finite so the select cannot propagate nan")

    # ---- 3. THE CHECK: full keyframe sequence, C3 vs the original inline branch ------
    def run_reference() -> tuple[list[bool], list[float]]:
        """The original logic, transcribed from gct_stream_window.py:575-587."""
        seq, margins = [], []
        last_kf_pose = make_pose_enc(torch.tensor(float(SCALE_FRAMES - 1)), dev)
        last_kf_idx = SCALE_FRAMES - 1
        for i in range(SCALE_FRAMES, N):
            t = torch.tensor(float(i), device=dev)
            cur_pose = make_pose_enc(t, dev)
            cur_depth = make_depth(t, dev, H, W)
            flow_mag = _compute_flow_magnitude(cur_pose, last_kf_pose, cur_depth, HW,
                                               stride=STRIDE)
            frames_since_kf = i - last_kf_idx
            is_kf = (
                (i == SCALE_FRAMES)
                or (flow_mag > FLOW_THRESHOLD)
                or (frames_since_kf >= MAX_GAP)
            )
            margins.append(abs(flow_mag - FLOW_THRESHOLD))
            seq.append(bool(is_kf))
            if is_kf:
                last_kf_pose = cur_pose
                last_kf_idx = i
        return seq, margins

    def run_c3(depth_perturb: float = 0.0) -> list[bool]:
        """C3, driven exactly as C4 will drive it.

        ``depth_perturb`` scales the depth map, which scales the reprojected flow — the
        deliberately-wrong variant for check 5. A dtype change would only raise a TypeError,
        which proves nothing about whether the comparison can detect a WRONG ANSWER.
        """
        seq = []
        last_kf_pose = make_pose_enc(torch.tensor(float(SCALE_FRAMES - 1)), dev)
        last_kf_idx = SCALE_FRAMES - 1
        for i in range(SCALE_FRAMES, N):
            t = torch.tensor(float(i), device=dev)
            cur_pose = make_pose_enc(t, dev)
            cur_depth = make_depth(t, dev, H, W)
            if depth_perturb:
                cur_depth = cur_depth * (1.0 + depth_perturb)
            frames_since_kf = torch.tensor(i - last_kf_idx, dtype=torch.int32, device=dev)
            dec = keyframe_decision(
                cur_pose, last_kf_pose, cur_depth, HW, frames_since_kf,
                flow_threshold=FLOW_THRESHOLD, max_non_keyframe_gap=MAX_GAP,
                is_first_streaming_frame=(i == SCALE_FRAMES), stride=STRIDE,
            )
            assert torch.is_tensor(dec) and dec.dim() == 0 and dec.dtype == torch.bool
            is_kf = bool(dec.item())        # the ONE sync per frame (design.md D3)
            seq.append(is_kf)
            if is_kf:
                last_kf_pose = cur_pose
                last_kf_idx = i
        return seq

    ref_seq, margins = run_reference()
    c3_seq = run_c3()
    n_kf = sum(ref_seq)
    n_non = len(ref_seq) - n_kf

    emit("keyframe_sequence_identical_to_gpu", ref_seq == c3_seq,
         n_frames=len(ref_seq), n_keyframes=n_kf, n_non_keyframes=n_non,
         first_divergence=next(
             (i for i, (a, b) in enumerate(zip(ref_seq, c3_seq)) if a != b), None),
         detail="element-for-element, not tolerance (FM8)")

    # Vacuity guard: an all-True or all-False sequence would pass identity trivially.
    emit("sequence_contains_both_outcomes", n_kf > 0 and n_non > 0,
         n_keyframes=n_kf, n_non_keyframes=n_non,
         detail="if either is 0 the identity check above is vacuous")

    # ---- 4. the fp32-vs-float64 threshold margin, measured not assumed -------------
    min_margin = min(margins)
    fp32_ulp_at_thresh = float(
        torch.nextafter(torch.tensor(FLOW_THRESHOLD), torch.tensor(1e9)) - FLOW_THRESHOLD
    )
    emit("threshold_margin_exceeds_fp32_ulp", min_margin > fp32_ulp_at_thresh * 4,
         min_abs_margin=round(min_margin, 8), fp32_ulp=fp32_ulp_at_thresh,
         detail="C3 compares in fp32 where the original compared float64; identity is safe "
                "only while flow_mag stays off the threshold by more than an ulp. This "
                "measures the margin on THIS trajectory -- a real sequence could sit closer, "
                "so the margin, not the identity, is the durable claim")

    # ---- 5. discriminating power: a wrong variant must FAIL ------------------------
    bad_seq = run_c3(depth_perturb=0.10)
    emit("wrong_variant_diverges_as_expected", bad_seq != ref_seq,
         n_differences=sum(1 for a, b in zip(ref_seq, bad_seq) if a != b),
         detail="10% depth scaling perturbs the reprojected flow; if this MATCHES, the "
                "identity check cannot detect a wrong answer and check 3 is vacuous")

    # ---- 6. no host sync inside C3 --------------------------------------------------
    # The decision tensor must come back on-device: if C3 had synced internally it would
    # have had to produce a host value somewhere.
    t = torch.tensor(20.0, device=dev)
    dec = keyframe_decision(
        make_pose_enc(t, dev), make_pose_enc(t * 0, dev), make_depth(t, dev, H, W), HW,
        torch.tensor(3, dtype=torch.int32, device=dev),
        flow_threshold=FLOW_THRESHOLD, max_non_keyframe_gap=MAX_GAP,
        is_first_streaming_frame=False, stride=STRIDE,
    )
    emit("decision_is_device_tensor_not_bool",
         torch.is_tensor(dec) and dec.device.type == dev.type and dec.dtype == torch.bool,
         type=type(dec).__name__, device=str(dec.device), dtype=str(dec.dtype))

    # ---- 7. depth=None forces a keyframe, matching the original's fallback ----------
    dec_nd = keyframe_decision(
        make_pose_enc(t, dev), make_pose_enc(t * 0, dev), None, None,
        torch.tensor(0, dtype=torch.int32, device=dev),
        flow_threshold=FLOW_THRESHOLD, max_non_keyframe_gap=MAX_GAP,
        is_first_streaming_frame=False,
    )
    emit("depth_none_forces_keyframe", bool(dec_nd.item()) is True,
         detail="matches gct_stream_window.py:572's flow_mag = flow_threshold + 1.0")

    # ---- 8. max_non_keyframe_gap still forces a keyframe ----------------------------
    # Zero motion => zero flow, so only the gap rule can fire. Guards against the gap
    # term being silently dropped in the rewrite.
    t_static = torch.tensor(5.0, device=dev)
    same = make_pose_enc(t_static, dev)
    dec_gap = keyframe_decision(
        same, same, make_depth(t_static, dev, H, W), HW,
        torch.tensor(MAX_GAP, dtype=torch.int32, device=dev),
        flow_threshold=FLOW_THRESHOLD, max_non_keyframe_gap=MAX_GAP,
        is_first_streaming_frame=False, stride=STRIDE,
    )
    dec_nogap = keyframe_decision(
        same, same, make_depth(t_static, dev, H, W), HW,
        torch.tensor(MAX_GAP - 1, dtype=torch.int32, device=dev),
        flow_threshold=FLOW_THRESHOLD, max_non_keyframe_gap=MAX_GAP,
        is_first_streaming_frame=False, stride=STRIDE,
    )
    emit("gap_rule_fires_at_boundary_only",
         bool(dec_gap.item()) is True and bool(dec_nogap.item()) is False,
         at_gap=bool(dec_gap.item()), below_gap=bool(dec_nogap.item()),
         detail="zero motion isolates the gap term from the flow term")

    n_fail = sum(1 for r in RESULTS if not r["pass"])
    out = {
        "suite": "neuron C3 keyframe_decision vs GPU original (design.md V4a)",
        "device": str(dev),
        "reference": "REAL gct_stream_window._compute_flow_magnitude + inline branch :575-587",
        "config": {"flow_threshold": FLOW_THRESHOLD, "max_non_keyframe_gap": MAX_GAP,
                   "scale_frames": SCALE_FRAMES, "n_frames": N, "stride": STRIDE},
        "keyframe_sequence": [int(b) for b in ref_seq],
        "min_threshold_margin": min_margin,
        "n_checks": len(RESULTS), "n_fail": n_fail,
        "verdict": "PASS" if n_fail == 0 else "FAIL",
        "results": RESULTS,
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keyframe_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n=== {out['verdict']}: {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    print(f"=== wrote {path}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
