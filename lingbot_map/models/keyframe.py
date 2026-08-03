"""Branchless flow-based keyframe decision for the Neuron (Trainium2) port.

spec/neuron-port/design.md component C3. Extracted from the two duplicated loop sites in
``gct_stream_window.py`` (design.md F6) so there is one implementation to verify rather than
two to keep in sync.

WHAT CHANGES AND WHAT DOES NOT
------------------------------
The projection math is a transcription of ``gct_stream_window._compute_flow_magnitude``
(``:29-112``), op for op and in the same order. That is deliberate: V4 requires the keyframe
sequence to be **element-for-element identical** to the GPU's, not merely close (design.md
FM8, REQ-025), and any reordering of the geometry would put that at risk for no benefit.

Only the three compile blockers are removed:

===========================================  =============================================
GPU original                                 Neuron form
===========================================  =============================================
``if valid_count < 1: return 0.0`` (``:108``) ``torch.where`` — both sides always computed
``return mean_mag.item()`` (``:112``)        return the device tensor; no host sync
``if cur_depth is not None:`` (``:566``)     resolved on the host at trace time
===========================================  =============================================

The depth-presence branch is not a runtime condition: whether the model emits ``depth`` is a
static property of the head configuration, fixed before the loop starts. So the caller
resolves it once (see :func:`keyframe_decision`'s ``flow_mag`` argument) instead of branching
on tensor presence inside the traced region.

WHY ONE HOST SYNC PER FRAME REMAINS
-----------------------------------
This function returns a device tensor and performs no sync. But the *caller* must still read
one scalar per frame, because commit-vs-rollback is Python-side cache bookkeeping — which
ring slot is live — and that cannot be expressed as a device-side select over
``NeuronRingKVCacheBackend``'s host-int state. design.md accepts this as D3 and tracks
eliminating it (a device-side state machine) as ODQ4, to be built only if measurement shows
the sync dominates. The point of C3 is that the *flow computation* — the expensive part, a
full reprojection over a strided depth map — stays on device and in one graph. The sync moves
to exactly one boolean at the end.

``is_first_streaming_frame`` stays a Python bool deliberately: it is loop-index arithmetic
(``i == scale_frames``), known at trace time, not runtime data. ``frames_since_kf`` is a
device tensor because it depends on past decisions.

A NOTE ON WHAT "BRANCHLESS" BUYS
--------------------------------
``torch.where`` here is not a micro-optimisation. On a fixed-shape accelerator a Python
branch on a tensor value forces a host sync to resolve, which both serialises the pipeline and
— because the two sides of the branch may trace to different graphs — can multiply the number
of NEFFs. Computing both sides and selecting is strictly cheaper than either consequence.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from lingbot_map.utils.geometry import closed_form_inverse_se3
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri


@torch.no_grad()
def flow_magnitude(
    cur_pose_enc: Tensor,
    kf_pose_enc: Tensor,
    cur_depth: Tensor,
    image_size_hw: Tuple[int, int],
    stride: int = 8,
) -> Tensor:
    """Mean camera-motion-induced optical flow, as a 0-d **device** tensor.

    Branchless, sync-free counterpart to ``gct_stream_window._compute_flow_magnitude``. Same
    inputs, same math; returns a tensor where the original returned a Python float.

    Args:
        cur_pose_enc: Current frame pose encoding [B, 1, 9].
        kf_pose_enc: Last keyframe pose encoding [B, 1, 9].
        cur_depth: Current frame depth map [B, 1, H, W, 1].
        image_size_hw: (H, W) of the depth map.
        stride: Subsampling stride for efficiency.

    Returns:
        0-d tensor: mean flow magnitude in pixels. Zero where no pixel is valid.
    """
    H, W = image_size_hw
    device = cur_pose_enc.device
    dtype = cur_depth.dtype

    cur_ext, cur_intr = pose_encoding_to_extri_intri(
        cur_pose_enc, image_size_hw=image_size_hw
    )
    kf_ext, kf_intr = pose_encoding_to_extri_intri(
        kf_pose_enc, image_size_hw=image_size_hw
    )
    B = cur_ext.shape[0]

    cur_ext = cur_ext[:, 0]
    cur_intr = cur_intr[:, 0]
    kf_ext = kf_ext[:, 0]
    kf_intr = kf_intr[:, 0]

    depth = cur_depth[:, 0, ::stride, ::stride, 0].to(dtype)
    Hs, Ws = depth.shape[1], depth.shape[2]

    v_coords = torch.arange(0, H, stride, device=device, dtype=dtype)
    u_coords = torch.arange(0, W, stride, device=device, dtype=dtype)
    v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing="ij")
    ones = torch.ones_like(u_grid)
    pixel_coords = torch.stack([u_grid, v_grid, ones], dim=-1)

    intr_inv = torch.inverse(cur_intr)
    cam_coords = torch.einsum("bij,hwj->bhwi", intr_inv, pixel_coords)
    cam_pts = cam_coords * depth.unsqueeze(-1)

    c2w = torch.zeros(B, 4, 4, device=device, dtype=dtype)
    c2w[:, :3, :] = cur_ext
    c2w[:, 3, 3] = 1.0

    ones_hw = torch.ones(B, Hs, Ws, 1, device=device, dtype=dtype)
    cam_pts_h = torch.cat([cam_pts, ones_hw], dim=-1)
    world_pts = torch.einsum("bij,bhwj->bhwi", c2w, cam_pts_h)[..., :3]

    kf_c2w = torch.zeros(B, 4, 4, device=device, dtype=dtype)
    kf_c2w[:, :3, :] = kf_ext
    kf_c2w[:, 3, 3] = 1.0
    kf_w2c = closed_form_inverse_se3(kf_c2w)
    world_pts_h = torch.cat([world_pts, ones_hw], dim=-1)
    kf_cam_pts = torch.einsum("bij,bhwj->bhwi", kf_w2c, world_pts_h)[..., :3]

    z = kf_cam_pts[..., 2:3].clamp(min=1e-6)
    kf_cam_norm = kf_cam_pts / z
    kf_pixels = torch.einsum("bij,bhwj->bhwi", kf_intr, kf_cam_norm)[..., :2]

    orig_pixels = torch.stack([u_grid, v_grid], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    flow = kf_pixels - orig_pixels
    valid = (depth > 1e-6) & (kf_cam_pts[..., 2] > 1e-6)

    flow_mag = flow.norm(dim=-1)
    valid_count = valid.to(flow_mag.dtype).sum()

    # THE FIRST BLOCKER. The original returns early on `if valid_count < 1`. Here both sides
    # are computed and selected. The clamp is what makes that safe: it keeps the unused
    # branch's divisor away from zero, so the discarded value is finite rather than nan.
    # Selecting between a finite value and a nan still yields nan on some backends, so the
    # nan must not be produced in the first place.
    safe_count = valid_count.clamp(min=1.0)
    mean_mag = (flow_mag * valid.to(flow_mag.dtype)).sum() / safe_count
    return torch.where(
        valid_count < 1,
        torch.zeros((), dtype=mean_mag.dtype, device=device),
        mean_mag,
    )
    # THE SECOND BLOCKER is the absence of `.item()` above.


@torch.no_grad()
def keyframe_decision(
    pose_enc: Tensor,
    last_kf_pose_enc: Tensor,
    depth: Optional[Tensor],
    hw: Optional[Tuple[int, int]],
    frames_since_kf: Tensor,
    *,
    flow_threshold: float,
    max_non_keyframe_gap: int,
    is_first_streaming_frame: bool,
    stride: int = 8,
) -> Tensor:
    """Decide whether the current frame is a keyframe. 0-d bool **device** tensor.

    Reproduces ``gct_stream_window.py:575-587`` and ``:1139-1144`` exactly:

        is_keyframe = first_streaming_frame
                      or flow_mag > flow_threshold
                      or frames_since_kf >= max_non_keyframe_gap

    Args:
        pose_enc: Current frame pose encoding [B, 1, 9].
        last_kf_pose_enc: Last keyframe's pose encoding [B, 1, 9].
        depth: Current depth map [B, 1, H, W, 1], or ``None`` if the model emits no depth.
        hw: (H, W) of the depth map. Required when ``depth`` is not None.
        frames_since_kf: 0-d int tensor. A tensor, not an int, because it depends on past
            decisions — the one genuinely dynamic input here.
        flow_threshold: Flow above this makes the frame a keyframe.
        max_non_keyframe_gap: Force a keyframe after this many non-keyframes.
        is_first_streaming_frame: Python bool. Trace-time loop-index arithmetic
            (``i == scale_frames``), not runtime data — deliberately not a tensor.
        stride: Depth subsampling stride, passed through to :func:`flow_magnitude`.

    Returns:
        0-d bool tensor. The caller reads exactly one scalar from it to choose commit vs
        rollback (design.md D3); this function itself never syncs.
    """
    device = pose_enc.device

    # THE THIRD BLOCKER, resolved on the host at trace time rather than in the graph. Whether
    # the model emits depth is a static property of the head configuration, so this `if` is
    # not a branch on runtime data and costs no sync. The value matches the original's
    # `flow_mag = flow_threshold + 1.0` fallback (gct_stream_window.py:572), which forces a
    # keyframe when depth is unavailable.
    if depth is None:
        flow_mag = torch.full((), flow_threshold + 1.0, dtype=torch.float32, device=device)
    else:
        if hw is None:
            raise ValueError("hw is required when depth is provided")
        flow_mag = flow_magnitude(pose_enc, last_kf_pose_enc, depth, hw, stride=stride)

    # The comparison happens in flow_mag's dtype (fp32), where the original compared Python
    # float64s: `mean_mag.item() > flow_threshold`. The two can only disagree when flow_mag
    # lands within one fp32 ulp of the threshold, since float64(fp32_value) is exact and only
    # the threshold's rounding differs. Not a tolerance question — V4 demands element-for-
    # element identity (FM8) — so V4 records the minimum observed |flow_mag - threshold|
    # margin. Identity is claimed on that measurement, not on this argument.
    thresh = torch.full((), flow_threshold, dtype=flow_mag.dtype, device=device)
    gap = torch.full(
        (), max_non_keyframe_gap, dtype=frames_since_kf.dtype, device=device
    )

    decision = (flow_mag > thresh) | (frames_since_kf >= gap)
    if is_first_streaming_frame:
        # Trace-time constant, so this is host-side folding, not a graph branch.
        return torch.ones((), dtype=torch.bool, device=device)
    return decision
