from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class KVCacheBackend(Protocol):
    """Structural interface satisfied by FlashInferKVCacheManager and
    SDPADictKVCacheBackend. Does not require inheritance — either class
    satisfies this Protocol by having matching methods/attributes.

    Scope: model-level cache operations, at the granularity GCTStream's six
    cache-touching methods need. Does NOT cover the fused per-layer
    append+evict+attend sequence the SDPA path performs inline inside
    SDPAAttention.forward — that remains untouched (see design.md Decision 1).
    """

    num_blocks: int
    """Number of transformer blocks this backend instance covers."""

    @property
    def num_frames(self) -> int:
        """Frames appended to block 0 (representative), mirroring
        FlashInferKVCacheManager.num_frames today. Read-only. Used by
        AggregatorStream._prepare_special_tokens to compute S_true without
        probing backend-specific attributes (has_flashinfer_cache /
        has_sdpa_cache booleans are removed)."""
        ...

    def append_frame(self, block_idx: int, k: Tensor, v: Tensor) -> None: ...

    def evict_frames(
        self,
        block_idx: int,
        scale_frames: int,
        sliding_window: int,
        cross_frame_special: bool = True,
        include_scale_frames: bool = True,
        camera_only: bool = False,
        num_register_tokens: int = 4,
    ) -> None: ...

    def execute_deferred_eviction(
        self, block_idx: int, scale_frames: int, sliding_window: int, **kwargs
    ) -> None: ...

    def execute_deferred_eviction_all_blocks(
        self, scale_frames: int, sliding_window: int
    ) -> None:
        """Runs execute_deferred_eviction for every block in one call — see
        Decision 1 / §5's 'Revised, corrected shape' for why GCTStream must
        call this instead of looping range(num_blocks) itself."""
        ...

    def rollback_last_frame(self, block_idx: int) -> None: ...

    def rollback_last_frame_all_blocks(self) -> None:
        """Runs rollback_last_frame for every block in one call. For
        SDPADictKVCacheBackend this is NOT num_blocks separate single-block
        calls — it is the verbatim-relocated SDPA rollback logic, executed
        exactly once regardless of num_blocks (see §2 and Decision 1's
        correctness argument)."""
        ...

    def compute_attention(self, block_idx: int, q: Tensor) -> Tensor: ...

    def reset(self) -> None: ...

    def get_cache_stats(self, block_idx: int = 0) -> dict: ...

    def get_skip_append(self) -> bool: ...
    def set_skip_append(self, flag: bool) -> None: ...
    def get_defer_eviction(self) -> bool: ...
    def set_defer_eviction(self, flag: bool) -> None: ...
