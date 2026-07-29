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


class SDPADictKVCacheBackend:
    """Thin facade over AggregatorStream.kv_cache (a dict). Reproduces —
    verbatim, not redesigned — the dict-mutation logic GCTStream inlines
    today. Does NOT change what SDPAAttention.forward/CausalAttention.forward
    do with the same underlying dict; this class is used only by the six
    model-level GCTStream methods, never passed as `kv_cache=` into the
    per-layer forward path."""

    def __init__(self, kv_cache: dict, num_blocks: int):
        self._kv_cache = kv_cache      # same dict object AggregatorStream owns
        self.num_blocks = num_blocks   # == AggregatorStream.depth

    @property
    def num_frames(self) -> int:
        k0 = self._kv_cache.get("k_0")
        return 0 if k0 is None else k0.shape[2]

    def append_frame(self, block_idx, k, v) -> None:
        raise NotImplementedError(
            "Not exercised in this pass: the SDPA per-layer forward path "
            "performs append+evict+attend fused inline in "
            "SDPAAttention.forward and is not decomposed by this refactor. "
            "See design.md Decision 1."
        )

    def evict_frames(
        self, block_idx, scale_frames, sliding_window,
        cross_frame_special=True, include_scale_frames=True,
        camera_only=False, num_register_tokens=4,
    ) -> None:
        raise NotImplementedError("See append_frame docstring.")

    def execute_deferred_eviction(self, block_idx, scale_frames, sliding_window, **kw) -> None:
        # Faithful reproduction of today's gap: gct_stream_window.py's
        # _execute_deferred_eviction has no SDPA branch at all. No-op.
        return None

    def execute_deferred_eviction_all_blocks(self, scale_frames, sliding_window) -> None:
        # Same no-op, once — Gap B's absence applies uniformly, not per-block.
        return None

    def rollback_last_frame(self, block_idx: int) -> None:
        raise NotImplementedError(
            "Single-block rollback has no meaning for the SDPA dict backend — "
            "the dict is not block-scoped the way FlashInferKVCacheManager's "
            "pages are. Use rollback_last_frame_all_blocks(), the Protocol "
            "method GCTStream actually calls. See Decision 1."
        )

    def rollback_last_frame_all_blocks(self) -> None:
        # Verbatim relocation of GCTStream._rollback_last_frame's SDPA branch.
        # Executed EXACTLY ONCE regardless of num_blocks — this is the specific
        # correctness subtlety Decision 1 calls out (a naive per-block loop
        # calling this num_blocks times would over-trim the cache).
        for key in list(self._kv_cache.keys()):
            v = self._kv_cache[key]
            if key.startswith(("k_", "v_")) and v is not None and torch.is_tensor(v):
                if v.dim() >= 3 and v.shape[2] > 1:
                    self._kv_cache[key] = v[:, :, :-1]
                elif v.dim() >= 3:
                    self._kv_cache[key] = None

    def compute_attention(self, block_idx, q):
        raise NotImplementedError("See append_frame docstring.")

    def reset(self) -> None:
        # Verbatim relocation of AggregatorStream.clean_kv_cache's dict branch.
        for key in list(self._kv_cache.keys()):
            self._kv_cache[key] = False if key == "_skip_append" else None

    def get_cache_stats(self, block_idx: int = 0) -> dict:
        k0 = self._kv_cache.get("k_0")
        return {
            "k_shape": tuple(k0.shape) if torch.is_tensor(k0) else None,
            "skip_append": self._kv_cache.get("_skip_append", False),
        }

    def get_skip_append(self) -> bool:
        return self._kv_cache.get("_skip_append", False)

    def set_skip_append(self, flag: bool) -> None:
        self._kv_cache["_skip_append"] = flag

    def get_defer_eviction(self) -> bool:
        return self._kv_cache.get("_defer_eviction", False)

    def set_defer_eviction(self, flag: bool) -> None:
        # Faithful reproduction of Gap B: this key is written but read by
        # nothing downstream. Do not add a reader as part of this pass.
        self._kv_cache["_defer_eviction"] = flag
