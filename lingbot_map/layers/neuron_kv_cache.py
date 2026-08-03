"""Neuron (Trainium2) fixed-capacity ring KV cache for the LingBot-Map aggregator trunk.

spec/neuron-port/design.md component C1 (``NeuronRingKVCacheBackend``). Satisfies the
existing ``KVCacheBackend`` Protocol verbatim, so no ``GCTStream``/``AggregatorStream`` call
site changes shape.

WHY THIS CLASS EXISTS
---------------------
Neuron compiles ahead of time to a fixed-shape NEFF. The GPU cache manager
(``flashinfer_cache.py``) is fixed-capacity in its *page pool*, but expresses the visible
region as a Python list of page IDs whose length changes every frame, and the SDPA fallback
grows tensors with ``torch.cat``. Neither survives NEFF compilation. This class keeps the
GPU manager's exact semantics while making every tensor SHAPE a compile-time constant:
append writes into a fixed slot, eviction advances a logical head, rollback rewinds a
pointer.

The ring bookkeeping is Phase 1's design, validated bit-identically against a
dynamically-growing reference across 8 keyframe policies x 6000 frames (74 ring-wrap events)
plus 496 targeted boundary cycles, before any hardware time
(``neuron_phase1/fixed_capacity_cache.py``).

THE TWO THINGS design.md's C1 SKETCH GETS WRONG
-----------------------------------------------
design.md specifies ``visible_kv -> (K_pad, V_pad, valid_len)`` with "K_pad/V_pad have the
COMPILE-TIME-CONSTANT shape [max_seqlen_k, H, D]". The buffer shape is constant, but the
obvious implementation of that contract is not, and the obvious masking is wrong:

1. **A variable-length gather is a variable SHAPE.** Writing only the live region
   (``K_pad[:n_live] = gather(live_slots)``) makes both the gather's index length and the
   destination slice a function of the frame index. ``n_live`` climbs every frame for the
   first 72 frames, so that is a new NEFF per frame through warmup, plus a dynamic-extent
   slice. The fix: **always gather all ``num_patch_slots`` slots**, in an order that puts
   the live ones first. The index tensor's *contents* change per frame; its *length* never
   does. Contents are data; only shapes trigger compilation.

2. **The ring's live window is a CIRCULAR run, which no prefix length can express.** Live
   window slots are ``[head, head+len) mod win_capacity``. Once the ring wraps, that is two
   disjoint physical runs, so ``valid_len`` used as a prefix bound over the physical buffer
   is simply incorrect — silently, with no crash. The gather is what makes a prefix bound
   legitimate: it *resolves* the wrap, permuting live slots into positions ``[0, n_live)``.
   Prefix masking is only correct downstream of that permutation.

Reordering keys is free here, and that is load-bearing for both fixes.
``softmax(q k^T) v`` is invariant to any permutation of the key axis applied jointly to k
and v, and this trunk has no causal mask, no sliding-window mask, and no positional bias on
the key axis (design.md F3), with RoPE already baked into k *before* the cache write
(``attention.py``, design.md F5). So the port is free to present keys in whatever order
makes shapes constant. It is not free to *include* a key it should not — hence the mask.

TWO GROUPS, NOT ONE BUFFER
--------------------------
The same freedom lets the visible region stay split the way the GPU cache already splits it,
instead of being concatenated into one 110_592-key buffer:

    patch group   : [num_patch_slots * patches_per_frame] , live prefix = n_frames * P
    special group : [max_total_frames * num_special]      , live prefix = special_token_count

Each group is independently prefix-masked, and C2's online-softmax combine merges the two
groups' tiles into one output. Two groups costs nothing numerically and avoids restacking
two streams that are already contiguous. It also means the special stream is not reshuffled
every frame merely because the patch region's live length changed underneath it — which is
what would force design.md's single-buffer version to rewrite the whole staging buffer per
frame.

Ordering *within* the patch group's live prefix still matches
``build_visible_page_table`` exactly (scale ++ window, oldest-first), because V3 compares
the port against the GPU element-for-element and an order difference there would be a silent
numerical divergence rather than a crash.

KNOWN COST, STATED PLAINLY
--------------------------
The gather is real work the GPU does not do: FlashInfer reads pages in place through a page
table, while ``attention_cte`` needs contiguous tensors. At production defaults it copies
``num_patch_slots * patches_per_frame`` = 101_306 tokens of K and V per block per frame —
about 9.3 GiB of copy traffic per frame across 24 blocks, roughly doubling KV bandwidth
versus attention alone. Since the objective is performance against the GPU, that is the
port's leading perf risk and the first thing to profile.

``headroom`` is the knob, and it defaults to 2 rather than the GPU's 16 for exactly this
reason — spare slots are free memory on the GPU but paid traversal here. See
:meth:`tiling_report` and the sweep in ``verify/neuron/check_ring_cache.py``.

A zero-copy path exists in principle — pass physical slot groups straight to the kernel with
a per-key liveness bias instead of gathering — but it depends on ``position_bias``
broadcasting over the query axis, which is unverified on hardware. Not taken here; noted so
the benchmark knows where the headroom is.
"""

from __future__ import annotations

import math
from typing import List, NamedTuple, Optional, Tuple

import torch
from torch import Tensor

from lingbot_map.layers.neuron_attention import MAX_SEQLEN, NeuronAttentionAdapter

#: Tile lengths are rounded up to this. Keeps every tile length friendly to the engine's
#: partition/free dimensions instead of an arbitrary ceil().
TILE_QUANTUM = 128


class TilePlan(NamedTuple):
    """How one key group splits into equal tiles of at most MAX_SEQLEN (design.md D1a)."""

    n_keys: int      #: live-capacity keys in the group, before tile padding
    n_tiles: int     #: compile-time-constant tile count
    tile: int        #: keys per tile; every tile is this long, so ONE NEFF shape per group
    padded: int      #: n_tiles * tile, the group buffer's allocated key extent

    @property
    def pad_waste(self) -> int:
        return self.padded - self.n_keys


def plan_tiles(n_keys: int, *, max_tile: int = MAX_SEQLEN) -> TilePlan:
    """Split ``n_keys`` into the fewest EQUAL tiles of at most ``max_tile``.

    Equal-length tiles are the point (design.md D1a): a group of ``n`` equal tiles compiles
    one NEFF and reuses it ``n`` times. Splitting 120_472 as 36864*3 + 9880 would need a
    second NEFF for the short final tile. So the tile length is ``ceil(n_keys / n_tiles)``
    rounded up to ``TILE_QUANTUM``, and the group buffer is padded to ``n_tiles * tile``.
    The padding is masked, never read as data.
    """
    if n_keys <= 0:
        raise ValueError(f"n_keys must be positive, got {n_keys}")
    n_tiles = math.ceil(n_keys / max_tile)
    while True:
        tile = TILE_QUANTUM * math.ceil(n_keys / n_tiles / TILE_QUANTUM)
        if tile <= max_tile:
            return TilePlan(n_keys, n_tiles, tile, n_tiles * tile)
        n_tiles += 1


class NeuronRingKVCacheBackend:
    """Fixed-capacity ring KV cache satisfying ``KVCacheBackend`` (design.md C1).

    Allocation per block, K and V each:

        patch slots   : [scale_frames + sliding_window + headroom, patches_per_frame, H, D]
        special       : [special_plan.padded, H, D]
        patch staging : [patch_plan.padded, H, D]

    At production defaults: 74 patch slots, 6744 special tokens (padded 6784), 101_376
    staging keys. Nothing is reallocated after construction.
    """

    def __init__(
        self,
        num_blocks: int,
        tokens_per_frame: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        num_special_tokens: int = 6,
        scale_frames: int = 8,
        sliding_window: int = 64,
        max_total_frames: int = 1124,
        # 2, not the GPU's 16. Deferred eviction leaves at most sliding_window + 1 frames in
        # the window, so +1 is the requirement and +2 is one slot of margin. The GPU's +16
        # is inherited from its page pool (flashinfer_cache.py:133), where spare pages cost
        # only memory. Here spare slots cost TRAVERSAL: they inflate the patch tile plan the
        # kernel walks every frame. Measured at production geometry: headroom 16 -> 127_616
        # keys traversed (21.18% over live) in 4 tiles; headroom 2 -> 108_160 (2.70%) in 3.
        # Verified by the replay sweep in verify/neuron/check_ring_cache.py.
        headroom: int = 2,
        attention_adapter: Optional[NeuronAttentionAdapter] = None,
        max_tile: int = MAX_SEQLEN,
    ):
        self.num_blocks = num_blocks
        self.tokens_per_frame = tokens_per_frame
        self.num_special_tokens = num_special_tokens
        self.patches_per_frame = tokens_per_frame - num_special_tokens
        if self.patches_per_frame <= 0:
            raise ValueError(
                f"tokens_per_frame={tokens_per_frame} <= "
                f"num_special_tokens={num_special_tokens}"
            )
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.scale_frames = scale_frames
        self.sliding_window = sliding_window
        self.max_total_frames = max_total_frames
        self.headroom = headroom

        # -- Fixed geometry: every one of these is a compile-time constant --------------
        self.win_capacity = sliding_window + headroom               # 66
        self.num_patch_slots = scale_frames + self.win_capacity     # 74
        self.special_capacity = max_total_frames * num_special_tokens   # 6744

        self.patch_plan = plan_tiles(
            self.num_patch_slots * self.patches_per_frame, max_tile=max_tile
        )                                                           # 101_306 -> 3 x 33_792
        self.special_plan = plan_tiles(self.special_capacity, max_tile=max_tile)
        #                                                           # 6_744  -> 1 x 6_784

        #: Live keys at steady state, for reference against design.md D2's 105_312. Not a
        #: shape: buffers are sized to capacity and the live count only ever masks.
        self.max_live_keys = (
            self.patches_per_frame * (scale_frames + sliding_window)
            + num_special_tokens * max_total_frames
        )

        # -- THE fixed allocation ------------------------------------------------------
        def _zeros(*shape: int) -> Tensor:
            return torch.zeros(*shape, dtype=dtype, device=device)

        self.patch_k: List[Tensor] = [
            _zeros(self.num_patch_slots, self.patches_per_frame, num_heads, head_dim)
            for _ in range(num_blocks)
        ]
        self.patch_v: List[Tensor] = [
            _zeros(self.num_patch_slots, self.patches_per_frame, num_heads, head_dim)
            for _ in range(num_blocks)
        ]
        self.special_k: List[Tensor] = [
            _zeros(self.special_plan.padded, num_heads, head_dim)
            for _ in range(num_blocks)
        ]
        self.special_v: List[Tensor] = [
            _zeros(self.special_plan.padded, num_heads, head_dim)
            for _ in range(num_blocks)
        ]
        # Staging for the gathered patch region; allocated once, and the gather writes its
        # whole live extent every call, so no stale data can survive.
        self._stage_k: List[Optional[Tensor]] = [None] * num_blocks
        self._stage_v: List[Optional[Tensor]] = [None] * num_blocks

        # -- Per-block bookkeeping. Host ints: this is cache STRUCTURE, not tensor data,
        #    and it only ever reaches the graph as index/mask CONTENTS. ----------------
        self.frame_count: List[int] = [0] * num_blocks
        self.win_len: List[int] = [0] * num_blocks    # live+speculative window frames
        self.win_head: List[int] = [0] * num_blocks   # ring index of the oldest live frame
        self.special_token_count: List[int] = [0] * num_blocks
        self._last_append_region: List[Optional[str]] = [None] * num_blocks

        self._defer_eviction = False
        self._skip_append = False

        self.attention = attention_adapter

    # ==============================================================================
    # KVCacheBackend Protocol
    # ==============================================================================

    @property
    def num_frames(self) -> int:
        """Frames appended to block 0, the representative block.

        ``AggregatorStream._prepare_special_tokens`` reads this to size the true sequence
        length (``stream.py:520-545``), so it must count *appended* frames — including a
        speculative one not yet committed — exactly as the GPU manager's ``frame_count``
        does.
        """
        return self.frame_count[0] if self.frame_count else 0

    def append_frame(self, block_idx: int, k: Tensor, v: Tensor) -> None:
        """Write one frame's K/V into its fixed slot.

        Token layout in, matching ``FlashInferKVCacheManager.append_frame``
        (``flashinfer_cache.py:203-228``): specials FIRST, then patches —
        ``[camera, reg0..regN, scale, patch0..patchP-1]``.

        Args:
            k: [tokens_per_frame, H, D] NHD
            v: [tokens_per_frame, H, D] NHD
        """
        n = self.num_special_tokens
        sp_k, patch_k = k[:n], k[n:]
        sp_v, patch_v = v[:n], v[n:]

        if patch_k.shape[0] != self.patches_per_frame:
            raise ValueError(
                f"block {block_idx}: expected {self.patches_per_frame} patch tokens, got "
                f"{patch_k.shape[0]} (tokens_per_frame={k.shape[0]})"
            )

        slot = self._alloc_patch_slot(block_idx)
        self.patch_k[block_idx][slot] = patch_k.to(self.dtype)
        self.patch_v[block_idx][slot] = patch_v.to(self.dtype)

        self._write_special_tokens(block_idx, sp_k.to(self.dtype), sp_v.to(self.dtype))
        self.frame_count[block_idx] += 1

    def evict_frames(
        self,
        block_idx: int,
        scale_frames: int,
        sliding_window: int,
        cross_frame_special: bool = True,
        include_scale_frames: bool = True,
        camera_only: bool = False,
        num_register_tokens: int = 4,
    ) -> None:
        """Recycle window slots beyond ``sliding_window``. No-op while deferring.

        The four trailing parameters exist to match the Protocol signature the GPU manager
        set; like the GPU manager (``flashinfer_cache.py:230-255``), this implementation
        reads none of them. Scale slots and specials are never evicted.
        """
        if self._defer_eviction:
            return
        self._evict_to_window(block_idx, sliding_window)

    def execute_deferred_eviction(
        self, block_idx: int, scale_frames: int, sliding_window: int, **kwargs
    ) -> None:
        """The COMMIT path: run the eviction that ``_defer_eviction`` skipped."""
        self._evict_to_window(block_idx, sliding_window)

    def execute_deferred_eviction_all_blocks(
        self, scale_frames: int, sliding_window: int
    ) -> None:
        for i in range(self.num_blocks):
            self.execute_deferred_eviction(i, scale_frames, sliding_window)

    def rollback_last_frame(self, block_idx: int) -> None:
        """The DISCARD path: undo the most recent ``append_frame`` for this block.

        Reverses all three sub-operations in the same order as
        ``FlashInferKVCacheManager.rollback_last_frame`` (``flashinfer_cache.py:269-302``).
        Must be called before any eviction for that frame — i.e. while eviction is deferred,
        which is what ``block.py:243-278`` arranges. Because of that, rollback only ever
        removes a frame written into headroom, so it never has to recover a slot the ring
        already recycled. That is the Phase-1-validated invariant.

        The GPU manager returns the page to its free list; the ring instead rewinds the
        write pointer, leaving stale K/V in the slot for the next append to overwrite. Stale
        contents are unreachable either way: the slot is not live, so the gather places it in
        the dead suffix, outside the mask's prefix.
        """
        if self.frame_count[block_idx] <= 0:
            raise AssertionError(f"block {block_idx}: cannot rollback, frame_count is 0")

        # 1) Undo the patch slot.
        if self._last_append_region[block_idx] == "window":
            if self.win_len[block_idx] <= 0:
                raise AssertionError(f"block {block_idx}: window underflow during rollback")
            self.win_len[block_idx] -= 1    # rewind the write pointer; head is untouched
        # Scale region: nothing to rewind but the frame count — the next append recomputes
        # the same slot from frame_count and overwrites it.

        # 2) Undo the special tokens.
        new_count = self.special_token_count[block_idx] - self.num_special_tokens
        if new_count < 0:
            raise AssertionError(
                f"block {block_idx}: special_token_count underflow "
                f"({self.special_token_count[block_idx]} - {self.num_special_tokens})"
            )
        self.special_token_count[block_idx] = new_count

        # 3) Decrement the frame count.
        self.frame_count[block_idx] -= 1
        self._last_append_region[block_idx] = None

    def rollback_last_frame_all_blocks(self) -> None:
        for i in range(self.num_blocks):
            self.rollback_last_frame(i)

    def compute_attention(self, block_idx: int, q: Tensor) -> Tensor:
        """Attend the visible region. The same call ``GCTStream`` makes against FlashInfer.

        Args:
            q: [tokens_per_frame, H, D] NHD
        Returns:
            [tokens_per_frame, H, D], in ``q``'s dtype.
        """
        if self.frame_count[block_idx] == 0:
            # Mirrors flashinfer_cache.py's empty-cache return.
            return torch.zeros_like(q)
        if self.attention is None:
            raise RuntimeError(
                "compute_attention needs an attention_adapter. Either construct with "
                "attention_adapter=NeuronAttentionAdapter(...), or drive visible_groups() "
                "directly (what verify/neuron/ does at the desk, with no Neuron stack)."
            )
        return self.attention.attend_groups(q, self.visible_groups(block_idx))

    def reset(self) -> None:
        """Reset per-block state for a new sequence. Buffers are NOT reallocated."""
        for i in range(self.num_blocks):
            self.frame_count[i] = 0
            self.win_len[i] = 0
            self.win_head[i] = 0
            self.special_token_count[i] = 0
            self._last_append_region[i] = None
        self._defer_eviction = False
        self._skip_append = False

    def get_cache_stats(self, block_idx: int = 0) -> dict:
        """Same first five keys as the GPU manager's, so GPU and Neuron runs diff directly.

        ``scale_pages``/``live_pages``/``free_pages`` deliberately keep the GPU's page
        vocabulary even though this cache has no pages — design.md's Observability section
        wants a diffable dict, not a prettier one. Ring-specific counters follow.
        """
        n_scale = min(self.frame_count[block_idx], self.scale_frames)
        return {
            "frame_count": int(self.frame_count[block_idx]),
            "scale_pages": int(n_scale),
            "live_pages": int(self.win_len[block_idx]),
            "free_pages": int(self.win_capacity - self.win_len[block_idx]),
            "special_tokens": int(self.special_token_count[block_idx]),
            # Neuron-specific (design.md Observability).
            "win_head": int(self.win_head[block_idx]),
            "live_keys": int(self.visible_len(block_idx)),
            "traversed_keys": self.patch_plan.padded + self.special_plan.padded,
            "n_tiles": self.patch_plan.n_tiles + self.special_plan.n_tiles,
        }

    def get_skip_append(self) -> bool:
        return self._skip_append

    def set_skip_append(self, flag: bool) -> None:
        self._skip_append = flag

    def get_defer_eviction(self) -> bool:
        return self._defer_eviction

    def set_defer_eviction(self, flag: bool) -> None:
        self._defer_eviction = flag

    # ==============================================================================
    # Additive: what the Neuron attention path needs beyond the Protocol
    # ==============================================================================

    def visible_len(self, block_idx: int) -> int:
        """Live key count at this instant (host int; diagnostics and desk checks).

        Grows by ``num_special_tokens`` every frame forever (design.md F2) — which is why
        buffers are sized to capacity and masked, rather than sized to the live length.
        """
        return (
            self.patches_per_frame * self._n_visible_frames(block_idx)
            + self.special_token_count[block_idx]
        )

    def _n_visible_frames(self, block_idx: int) -> int:
        """Patch frames the attention sees right now: all scale slots + the WHOLE window.

        ``win_len`` is deliberately NOT clamped to ``sliding_window``. The GPU's
        ``build_visible_page_table`` lists the entire ``live_window_patch_pages`` deque
        (``flashinfer_cache.py:459-471``), and eviction is the only thing that bounds it. So
        while eviction is deferred — which is exactly when attention runs, per
        ``block.py:243-278`` — the window legitimately holds ``sliding_window + 1`` frames
        and all of them are attended. Clamping here silently drops the speculative frame's
        own patches from its own attention, changing the numerics with no error.

        Verified against the real GPU manager: clamping diverges at frame 9 of the V3 replay
        (9 visible patch frames vs 8).
        """
        n_scale = min(self.frame_count[block_idx], self.scale_frames)
        return n_scale + self.win_len[block_idx]

    def visible_index(self, block_idx: int) -> Tensor:
        """Permutation of ALL ``num_patch_slots`` slots: live ones first, in logical order.

        Fixed length always — that is the point (module docstring, bug #1). The live prefix
        is ``scale ++ window`` oldest-first, matching ``build_visible_page_table``
        (``flashinfer_cache.py:459-471``) element for element. The dead suffix is whatever
        remains, ascending; its contents are masked out, so its order is arbitrary and only
        its length matters.
        """
        n_scale = min(self.frame_count[block_idx], self.scale_frames)
        live = list(range(n_scale))
        # The whole window, oldest-first from win_head — NOT clamped to sliding_window; see
        # _n_visible_frames for why the speculative overflow frame must stay visible.
        live += [self._win_slot(block_idx, j) for j in range(self.win_len[block_idx])]
        live_set = set(live)
        dead = [s for s in range(self.num_patch_slots) if s not in live_set]
        return torch.tensor(live + dead, dtype=torch.long, device=self.device)

    def visible_patch_kv(self, block_idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Gather the patch region into fixed-shape staging.

        Returns ``(K, V, valid_len)`` where K/V are ``[patch_plan.padded, H, D]`` — a
        compile-time constant — and ``valid_len`` is a 0-d int32 **device** tensor holding
        ``n_visible_frames * patches_per_frame``.

        ``valid_len`` must never be a Python int: it changes every frame (design.md F2), and
        as an int it would bake the frame's length into the graph and retrigger NEFF
        compilation each frame.
        """
        k_stage, v_stage = self._staging(block_idx)
        idx = self.visible_index(block_idx)
        P = self.patches_per_frame
        n_gathered = self.num_patch_slots * P
        flat = (n_gathered, self.num_heads, self.head_dim)

        # Fixed-extent gather into a fixed-extent destination. Frame-major, so the f-th
        # visible frame's patches land at [f*P, (f+1)*P) — the GPU's page ordering.
        k_stage[:n_gathered] = self.patch_k[block_idx].index_select(0, idx).reshape(flat)
        v_stage[:n_gathered] = self.patch_v[block_idx].index_select(0, idx).reshape(flat)
        # [n_gathered, padded) is tile padding: zeroed at construction, never written again,
        # and unreachable since valid_len <= n_gathered.

        valid = torch.tensor(
            self._n_visible_frames(block_idx) * P, dtype=torch.int32, device=self.device
        )
        return k_stage, v_stage, valid

    def visible_special_kv(self, block_idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """The special stream: no gather needed, it is already a live prefix.

        Specials are append-only and never evicted (``flashinfer_cache.py:243``), so slot
        order *is* frame order and ``special_token_count`` is a correct prefix bound.
        """
        valid = torch.tensor(
            self.special_token_count[block_idx], dtype=torch.int32, device=self.device
        )
        return self.special_k[block_idx], self.special_v[block_idx], valid

    def visible_groups(
        self, block_idx: int
    ) -> List[Tuple[Tensor, Tensor, Tensor, TilePlan]]:
        """Both key groups, in ``build_visible_page_table`` order: patches then specials.

        Each entry is ``(K, V, valid_len, plan)``. C2 tiles each group by its own plan and
        merges every tile's online-softmax statistics into one output. Key order across
        groups cannot affect the result (module docstring), but patches-then-specials is
        kept so a dump of the concatenated live keys matches the GPU's visible sequence.
        """
        pk, pv, pvalid = self.visible_patch_kv(block_idx)
        sk, sv, svalid = self.visible_special_kv(block_idx)
        return [
            (pk, pv, pvalid, self.patch_plan),
            (sk, sv, svalid, self.special_plan),
        ]

    def visible_kv_concat(self, block_idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """design.md's original single-buffer form. VERIFICATION ONLY.

        Concatenates the two groups' live keys into one contiguous ``[n_live, H, D]``, which
        is what V3 compares against ``flashinfer_cache``'s visible sequence. Deliberately
        NOT fixed-shape and NOT for the Neuron path — its shape is a function of the frame
        index, which is exactly what :meth:`visible_groups` exists to avoid.
        """
        pk, pv, pvalid = self.visible_patch_kv(block_idx)
        sk, sv, svalid = self.visible_special_kv(block_idx)
        np_, ns = int(pvalid.item()), int(svalid.item())
        return (
            torch.cat([pk[:np_], sk[:ns]], dim=0),
            torch.cat([pv[:np_], sv[:ns]], dim=0),
            torch.tensor(np_ + ns, dtype=torch.int32, device=self.device),
        )

    def tiling_report(self) -> dict:
        """What the fixed shapes cost, for the perf-vs-GPU comparison.

        ``traversed_keys`` is what the kernel walks every frame regardless of how many
        frames are live; ``live_max_keys`` is design.md D2's steady-state 105_312.
        ``headroom`` is the knob: it inflates ``num_patch_slots``, so it pays for rollback
        safety in traversed keys.
        """
        traversed = self.patch_plan.padded + self.special_plan.padded
        return {
            "patch_plan": self.patch_plan._asdict(),
            "special_plan": self.special_plan._asdict(),
            "num_patch_slots": self.num_patch_slots,
            "headroom": self.headroom,
            "traversed_keys": traversed,
            "live_max_keys": self.max_live_keys,
            "overhead_pct": round(
                100.0 * (traversed - self.max_live_keys) / self.max_live_keys, 2
            ),
            "n_neff_shapes": 1 if self.patch_plan.tile == self.special_plan.tile else 2,
            "gather_bytes_per_frame_per_block": (
                2 * self.num_patch_slots * self.patches_per_frame
                * self.num_heads * self.head_dim
                * torch.empty((), dtype=self.dtype).element_size()
            ),
        }

    # ==============================================================================
    # Internals
    # ==============================================================================

    def _staging(self, block_idx: int) -> Tuple[Tensor, Tensor]:
        if self._stage_k[block_idx] is None:
            self._stage_k[block_idx] = torch.zeros(
                self.patch_plan.padded, self.num_heads, self.head_dim,
                dtype=self.dtype, device=self.device,
            )
            self._stage_v[block_idx] = torch.zeros_like(self._stage_k[block_idx])
        return self._stage_k[block_idx], self._stage_v[block_idx]

    def _win_slot(self, block_idx: int, i: int) -> int:
        """Physical patch slot of the i-th appended live window frame (0 = oldest)."""
        return self.scale_frames + (self.win_head[block_idx] + i) % self.win_capacity

    def _alloc_patch_slot(self, block_idx: int) -> int:
        if self.frame_count[block_idx] < self.scale_frames:
            self._last_append_region[block_idx] = "scale"
            return self.frame_count[block_idx]
        if self.win_len[block_idx] >= self.win_capacity:
            # Phase 1's invariant guard, retained deliberately. Phase 1's boundary sweep
            # establishes this cannot fire under any of 8 keyframe policies; if it does, an
            # invariant is broken, and silent corruption is worse than a raise
            # (design.md FM5).
            raise RuntimeError(
                f"block {block_idx}: window ring overflow: win_len="
                f"{self.win_len[block_idx]} >= capacity={self.win_capacity}. Eviction is "
                f"not keeping up with appends; headroom={self.headroom} is insufficient "
                f"for this usage pattern."
            )
        slot = self._win_slot(block_idx, self.win_len[block_idx])
        self.win_len[block_idx] += 1
        self._last_append_region[block_idx] = "window"
        return slot

    def _evict_to_window(self, block_idx: int, sliding_window: int) -> None:
        while self.win_len[block_idx] > sliding_window:
            self.win_head[block_idx] = (self.win_head[block_idx] + 1) % self.win_capacity
            self.win_len[block_idx] -= 1

    def _write_special_tokens(self, block_idx: int, sp_k: Tensor, sp_v: Tensor) -> None:
        """Append this frame's specials to the append-only special stream.

        Flat and contiguous — no pages, so no page-boundary straddling and no partial-page
        waste. The GPU's special pool rounds up to whole 1369-token pages and reserves 21 of
        them for 6744 tokens; this reserves 6784. That is where C1's cache comes out smaller
        than the GPU's (design.md Data Model).

        **Matched-failure requirement (design.md F2/FM4).** Exhaustion must raise at the same
        frame count as the GPU's assert (``flashinfer_cache.py:554-557``) and name
        ``max_total_frames``. Divergence here is a behavioral difference, not an
        implementation detail — V5(b) tests it. The bound is the LOGICAL capacity, not the
        tile-padded buffer extent, or the port would accept frames the GPU rejects.
        """
        n = self.num_special_tokens
        start = self.special_token_count[block_idx]
        end = start + n
        if end > self.special_capacity:
            raise AssertionError(
                f"block {block_idx}: special page pool exhausted at "
                f"special_token_count={start}. Increase max_total_frames "
                f"(currently {self.max_total_frames})."
            )
        self.special_k[block_idx][start:end] = sp_k
        self.special_v[block_idx][start:end] = sp_v
        self.special_token_count[block_idx] = end
