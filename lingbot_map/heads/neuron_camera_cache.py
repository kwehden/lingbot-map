"""Neuron (Trainium2) fixed-capacity KV cache for ``CameraCausalHead``'s 16 streams.

spec/neuron-port/design.md component C5 (``NeuronCameraCache``). This is the camera-head
analogue of C1 (``NeuronRingKVCacheBackend``): it replaces the third unbounded cache of
design.md F6 with a preallocated buffer set whose every tensor SHAPE is a compile-time
constant.

WHY THIS CLASS EXISTS
---------------------
``CameraCausalHead.kv_cache`` is a Python ``list`` of ``num_iterations`` dicts, each holding
``k_{j}``/``v_{j}`` for ``j in range(trunk_depth)`` plus ``"_skip_append"``
(``camera_head.py:284-291``). Every keyframe grows all 32 of those tensors with ``torch.cat``
on the frame axis (``attention.py:239-240``) — **32 ``torch.cat`` calls per streaming frame**,
each producing a new tensor one frame longer than the last. On Neuron that is a new NEFF per
frame, forever. This class keeps the semantics and makes the shape constant: the ``cat``
becomes an ``index_copy_`` at a device-tensor cursor into a buffer of fixed extent
``max_total_frames``, and the live region is expressed as a device-resident ``valid_len``
rather than by the buffer's length (design.md :711-712, :776-802).

ALLOCATION
----------
One buffer per ``(iteration, block)`` slot, per K and V (design.md :704-709, Data Model :741)::

    [B, num_heads, padded_frames, 1, head_dim]           # [1,16,1152,1,128] fp32
    live capacity   = max_total_frames = 1124 committed frames (+1 speculative row)
    padded_frames   = plan_tiles(1125).padded = 1152      # 1 tile x 1152, TILE_QUANTUM=128
    16 slots x K only  = 16 x 1152 x 16 x 128 x 4 B  ~=  151 MB   (147 MB live + 4 MB pad)
    16 slots x (K+V)                                 ~=  302 MB   (design.md row: ~295 MB)

**The frame axis is tile-padded, and it has to be** (design.md D1a, C1's
``neuron_kv_cache.py:196-199``). C2's kernel seam accepts a key axis of exactly
``n_tiles x tile`` with ``tile`` a multiple of ``TILE_QUANTUM = 128``, and
``[t for t in range(128, 36865, 128) if 1124 % t == 0] == []`` — **no legal quantized tile can
ever make a 1124-key buffer acceptable**, so a buffer allocated to the raw
``max_total_frames`` cannot be attended at all: every adapter tile choice raises out of
``attend``/``attend_groups``' shape check. That is why :attr:`plan` is computed in
:meth:`__init__` and the frame axis is ``plan.padded``, exactly as C1 allocates
``patch_plan.padded`` staging rows. The 28 padded rows are never written, are outside
``valid_len``, and are **not** capacity: the exhaustion guard still fires at
``max_total_frames`` committed frames. design.md's 147/295 MB Data Model row states the live
figure; the 2.5% padding delta is reported by :meth:`memory_report` and gated by the harness
rather than hidden (Deltas Owed #12).

16 is structural, not configured: ``trunk_depth=4`` (``camera_head.py:167``) and
``num_iterations=4`` (``camera_head.py:174``) are never overridden by any of the three
``CameraCausalHead(`` call sites. Both are constructor arguments here anyway, so the desk
harness can run a small geometry.

THREE PLACES C5 DELIBERATELY DIFFERS FROM ITS C1 TEMPLATE
---------------------------------------------------------
1. **fp32, not bf16.** Not a preference — forced. ``gct_base.py:165-177`` calls the camera
   head under ``torch.amp.autocast('cuda', enabled=False)`` with ``[t.float() for t in
   aggregated_tokens_list]``. A bf16 buffer would silently downcast every stored key and V5(a)
   is a bit-exactness gate. Do not "fix" this to match C1.

2. **The write cursor is a DEVICE TENSOR, where C1's is a host int.** C1's rule
   (``neuron_kv_cache.py:234-236``) is that cache *structure* may live in host ints because it
   only ever reaches the graph as index/mask *contents*; C1 relies on that with a host-int
   slot index whose range is ``num_patch_slots = 74``, so the worst case is 74 write-offset
   specializations. C5's cursor ranges over ``max_total_frames = 1124``. Copying C1's host-int
   write verbatim would be 1124 specializations, which is why design.md :711-712 says
   *indexed write at a device-tensor cursor* for C5 and says no such thing for C1. So:
   :meth:`append` writes with ``index_copy_(2, cursor, k)``, advances with
   ``cursor = cursor + n`` (never ``+=`` on a tensor it also reads), and never calls
   ``.item()``. Host-int mirrors exist for diagnostics and for the capacity guard only.

3. **No gather, and no special stream.** C1's ``index_select`` permutation
   (``neuron_kv_cache.py:497-498``) exists solely to resolve its ring's *circular* live
   window: once the ring wraps, no prefix bound over the physical buffer is correct. C5 is
   append-only with no eviction (below), so slot order *is* frame order and a physical prefix
   is legitimate — the same argument C1 makes for its own append-only special stream
   (``neuron_kv_cache.py:507-512``). Importing a gather C5 does not need would be pure
   traversal cost. And ``k_{j}_special`` is created *only* inside the eviction body
   (``attention.py:330-336``), so camera streams never have special keys at all; the
   special-prepend at ``attention.py:262-272`` is dead here.

WHY THERE IS NO EVICTION, AND WHAT V6 GUARDS
--------------------------------------------
``_apply_kv_cache_eviction_causal``'s entire body is gated on
``kv_cache[f"k_{global_idx}"].shape[3] > 1`` (``attention.py:307``), and camera streams always
have ``shape[3] == 1``, so the body is unreachable. Measured on the real head: 336 calls, body
entered 0 times, no ``_special`` key ever created. Capacity is therefore purely
``max_total_frames`` (design.md :700-702, :720-723) — no policy invention, no config flag.

design.md F6 attributes ``shape[3] == 1`` to ``frame_seqlen=1`` (``camera_head.py:358``). That
is not the mechanism: ``frame_seqlen`` is not read anywhere in ``attention.py``'s cached
branch. The real invariant is two-step, and worth stating because a future change could pass
F6's stated reasoning while breaking the fact:

  a. The FIRST causal call stores ``k.view(B,H,nfpb, N//nfpb, D)`` using the **caller's**
     ``num_frame_per_block`` (``attention.py:232-234``). Phase 1 passes
     ``num_frame_per_block=scale_frames`` with ``S == scale_frames``
     (``gct_stream_window.py:526-533``), so ``N//nfpb == 1``.
  b. Every LATER append rederives ``num_frame_per_block = k.shape[2] //
     kv_cache[...].shape[3]`` (``attention.py:236``) — so ``shape[3]`` is self-perpetuating
     from (a) forever.

The fragile premise is ``num_frame_per_block == S`` on the first Phase-1 call, not
``frame_seqlen``. Feeding ``num_frame_per_block=4, S=8`` on the first call yields
``shape[3] == 2`` and switches GPU-side eviction ON. :meth:`append` therefore hard-checks the
incoming ``shape[3]`` against the **literal 1**, not against :attr:`frame_seqlen`: a guard
that compares against its own config value is a config echo, not an invariant — relax the
constructor and the write-time check relaxes with it, which is exactly how this guard was
measured to be neuterable by editing one line. ``self.slot_shape[3] == 1`` is asserted after
allocation for the same reason. V6 asserts ``shape[3] == 1`` on the GPU reference for all 16
streams at every frame, and the port-side gates in
``verify/neuron/check_c5_camera_cache.py`` assert it on the live C5 instance, on the
constructor (``frame_seqlen=2`` must raise) and on the write (``k.shape[3] == 2`` must
raise), with an ``accept_shape3_gt_1`` injection proving those gates discriminate.

GAP A IS PRESERVED, NOT FIXED — DO NOT ADD ``rollback_last_frame``
------------------------------------------------------------------
``CameraCausalHead`` has no ``rollback_last_frame`` (spec/design.md Gap A; measured:
``hasattr(head, 'rollback_last_frame') is False``). The two call sites that would use one,
``gct_stream_window.py:397-398`` and ``gct_stream_window_v2.py:466-467``, are
``hasattr``-guarded, i.e. permanently-false branches. So in the flow-keyframe loop
(``gct_stream_window.py:556-600``) a non-keyframe frame is written into the camera cache by
the forward pass and **never removed**: ``_rollback_last_frame`` rewinds the aggregator and
``total_frames_processed``, but the camera cache keeps it.

design.md :715-718 requires the port to reproduce that, and V6 asserts the *absence* of the
method. There is deliberately no rollback below — but be precise about **what protects that**,
because the obvious reading is wrong and inverts the conclusion:

  * The ``hasattr`` guards at ``gct_stream_window.py:397`` and ``gct_stream_window_v2.py:466``
    interrogate ``self.camera_head``, i.e. the **HEAD**, not the cache. They are permanently
    false because ``CameraCausalHead`` itself lacks the method, and they stay false no matter
    what the cache object exposes. Verified: giving a ``NeuronCameraCache`` subclass a
    ``rollback_last_frame`` and assigning it as ``head.kv_cache`` leaves
    ``hasattr(camera_head, 'rollback_last_frame') is False`` and the guard does not fire.
    So "adding a rollback here would make the guard start firing" would be FALSE, and a
    future implementer who traced the guard from such a claim could reasonably conclude the
    opposite ("the guard tests the head, my cache is not the head, therefore harmless").
  * What actually holds Gap A on the port side is the harness: the gate
    ``no_rollback_method_on_C5`` (``check_c5_camera_cache.py``) asserts
    ``not hasattr(c5, 'rollback_last_frame')`` and FAILS if one is added here, and
    ``hasattr_rollback_is_False_before_and_after`` asserts V6's condition on the real head
    instance.
  * A rollback added to C5 would only be *reached* if the deferred wiring follow-on also added
    a forwarding ``rollback_last_frame`` to ``CameraCausalHead``. That is the coupled change to
    refuse; the cache-side method is the half that makes it look harmless.

WHAT THIS CLASS DOES NOT OWN
----------------------------
``camera_head.frame_idx``. ``clean_kv_cache`` zeroes both ``kv_cache`` and ``frame_idx``
(``camera_head.py:251-254``), but ``frame_idx`` drives ``is_scale_frames``
(``camera_head.py:314``) and 3D-RoPE positions (``:322-331``) — that is head state, not cache
state. It is also incremented **unconditionally** (``camera_head.py:374-375``, no
``_skip_append`` guard, unlike the aggregator's ``total_frames_processed`` at
``stream.py:535``), so it deliberately drifts from the stored-frame count by the non-keyframe
count. :meth:`reset` must not touch it and must not "fix" the drift; see the 2026-07-29
amendment in the sibling ``spec/design.md:22-39`` for the hazard.

CAPACITY: AN ASYMMETRY design.md DOES NOT RECORD
------------------------------------------------
C1's exhaustion guard has a GPU counterpart (``flashinfer_cache.py:554-557``), which is what
lets V5(b) say "the GPU raises no earlier". **The camera-head GPU path has no assert at all**
— ``attention.py:239-240`` ``cat``s until the host OOMs; measured growth to 62 frames with no
complaint, and ``camera_head.py:189``'s ``max_frame_num=1024`` feeds only ``rope3d``'s
``patch_size`` (``:215``). So the correct statement is *not* design.md FM4's matched failure:
C5 raises at exactly ``max_total_frames`` **committed** frames, naming it, and the GPU never
raises — which is the unbounded growth C5 exists to bound. Deltas Owed #12.

The asymmetry has a second, subtler half, and getting it wrong turns a bound into a spurious
abort. Capacity is charged for **commits only**, so the guard is inside the keyframe branch of
:meth:`append`. A non-keyframe's speculative frame is written at the cursor and never
committed, and the GPU's own non-keyframe branch (``attention.py:249-256``) is a purely LOCAL
``torch.cat`` that never mutates ``kv_cache[k_j]`` — so the GPU can process non-keyframes
indefinitely at any dict depth, with no error and no depth change. Charging capacity for that
frame made C5 raise in a state where the GPU provably cannot fail: at ``max_total_frames``
committed frames a following non-keyframe aborted inference while the GPU attended one extra
key and kept its depth unchanged.

Hence :attr:`spec_headroom` — **one row allocated above capacity**, so the speculative frame
always has a physical row of its own and ``valid_len == cursor + pending`` still names exactly
where the write landed. No clamping (which would put ``valid_len`` and the write offset in
disagreement) and no aliasing of a committed row. The row is free: the frame axis is
tile-padded to 1152 anyway, so 1124 + 1 costs nothing. The gates
``non_keyframe_at_capacity_does_not_raise`` and ``keyframe_at_capacity_still_raises`` pin both
halves.

DESK-RUNNABLE WITH NO NEURON STACK
----------------------------------
Nothing here imports ``nkilib`` or ``torch_xla``, and importing this module imports nothing but
``torch``: the one cross-module dependency, ``plan_tiles`` (C1's tiling arithmetic, reused
rather than copied), is imported inside :meth:`__init__`, and ``plan_tiles`` itself is pure
integer arithmetic. The attention adapter is an optional constructor argument and
:meth:`compute_attention` raises a message naming the desk alternative, mirroring
``neuron_kv_cache.py:376-381``; the kernel import itself stays gated in
``NeuronAttentionAdapter._load_kernel`` (``neuron_attention.py:382-397``). Verified by
``verify/neuron/check_c5_camera_cache.py``, which drives the real
``CausalAttention.forward`` cache mutation (``attention.py:229-256``) as the reference on CPU
or GPU with nothing stubbed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover - typing only; never imported at runtime
    # Deliberately not a runtime import. neuron_attention is pure torch and safe to import,
    # but keeping it out of C5's import graph is what makes "importing this module cannot
    # pull in a Neuron stack" true by construction rather than by inspection.
    from lingbot_map.layers.neuron_attention import NeuronAttentionAdapter


class NeuronCameraCache:
    """Fixed-capacity camera-head KV cache, one buffer set per ``(iteration, block)`` slot.

    There is no Protocol to satisfy: ``kv_cache_backend.py``'s ``@runtime_checkable
    KVCacheBackend`` covers the aggregator trunk only, and ``CameraCausalHead.kv_cache`` is a
    raw ``list[dict]`` with keys ``k_{j}`` / ``v_{j}`` / ``_skip_append``
    (``camera_head.py:285-291``). That dict-key surface *is* C5's interface contract, so the
    accessors below mirror it (:meth:`get_skip_append` / :meth:`set_skip_append`, as
    ``neuron_kv_cache.py:416-420`` does) and the deferred seam wiring is a substitution rather
    than a rewrite.

    No base class, by design: design.md's Rejected Abstractions (:1008-1013) kills a shared
    ``FixedCapacityCache`` for C1+C5 because rank, dtype, eviction policy and stream count all
    differ — "the only common member would be 'an integer cursor'".

    Allocation, per slot, K and V each::

        [batch_size, num_heads, padded_frames, frame_seqlen, head_dim]

    ``padded_frames >= max_total_frames`` is the tile-padded extent C2's kernel seam requires
    (:attr:`plan`; see the module docstring's ALLOCATION note). Capacity — what
    :meth:`append` refuses to exceed — is still ``max_total_frames``; the pad rows are never
    written and never inside ``valid_len``.

    Nothing is reallocated after construction, including by :meth:`reset`.
    """

    def __init__(
        self,
        num_iterations: int,
        trunk_depth: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        *,
        batch_size: int = 1,
        max_total_frames: int = 1124,
        frame_seqlen: int = 1,
        # fp32, NOT C1's bf16, and this asymmetry is deliberate: the camera head runs under
        # `torch.amp.autocast('cuda', enabled=False)` on `.float()` inputs
        # (gct_base.py:165-177, design.md F6). A bf16 buffer would silently downcast every
        # stored key. Do not change this to match C1.
        dtype: torch.dtype = torch.float32,
        attention_adapter: Optional["NeuronAttentionAdapter"] = None,
    ):
        if num_iterations <= 0 or trunk_depth <= 0:
            raise ValueError(
                f"num_iterations={num_iterations} and trunk_depth={trunk_depth} must both "
                f"be positive"
            )
        if max_total_frames <= 0:
            raise ValueError(f"max_total_frames must be positive, got {max_total_frames}")
        if frame_seqlen != 1:
            # A hard requirement of the no-eviction design: the eviction body at
            # attention.py:308-349 is reachable the moment shape[3] > 1, and C5 implements no
            # eviction. Refusing here converts a silent semantic divergence into a
            # construction-time error. See the module docstring's (a)/(b).
            #
            # This is the FIRST of three independent guards, deliberately not one guard in
            # three places: neutering this line alone (its earlier revision's comment called
            # it "not a hard requirement") was measured to let a frame_seqlen=2 buffer through
            # and be silently accepted by the write, because the write compared shape[3]
            # against frame_seqlen rather than against 1. The other two -- the post-allocation
            # assert below and _check_and_n_frames' `fs != 1` -- test the literal 1 and so
            # survive any relaxation of this one.
            raise ValueError(
                f"frame_seqlen={frame_seqlen} != 1. Camera streams have shape[3] == 1, which "
                f"is what makes _apply_kv_cache_eviction_causal's body unreachable "
                f"(attention.py:307). C5 implements no eviction, so shape[3] > 1 would "
                f"diverge from the GPU silently rather than loudly."
            )

        self.num_iterations = num_iterations
        self.trunk_depth = trunk_depth
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.batch_size = batch_size
        self.max_total_frames = max_total_frames
        self.frame_seqlen = frame_seqlen
        self.dtype = dtype

        #: 16 at production. Flat index = iteration * trunk_depth + block.
        self.num_slots = num_iterations * trunk_depth

        # -- Tile plan for the key axis (design.md D1a) ---------------------------------
        # Function-level import, not module-level, and that distinction is the whole point of
        # the DESK-RUNNABLE claim in the module docstring: `import
        # lingbot_map.heads.neuron_camera_cache` still pulls in nothing. plan_tiles lives in
        # C1 (neuron_kv_cache.py) rather than being copied here so there is one definition of
        # how a key axis is tiled; neuron_kv_cache imports neuron_attention, which is pure
        # torch and keeps its own nkilib import lazy inside _load_kernel.
        from lingbot_map.layers.neuron_kv_cache import plan_tiles

        #: Rows reserved ABOVE capacity for the uncommitted ``_skip_append`` frame. 1, because
        #: a non-keyframe is always a single streaming frame (gct_stream_window.py:566, :607;
        #: the multi-frame scale prefill at :526-533 is unconditionally a keyframe). With this
        #: row in hand, a non-keyframe at ``n_stored == max_total_frames`` writes at a row that
        #: physically exists and ``valid_len == cursor + pending`` stays consistent with where
        #: the write landed -- no clamping, no aliasing of a committed row.
        self.spec_headroom = 1

        #: How the frame axis splits into equal, TILE_QUANTUM-aligned tiles. At production:
        #: 1124 live + 1 speculative -> 1 tile x 1152, padded 1152. compute_attention hands
        #: this to C2's ``attend_groups``, which validates the buffer against ``plan.padded``
        #: -- so allocating the raw ``max_total_frames`` would make the graph-facing path
        #: unreachable at EVERY tile choice (1124 has no divisor that is a multiple of 128).
        self.plan = plan_tiles((max_total_frames + self.spec_headroom) * frame_seqlen)

        #: Allocated frame extent: >= max_total_frames + spec_headroom, aligned for the kernel.
        #: NOT capacity -- capacity is ``max_total_frames`` committed frames.
        self.padded_frames = self.plan.padded // frame_seqlen

        #: Compile-time-constant buffer shape, rank 5, matching the GPU dict's rank exactly so
        #: a dump diffs directly and V6's ``shape[3] == 1`` is meaningful on the port side too.
        #: Do NOT squeeze the frame_seqlen axis.
        self.slot_shape: Tuple[int, int, int, int, int] = (
            batch_size, num_heads, self.padded_frames, frame_seqlen, head_dim,
        )

        # -- THE fixed allocation ------------------------------------------------------
        def _zeros() -> Tensor:
            return torch.zeros(*self.slot_shape, dtype=dtype, device=device)

        self.k: List[Tensor] = [_zeros() for _ in range(self.num_slots)]
        self.v: List[Tensor] = [_zeros() for _ in range(self.num_slots)]

        # Guard 2 of 3 on shape[3] == 1, stated against the LITERAL 1 rather than against
        # frame_seqlen so it cannot self-adjust to whatever the constructor above permitted.
        # design.md's Data Model row for C5 is [B,H,max_total_frames,1,head_dim]; this is that
        # row, assertable.
        assert self.slot_shape[3] == 1, (
            f"slot_shape={self.slot_shape}: the frame axis (dim 3) must be exactly 1 wide. "
            f"F6's no-eviction design rests on attention.py:307's shape[3] > 1 guard being "
            f"permanently false; a wider frame axis switches GPU-side eviction on while C5 "
            f"keeps storing everything."
        )

        # -- Write cursors: DEVICE tensors, one per slot -------------------------------
        # 1-element (not 0-d) on purpose: index_copy_ accepts either, but the 1-element form
        # broadcasts to `cursor + arange(n)` for an n-frame write and keeps the source's dim-2
        # extent at n, matching k_reshaped's [B,H,n,1,D] with no squeeze.
        self._cursor: List[Tensor] = [
            torch.zeros(1, dtype=torch.long, device=device) for _ in range(self.num_slots)
        ]

        # -- Uncommitted (speculative) frame count, per slot. DEVICE tensors, for the same
        #    reason the cursor is: `valid_len` is `cursor + pending`, and if `pending` were a
        #    Python int the +1 that makes a non-keyframe's speculative frame visible would
        #    enter the graph as a trace-time literal -- two NEFF specializations of one
        #    streaming step (keyframe vs non-keyframe), reintroducing exactly the host-int
        #    coupling design.md :711-712 chose a device cursor to eliminate. So valid_len is
        #    device arithmetic over two device tensors, end to end. ---------------------
        self._pending: List[Tensor] = [
            torch.zeros(1, dtype=torch.long, device=device) for _ in range(self.num_slots)
        ]

        # -- Host-int mirrors. C1's rule (neuron_kv_cache.py:234-236): host ints are cache
        #    STRUCTURE, device tensors are graph-visible CONTENTS. This one is for diagnostics,
        #    desk checks and the capacity guard ONLY. Nothing on the compute path reads it;
        #    valid_len is derived from ``_cursor``/``_pending``, never from here. ----------
        self._n_stored: List[int] = [0] * self.num_slots
        #: Host mirror of :attr:`_pending`, for :meth:`get_cache_stats` only.
        self._n_pending: List[int] = [0] * self.num_slots

        # Per-iteration flags, mirroring the per-iteration-dict surface that
        # ``GCTStream._set_skip_append`` (gct_stream_window.py:365-367) and
        # ``_set_defer_eviction`` (:381-383) write into.
        self._skip_append: List[bool] = [False] * num_iterations
        self._defer_eviction: List[bool] = [False] * num_iterations

        self.attention = attention_adapter
        if attention_adapter is not None:
            self._check_adapter_geometry(attention_adapter)

    def _check_adapter_geometry(self, adapter: "NeuronAttentionAdapter") -> None:
        """Reject an adapter whose geometry disagrees with this cache's.

        Without this, a mis-wired or shared adapter is silently wrong rather than loud.
        ``NeuronAttentionAdapter`` folds ``scale = head_dim ** -0.5`` into q
        (``neuron_attention.py:222``, applied at ``:302``/``:351``), so handing the camera head
        (``head_dim = 2048/16 = 128``, scale 0.3536) the aggregator trunk's adapter
        (``head_dim = 64``, scale 0.125) produces finite, correctly-shaped, plausible-looking
        output that is badly wrong — measured ``rel_err`` 0.27–0.34 vs GPU SDPA over the
        *identical* visible key set. Nothing downstream can detect it: the shapes agree, so
        ``attend_groups``' own extent check passes.

        The tile plan is NOT checked here. ``visible_group`` ships ``self.plan`` *with* the
        buffer, and ``attend_groups`` validates the extent against that travelling plan, so a
        disagreeing ``padded_seqlen_k`` on the adapter is unused rather than wrong. Only the
        per-head geometry, which the adapter applies unilaterally, can silently corrupt.
        """
        a_head_dim = getattr(adapter, "head_dim", None)
        if a_head_dim is not None and int(a_head_dim) != int(self.head_dim):
            raise ValueError(
                f"attention_adapter geometry disagrees with this cache: adapter.head_dim="
                f"{int(a_head_dim)} but cache head_dim={int(self.head_dim)}. The adapter folds "
                f"scale = head_dim ** -0.5 into q, so a mismatch silently applies the wrong "
                f"softmax scale (finite, right-shaped, wrong). Build a separate "
                f"NeuronAttentionAdapter for the camera head; do not share the trunk's."
            )

    # ==============================================================================
    # The dict-key surface: what attention.py:225-256 reads and writes
    # ==============================================================================

    def slot(self, iter_idx: int, block_idx: int) -> int:
        """Flat slot index for ``(iteration, block)``. The two axes are independent.

        ``camera_head.py:358`` passes ``kv_cache=self.kv_cache[i]`` (the iteration dict) and
        ``global_idx=idx`` (the trunk block index), so a stream is addressed by both. There is
        no cross-iteration or cross-block sharing.
        """
        if not 0 <= iter_idx < self.num_iterations:
            raise IndexError(
                f"iter_idx={iter_idx} out of range for num_iterations={self.num_iterations}"
            )
        if not 0 <= block_idx < self.trunk_depth:
            raise IndexError(
                f"block_idx={block_idx} out of range for trunk_depth={self.trunk_depth}"
            )
        return iter_idx * self.trunk_depth + block_idx

    def append(
        self, iter_idx: int, block_idx: int, k: Tensor, v: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Present this frame's K/V and return what attention sees. Replaces the ``cat``.

        This is the fused store-then-read that ``attention.py:229-256`` performs: the GPU
        stores into the dict and immediately reads the result back
        (``kv_cache[k_j].clone()``, ``:247-248``) on a keyframe, or builds a local
        ``cat((cache, current))`` on a non-keyframe (``:252-253``). Returning the visible
        triple keeps those two steps atomic at the seam.

        Args:
            k: ``[B, num_heads, n_frames, frame_seqlen, head_dim]`` — exactly
               ``k_reshaped`` from ``attention.py:227-228``. ``n_frames >= 1``: the scale
               prefill writes a block of ``scale_frames`` (``gct_stream_window.py:526-533``)
               and streaming writes 1 (``:566``, ``:607``). Two call shapes, hence two NEFFs,
               which is consistent with design.md D4 — the prefill is a separate
               once-per-sequence graph.
            v: same shape as ``k``.

        Returns:
            ``(K, V, valid_len)`` from :meth:`visible_kv`, evaluated after this frame is in
            place.

        HOW ``_skip_append`` IS PRESERVED (design.md :712-713)
        -----------------------------------------------------
        The write is unconditional; only the **commit** — the cursor advance — is conditional.
        On a non-keyframe the frame lands at the cursor, ``valid_len`` counts it
        (``n_stored + n_frames``), and the cursor does not move, so the next keyframe
        overwrites that row. This reproduces ``attention.py:249-256`` with no second buffer
        and, crucially, no branch on tensor data: ``skip_append`` is a **host** bool set from
        Python by ``_set_skip_append``, known at trace time. It is loop-level control, the
        same carve-out design.md grants ``is_first_streaming_frame``.

        The two halves are independently wrong-able and both silent. Getting the visible set
        right but the store wrong (or vice versa) changes nothing about this frame's own
        attention output — the attended tensor is element-for-element identical on both
        branches, measured — and only diverges one frame later. V5(a) therefore gates the
        stored count AND the attended set, not just one of them.

        Raises:
            AssertionError: only on a KEYFRAME whose commit would exceed
                ``max_total_frames``. A non-keyframe never raises, at any depth — see the
                module docstring's CAPACITY section.
        """
        s = self.slot(iter_idx, block_idx)
        n = self._check_and_n_frames(s, k, v)
        skip = self._skip_append[iter_idx]

        # -- Capacity guard, inside the KEYFRAME branch only. Reads the host mirror
        #    deliberately: this is a fatal-error path, not a value-dependent computation, so it
        #    costs no host sync on the happy path and introduces no device->host dependency in
        #    the graph.
        #
        #    Charging capacity for a non-keyframe would be wrong, not merely strict: the GPU's
        #    non-keyframe branch (attention.py:249-256) is a LOCAL torch.cat that never mutates
        #    kv_cache[k_j], so the GPU processes non-keyframes indefinitely at any depth with
        #    no error and no depth change. Guarding before the branch made C5 abort inference
        #    in a state where the reference provably cannot fail. --------------------------
        end = self._n_stored[s] + n
        if not skip and end > self.max_total_frames:
            raise AssertionError(
                f"slot (iter={iter_idx}, block={block_idx}): camera cache exhausted at "
                f"n_stored={self._n_stored[s]} + {n} frames > capacity. Increase "
                f"max_total_frames (currently {self.max_total_frames}). NOTE: the GPU path "
                f"has no counterpart assert -- attention.py:239-240 cat()s until OOM -- so "
                f"this is a new failure mode, not design.md FM4's matched failure. Only "
                f"KEYFRAMES are charged capacity; a non-keyframe never raises."
            )

        # -- Speculative-write bound. A non-keyframe is allowed exactly `spec_headroom`
        #    uncommitted frames past capacity, which is the row reserved for it at
        #    construction, so the write below always lands inside the buffer and never aliases
        #    a committed row. n > spec_headroom on a non-keyframe cannot arise from the real
        #    driver (non-keyframes are always single frames) but is refused rather than
        #    silently clamped, because clamping would put valid_len and the write offset in
        #    disagreement. ---------------------------------------------------------------
        if skip and end > self.max_total_frames + self.spec_headroom:
            raise AssertionError(
                f"slot (iter={iter_idx}, block={block_idx}): non-keyframe of {n} frames at "
                f"n_stored={self._n_stored[s]} exceeds capacity {self.max_total_frames} by "
                f"more than the reserved speculative headroom ({self.spec_headroom} frame). "
                f"Non-keyframes are single frames on every real path "
                f"(gct_stream_window.py:566, :607); a multi-frame non-keyframe is a wiring "
                f"error, not a capacity condition."
            )

        # -- The indexed write. This IS the replacement for
        #    `torch.cat((kv_cache[k_j], k_reshaped), dim=2)` at attention.py:239-240.
        #    dim 2 is the frame axis, the same axis the cat grows. ------------------------
        cur = self._cursor[s]
        idx = cur + torch.arange(n, dtype=torch.long, device=self.device)
        self.k[s].index_copy_(2, idx, k.to(self.dtype))
        self.v[s].index_copy_(2, idx, v.to(self.dtype))

        # Any previous frame's speculative row stops being visible here, on BOTH branches:
        # cleared unconditionally rather than only on the keyframe branch, so a pending count
        # can never outlive the frame that produced it.
        self._clear_pending(s)

        if skip:
            # NON-KEYFRAME: attended but not stored. Cursor unadvanced == not committed.
            # `_pending` is a device tensor so the +n in valid_len is device data, not a
            # trace-time constant. Rebound, never mutated in place, for the same reason the
            # cursor is: the graph edge the write above consumed stays intact.
            self._pending[s] = torch.full_like(self._cursor[s], n)
            self._n_pending[s] = n
        else:
            # KEYFRAME: commit. `cur + n`, never `cur += n`: `cur` is the tensor the write
            # above indexed with, and rebinding leaves that graph edge intact.
            self._cursor[s] = cur + n
            self._n_stored[s] = end

        return self.visible_kv(iter_idx, block_idx)

    def visible_kv(self, iter_idx: int, block_idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Fixed-extent ``(K, V, valid_len)`` for the stream attention reads right now.

        ``K``/``V`` are the whole ``[B, num_heads, padded_frames, frame_seqlen, head_dim]``
        buffers — a compile-time constant, per design.md F7(a). Slicing them to the live
        extent (``K[:, :, :n]``) is banned: that is a shape which changes every frame, i.e.
        exactly the defect C1's module docstring calls bug #1.

        ``valid_len`` is a **0-d int32 device tensor** holding
        ``(cursor + pending) * frame_seqlen`` keys, computed from two DEVICE tensors — no
        Python int enters this expression, so keyframe and non-keyframe frames trace to the
        same graph instead of to two constant-folded specializations. C1's rationale applies
        verbatim (``neuron_kv_cache.py:485-488``): *"``valid_len`` must never be a Python int:
        it changes every frame (design.md F2), and as an int it would bake the frame's length
        into the graph and retrigger NEFF compilation each frame."*

        A physical prefix bound is legitimate here where it is not in C1: C5 never evicts, so
        slot order *is* frame order and there is no circular live window to resolve. That is
        the same argument C1 makes for its append-only special stream
        (``neuron_kv_cache.py:507-512``). No gather is needed and none is done.

        The prefix is also *sufficient*, not merely convenient: the camera head applies an
        all-ones mask (``torch.ones(B, 1, q_len, k_len)``, ``attention.py:279``) — no causal
        mask, no sliding-window mask, no key-axis positional bias — and RoPE is baked into
        ``k`` before the store (``attention.py:214-222``). So the key axis is
        permutation-invariant and only the *set* of included keys matters.

        Validity of ``valid_len`` is defined immediately after the :meth:`append` for this
        frame, which is where ``attention.py:247-256`` reads it. The speculative count is
        cleared at the top of EVERY :meth:`append` on this slot (either branch) and by
        :meth:`reset`, so it cannot outlive the frame that produced it — an earlier revision
        cleared it only on the keyframe branch, which left a stale speculative row attended
        after ``set_skip_append(False)`` with no following append. Prefer
        :meth:`stored_frames` if you want the committed depth instead.
        """
        s = self.slot(iter_idx, block_idx)
        # Pure DEVICE tensor arithmetic: cursor + pending, both tensors. squeeze(0) turns the
        # 1-element result into the 0-d form C2's tile_valid_lens expects; no .item(), no host
        # sync, no host int.
        valid = (
            (self._cursor[s] + self._pending[s]) * self.frame_seqlen
        ).squeeze(0).to(torch.int32)
        return self.k[s], self.v[s], valid

    def visible_mask(self, iter_idx: int, block_idx: int) -> Tensor:
        """Branchless per-frame visibility mask over the frame axis, ``[padded_frames]``.

        Fixed length always, and length ``padded_frames`` so it indexes the buffers' own frame
        axis (the tile padding is always masked out, since ``cursor + pending <=
        max_total_frames <= padded_frames``). Provided for a masking attention path that wants
        a boolean rather than a length; ``valid_len`` is the primary form because that is what
        C2's ``tile_valid_lens`` (``neuron_attention.py:85-98``) consumes. Device tensors only,
        same as :meth:`visible_kv`.
        """
        s = self.slot(iter_idx, block_idx)
        frames = torch.arange(self.padded_frames, device=self.device)
        return frames < (self._cursor[s] + self._pending[s])

    def visible_group(
        self, iter_idx: int, block_idx: int
    ) -> Tuple[Tensor, Tensor, Tensor, object]:
        """This stream as one C2 key group: ``(K_nhd, V_nhd, valid_len, plan)``.

        The tuple ``NeuronAttentionAdapter.attend_groups`` consumes (C1's
        ``visible_groups`` returns a list of exactly these). K/V are
        ``[plan.padded, num_heads, head_dim]`` NHD — the adapter validates that extent against
        ``plan.padded``, which is why the frame axis is allocated padded rather than raw
        (module docstring, ALLOCATION).

        ``attend_groups`` rather than ``attend``: the plan travels WITH the buffer, so the
        adapter cannot be constructed with a tile that disagrees with the allocation. The
        ``attend`` path requires ``k.shape[0] == adapter.padded_seqlen_k``, which couples the
        buffer to a second, independently-configured tiling — and at the raw
        ``max_total_frames`` extent no tile choice could satisfy it at all.
        """
        k_buf, v_buf, valid = self.visible_kv(iter_idx, block_idx)
        # [B,H,F,frame_seqlen,D] -> [F*frame_seqlen, H, D] NHD, the adapter's contract. This
        # mirrors attention.py:257-260's `k.reshape(a, b, c*d, e)`; at frame_seqlen == 1 the
        # key-axis length is exactly the (padded) frame count.
        n_keys = self.plan.padded
        k_nhd = k_buf[0].reshape(self.num_heads, n_keys, self.head_dim).permute(1, 0, 2)
        v_nhd = v_buf[0].reshape(self.num_heads, n_keys, self.head_dim).permute(1, 0, 2)
        return k_nhd.contiguous(), v_nhd.contiguous(), valid, self.plan

    def compute_attention(self, iter_idx: int, block_idx: int, q: Tensor) -> Tensor:
        """Attend the visible region through C2's adapter. Requires ``batch_size == 1``.

        Args:
            q: ``[B, num_heads, seqlen_q, head_dim]``, the layout ``attention.py``'s
               ``q`` already has at ``:284``.
        Returns:
            ``[B, num_heads, seqlen_q, head_dim]``.
        """
        if self.attention is None:
            raise RuntimeError(
                "compute_attention needs an attention_adapter. Either construct with "
                "attention_adapter=NeuronAttentionAdapter(...), or drive visible_kv() "
                "directly (what verify/neuron/check_c5_camera_cache.py does at the desk, with "
                "no Neuron stack)."
            )
        if self.batch_size != 1:
            raise NotImplementedError(
                f"compute_attention is implemented for batch_size == 1 (streaming), got "
                f"{self.batch_size}. Drive visible_kv() per batch element instead."
            )
        q_nhd = q[0].permute(1, 0, 2).contiguous()
        out = self.attention.attend_groups(
            q_nhd, [self.visible_group(iter_idx, block_idx)]
        )
        return out.permute(1, 0, 2).unsqueeze(0)

    # ==============================================================================
    # Flags: the per-iteration-dict surface, mirrored
    # ==============================================================================

    def get_skip_append(self, iter_idx: int = 0) -> bool:
        return self._skip_append[iter_idx]

    def set_skip_append(self, flag: bool, iter_idx: Optional[int] = None) -> None:
        """Set ``_skip_append``; all iterations by default, as the GPU setter does.

        ``GCTStream._set_skip_append`` (``gct_stream_window.py:365-367``) loops every
        iteration dict and writes the same value, so the default mirrors that. Stays a host
        bool: promoting it to a device tensor would turn loop-level control into a
        data-dependent branch, which is the opposite of the goal.

        .. note:: **The flag's LIFETIME differs from the GPU's, and the seam must absorb that**
           (adversarial parity review, 2026-08-03). On the GPU the flag lives *inside* the
           per-iteration dicts, which do not exist until the first causal forward builds them
           (``camera_head.py:285-292``), and the driver's setter is guarded by
           ``... and self.camera_head.kv_cache is not None`` (``gct_stream_window.py:365``).
           So ``_set_skip_append(True)`` **before frame 0 is a silent no-op on the GPU**, while
           this object's host bool persists from construction. Flagging frame 0 as a
           non-keyframe therefore makes C5 store 0 frames where the GPU stores all of them, and
           because depth never re-synchronises the divergence is permanent (measured: C5 depth 0
           vs GPU depth 1 at frame 0, then off-by-one in depth, ``valid_len`` and contents on
           every later frame; off-by-3 for a 3-frame prefill).

           This is a *seam* obligation, not a bug in this method — mirroring the GPU's
           accidental no-op here would be worse, since it would bake a quirk of dict-creation
           order into the port's contract. The wiring gate must assert that no ``skip_append``
           is honoured before the first ``append``, or that the driver never sets it that early.
        """
        if iter_idx is None:
            self._skip_append = [bool(flag)] * self.num_iterations
        else:
            self._skip_append[iter_idx] = bool(flag)

    def get_defer_eviction(self, iter_idx: int = 0) -> bool:
        return self._defer_eviction[iter_idx]

    def set_defer_eviction(self, flag: bool, iter_idx: Optional[int] = None) -> None:
        """Store ``_defer_eviction`` and add no reader. Faithful reproduction of Gap B.

        ``gct_stream_window.py:381-383`` writes this key into all four iteration dicts and
        **nothing** in ``camera_head.py`` or ``attention.py`` reads it — ``CausalAttention``
        contains no ``_defer_eviction`` reference at all. Same stance as
        ``SDPADictKVCacheBackend.set_defer_eviction`` (``kv_cache_backend.py:161-167``): store
        it, do not act on it, do not add a reader as part of this pass. Also note C5 must not
        participate in the deferred-eviction protocol regardless, because there is no rollback
        (Gap A, module docstring).
        """
        if iter_idx is None:
            self._defer_eviction = [bool(flag)] * self.num_iterations
        else:
            self._defer_eviction[iter_idx] = bool(flag)

    # ==============================================================================
    # Lifecycle and observability
    # ==============================================================================

    def reset(self) -> None:
        """Reset per-slot state for a new sequence. Buffers are NOT reallocated.

        The GPU's ``clean_kv_cache`` (``camera_head.py:251-254``) destroys the container
        (``del self.kv_cache; self.kv_cache = None``) so the next causal call reallocates; it
        does not zero anything either. Here the buffers persist and the cursors go to zero,
        which is observationally identical because ``valid_len`` bounds every read: stale rows
        sit outside the prefix and are unreachable.

        Deliberately does NOT touch ``camera_head.frame_idx``, which ``clean_kv_cache`` also
        zeroes. That is head state (it drives ``is_scale_frames`` and 3D-RoPE), not cache
        state; see the module docstring.

        .. warning:: **NOTHING CALLS THIS YET, and the wiring step must not forget it**
           (adversarial parity review, 2026-08-03). ``clean_kv_cache`` is not a
           start-of-sequence-only call: the windowed drivers invoke it once per *window*,
           mid-run, inside the ``while cursor < S`` loop — ``gct_stream_window.py:1094`` and
           ``:1213``, ``gct_stream_window_v2.py:1159`` and ``:1280`` ("Fresh KV cache" at the
           top of every window). Two consequences, both verified against the real head:

           1. If ``reset()`` is not called at that boundary, the GPU dict restarts at depth 0
              while this cache's cursor keeps advancing, so the visible set carries the previous
              window's keys. Measured: after a boundary the GPU attends 3 keys and C5 presents
              8, attention ``rel_err`` 0.90.
           2. Worse, simply assigning a ``NeuronCameraCache`` as ``head.kv_cache`` does not
              survive the call at all — ``clean_kv_cache`` does ``del self.kv_cache;
              self.kv_cache = None``, and the next causal forward hits ``camera_head.py:285``
              (``if self.kv_cache is None``) and silently rebuilds the list-of-dicts, resuming
              the ``torch.cat`` path with no error and no log line. The C5 instance is orphaned.

           So the seam cannot be a plain attribute assignment. It must either override
           ``clean_kv_cache`` to call ``reset()`` and re-install this object, or hold the cache
           somewhere ``clean_kv_cache`` does not destroy. The wiring gate must assert the
           instance is still installed *after* a window boundary, and that a boundary resets it.

           Note also that "observationally identical" above holds only while every read is
           bounded by ``valid_len``. Because the buffers persist, the rows above the cursor hold
           the previous sequence's *real* keys (measured: after 600 frames + ``reset()``, row
           500 still holds a key of magnitude 3.63) — realistic data, not zeros. If the padded
           tail ever becomes reachable (see ``compute_attention`` on ``prior_used_len``), the
           failure looks plausible rather than obviously broken.
        """
        for s in range(self.num_slots):
            # Rebind rather than zero_() in place: same reason as append's `cur + n`.
            self._cursor[s] = torch.zeros(1, dtype=torch.long, device=self.device)
            self._n_stored[s] = 0
            self._clear_pending(s)
        self._skip_append = [False] * self.num_iterations
        self._defer_eviction = [False] * self.num_iterations

    def stored_frames(self, iter_idx: int, block_idx: int) -> int:
        """Committed frame count for this slot (host int; diagnostics and desk checks).

        This is the port's analogue of the GPU dict's ``k_{j}.shape[2]``. It counts keyframes
        only, so it drifts from ``camera_head.frame_idx`` by the non-keyframe count — measured
        on the GPU: depth 6, ``frame_idx`` 7 after one skipped frame. That drift is real
        behaviour (``camera_head.py:374-375`` has no ``_skip_append`` guard) and must not be
        "fixed".
        """
        return self._n_stored[self.slot(iter_idx, block_idx)]

    def get_cache_stats(self, iter_idx: int = 0, block_idx: int = 0) -> dict:
        """Per-slot counters, from the host mirror. Nothing on the compute path reads this."""
        s = self.slot(iter_idx, block_idx)
        return {
            "n_stored": int(self._n_stored[s]),
            "n_pending": int(self._n_pending[s]),
            "capacity": int(self.max_total_frames),
            "free": int(self.max_total_frames - self._n_stored[s]),
            "visible_keys": int(
                (self._n_stored[s] + self._n_pending[s]) * self.frame_seqlen
            ),
            "skip_append": bool(self._skip_append[iter_idx]),
            "slot_shape": tuple(self.slot_shape),
            "padded_frames": int(self.padded_frames),
            "pad_frames": int(self.padded_frames - self.max_total_frames),
        }

    def memory_report(self) -> dict:
        """Bytes actually allocated, measured from the buffers rather than restated.

        design.md's Data Model row for C5 claims ``2 x 147 MB ~= 0.30 GB`` for K+V at
        production geometry. Computed here so the harness can check the table rather than
        trust it — an earlier draft of that table dropped the factor of 2 in three of four
        rows, and C1's harness caught a stale row exactly this way.

        design.md's figure is the LIVE extent (``max_total_frames`` frames). The buffers are
        allocated to :attr:`padded_frames` for C2's tiling, so ``*_live`` matches the table and
        the unsuffixed keys are what is really allocated; ``pad_overhead_pct`` is the delta,
        reported rather than hidden. Same shape of accounting as C1's ``tiling_report``.
        """
        elem = torch.empty((), dtype=self.dtype).element_size()
        per_slot = self.k[0].numel() * elem
        live_per_slot = (
            self.batch_size * self.num_heads * self.max_total_frames
            * self.frame_seqlen * self.head_dim * elem
        )
        return {
            "num_slots": self.num_slots,
            "slot_shape": tuple(self.slot_shape),
            "dtype": str(self.dtype),
            "bytes_per_slot_per_side": per_slot,
            "bytes_k_only": per_slot * self.num_slots,
            "bytes_k_and_v": 2 * per_slot * self.num_slots,
            "MB_k_only": round(per_slot * self.num_slots / 1e6, 2),
            "MB_k_and_v": round(2 * per_slot * self.num_slots / 1e6, 2),
            "MB_k_only_live": round(live_per_slot * self.num_slots / 1e6, 2),
            "MB_k_and_v_live": round(2 * live_per_slot * self.num_slots / 1e6, 2),
            "pad_overhead_pct": round(100.0 * (per_slot - live_per_slot) / live_per_slot, 2),
            "plan": self.plan._asdict(),
            "cat_calls_per_frame_avoided": 2 * self.num_slots,
        }

    # NOTE: there is deliberately NO rollback_last_frame here. Gap A is preserved, not fixed
    # (design.md :715-718). The hasattr guards at gct_stream_window.py:397 and
    # gct_stream_window_v2.py:466 interrogate the HEAD, not the cache, so adding one here would
    # NOT make them fire -- they are permanently false because CameraCausalHead itself lacks
    # the method, and stay false whatever the cache exposes. What holds this line is the
    # harness gate `no_rollback_method_on_C5`, which asserts `not hasattr(c5,
    # 'rollback_last_frame')`. See the module docstring's Gap A section for the full mechanism.

    # ==============================================================================
    # Internals
    # ==============================================================================

    def _clear_pending(self, s: int) -> None:
        """Zero the speculative count for slot ``s``, device tensor and host mirror both.

        Rebinds rather than ``zero_()``-ing in place, for the same reason :meth:`append` does
        ``cur + n``: an in-place mutation of a tensor already consumed by a traced op edits
        that edge's input.
        """
        self._pending[s] = torch.zeros_like(self._cursor[s])
        self._n_pending[s] = 0

    def _check_and_n_frames(self, s: int, k: Tensor, v: Tensor) -> int:
        """Validate the incoming rank-5 pair and return its frame count.

        The ``shape[3]`` check is the port-side half of V6 and guard 3 of 3: if a future
        ``num_frame_per_block``/``S`` change makes the GPU store ``shape[3] > 1`` (module
        docstring (a)), the GPU's eviction body switches on while C5 would keep storing
        everything. Raising here makes that loud.

        It tests ``fs != 1``, the LITERAL, not ``fs != self.frame_seqlen``. That is the whole
        point: a check against the config value is a config echo that relaxes in lockstep with
        the constructor, and it was measured to accept a ``frame_seqlen=2`` buffer silently once
        the constructor guard was neutered. ``frame_seqlen`` is still asserted equal to 1 so
        the two can never disagree.
        """
        if k.shape != v.shape:
            raise ValueError(f"k and v must have the same shape, got {tuple(k.shape)} vs "
                             f"{tuple(v.shape)}")
        if k.dim() != 5:
            raise ValueError(
                f"expected rank-5 [B, num_heads, n_frames, frame_seqlen, head_dim] as "
                f"produced by attention.py:227-228, got {tuple(k.shape)}"
            )
        b, h, n, fs, d = k.shape
        if (b, h, d) != (self.batch_size, self.num_heads, self.head_dim):
            raise ValueError(
                f"expected (B, num_heads, head_dim) == "
                f"({self.batch_size}, {self.num_heads}, {self.head_dim}), got ({b}, {h}, {d})"
            )
        if fs != 1 or self.frame_seqlen != 1:
            raise ValueError(
                f"shape[3]={fs} (frame_seqlen={self.frame_seqlen}): both must be exactly 1. "
                f"Camera streams keep shape[3] == 1; that guard (attention.py:307) is the "
                f"only thing making the GPU's eviction body unreachable, and C5 implements "
                f"no eviction. Compared against the literal 1, NOT against frame_seqlen, so "
                f"this survives any relaxation of the constructor check."
            )
        if n < 1:
            raise ValueError(f"n_frames must be >= 1, got {n}")
        return n

    def visible_kv_reference(
        self, iter_idx: int, block_idx: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """The GPU dict's post-reshape form. VERIFICATION ONLY.

        Returns ``(K, V, n_keys)`` with K/V as ``[B, num_heads, n_keys, head_dim]`` — what
        ``attention.py:257-260``'s ``k.reshape(a, b, c*d, e)`` hands to SDPA — so V5(a) can
        ``torch.equal`` the port against the reference directly.

        Deliberately NOT fixed-shape and NOT for the Neuron path: its extent is a function of
        the frame index, which is precisely what :meth:`visible_kv` exists to avoid. This is
        also the ONLY ``.item()`` in this module, held to the same standard as C1's single one
        at ``neuron_kv_cache.py:545``.
        """
        k_buf, v_buf, valid = self.visible_kv(iter_idx, block_idx)
        n_keys = int(valid.item())
        n_frames = n_keys // self.frame_seqlen
        flat = (self.batch_size, self.num_heads, n_keys, self.head_dim)
        return (
            k_buf[:, :, :n_frames].reshape(flat),
            v_buf[:, :, :n_frames].reshape(flat),
            valid,
        )
