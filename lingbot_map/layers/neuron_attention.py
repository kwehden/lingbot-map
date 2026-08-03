"""Neuron (Trainium2) attention seam: KV-tiled ``attention_cte`` with online-softmax merge.

spec/neuron-port/design.md components C2 (``NeuronAttentionAdapter``) and decision D1a.

WHY TILING EXISTS (design.md F1, confirmed on trn2 2026-08-03)
-------------------------------------------------------------
``nkilib.core.attention.attention_cte`` enforces

    kernel_assert(seqlen_k_active + seqlen_k_prior <= _MAX_SEQLEN)   # 36864

as a HARD assert (``attention_cte.py:356``, ``NCC_INKI016``) that fires during *tracing*.
LingBot-Map's streaming step attends a padded ``seqlen_k = 105_312`` — 2.86x over. A single
call is therefore impossible. V1 proved the way through: ``cache_softmax=True`` makes the
kernel return ``(output, out_neg_max, out_sum_recip)``, the per-query online-softmax
statistics, and merging tiles of <= _MAX_SEQLEN keys reproduces a full single call at the
bf16 noise floor (rel_err 0.00274 for 2 tiles, 0.00289 for 4).

This module holds the merge, the tile bookkeeping, and the kernel seam. The merge is
**pure torch** and deliberately importable without any Neuron stack, so the numerics can be
verified at the desk on a GPU/CPU before spending trn2 time (verify/neuron/).

THE PART THAT IS NOT IN design.md: EMPTY TILES
----------------------------------------------
design.md's C2 describes three tiles as though all three always carry keys. They do not.
The attended length is ``patches_per_frame * (n_scale + n_window) + num_special * N``, which
at production defaults is 11_000 right after the scale prefill and only crosses one tile at
frame 27 and two tiles at frame 54:

    frame   8 -> visible  11_000 -> tile valid lens [11000,     0,     0]
    frame  27 -> visible  37_125 -> tile valid lens [36864,   261,     0]
    frame  54 -> visible  74_250 -> tile valid lens [36864, 36864,   522]
    frame 1124 -> visible 105_312 -> tile valid lens [36864, 36864, 31584]

So for the first 26 frames of *every* run, two of the three tiles are entirely padding, and
for the first 53 frames one of them is. A tile whose keys are all zeros is not harmless: with
no mask the kernel softmaxes over 36_864 equal logits and returns a confident-looking
``l_i = 36864``, which then dominates the merge. That is precisely the failure V1's
calibration measured at rel_err ~0.377 against a ~0.004 noise floor.

``merge_online_softmax`` therefore neutralises a tile by its valid length, branchlessly
(``torch.where`` on a device tensor, no Python branch, no ``.item()``), so the tile count
stays a compile-time constant and one NEFF shape serves every frame.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor

# Hard kernel ceiling, verified on trn2 2026-08-03 (design.md F1). Not a tuning knob:
# raising it makes attention_cte assert during tracing.
MAX_SEQLEN = 36864

# The kernel's per-query-group size, from the V1 probe's observed stat shapes.
Q_GRP_SZ = 128

# Stand-in for -inf in the running max. A true -inf would make (acc_m - M) evaluate to
# nan when both are -inf; a large finite sentinel keeps every intermediate finite, which
# matters more on a fixed-shape accelerator than the last ulp of a value that is about to
# be multiplied by zero anyway.
NEG_BIG = -1.0e30


def num_tiles(max_seqlen_k: int, tile: int = MAX_SEQLEN) -> int:
    """Compile-time-constant tile count for a padded ``seqlen_k``."""
    if max_seqlen_k <= 0:
        raise ValueError(f"max_seqlen_k must be positive, got {max_seqlen_k}")
    return math.ceil(max_seqlen_k / tile)


def padded_seqlen_k(max_seqlen_k: int, tile: int = MAX_SEQLEN) -> int:
    """``max_seqlen_k`` rounded up so every tile is exactly ``tile`` keys.

    design.md D1a: padding the *total* to ``num_tiles * tile`` (110_592 at production
    defaults) means all tiles share one NEFF shape, compiled once and reused. Tiling
    105_312 as 36864+36864+31584 would need a second NEFF for the short final tile.
    """
    return num_tiles(max_seqlen_k, tile) * tile


def tile_valid_lens(valid_len: Tensor, n_tiles: int, tile: int = MAX_SEQLEN) -> List[Tensor]:
    """Per-tile valid key count, as device tensors.

    ``valid_len`` is a 0-d int tensor (never a Python int — a Python int here would bake
    the frame's length into the graph and retrigger NEFF compilation every frame).

    Returns ``n_tiles`` 0-d tensors, tile i holding ``clamp(valid_len - i*tile, 0, tile)``.
    Pure arithmetic: no branch, no host sync.
    """
    return [
        (valid_len - i * tile).clamp(min=0, max=tile)
        for i in range(n_tiles)
    ]


def merge_online_softmax(
    parts: Sequence[Tuple[Tensor, Tensor, Tensor, Tensor]],
    *,
    compute_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Combine per-tile ``attention_cte(cache_softmax=True)`` results into one output.

    Args:
        parts: one tuple per tile, ordered by key position:
            ``(out_i, out_neg_max_i, out_sum_recip_i, tile_valid_i)``
            - ``out_i``:            [..., seqlen_q, head_dim]
            - ``out_neg_max_i``:    [..., seqlen_q, 1]   == -(row max of logits)
            - ``out_sum_recip_i``:  [..., seqlen_q, 1]   == 1 / (row sum of exp)
            - ``tile_valid_i``:     0-d (or broadcastable) int tensor, valid keys in tile i
        compute_dtype: accumulation dtype. fp32 by default, matching the kernel's own
            ``softmax_dtype='float32'`` default — the merge is where catastrophic
            cancellation would show up, so it does not run in bf16.

    Returns:
        Tensor shaped like ``out_0``, in ``out_0``'s dtype.

    The convention is V1-verified ("A"): the kernel returns the NEGATED max and the
    RECIPROCAL sum. Convention B (max and sum directly) fails at rel_err 0.076-0.149,
    which is how V1 knew the test discriminated rather than passing vacuously.
    """
    if not parts:
        raise ValueError("merge_online_softmax needs at least one tile")

    out_dtype = parts[0][0].dtype

    acc_m: Optional[Tensor] = None   # running max      [..., seqlen_q, 1]
    acc_l: Optional[Tensor] = None   # running denom    [..., seqlen_q, 1]
    acc_o: Optional[Tensor] = None   # running output   [..., seqlen_q, head_dim]

    for out_i, neg_max_i, sum_recip_i, tile_valid_i in parts:
        o_i = out_i.to(compute_dtype)
        m_i = -neg_max_i.to(compute_dtype)
        # sum_recip is 1/l. Guard the reciprocal: a fully-masked tile can legitimately
        # produce sum_recip == 0 (l == inf) or a denormal, and we are about to discard
        # that tile anyway.
        l_i = torch.where(
            sum_recip_i.to(compute_dtype) > 0,
            1.0 / sum_recip_i.to(compute_dtype).clamp(min=torch.finfo(compute_dtype).tiny),
            torch.zeros((), dtype=compute_dtype, device=o_i.device),
        )

        # --- neutralise an empty tile -------------------------------------------------
        # An all-padding tile carries no information but, unmasked, carries a large l_i
        # (uniform softmax over 36864 zero-keys). Force it to the additive identity of
        # the merge: m = -BIG, l = 0, O = 0. Branchless, so the tile count stays a
        # compile-time constant and one NEFF serves every frame (see module docstring).
        live = (tile_valid_i > 0).to(o_i.device)
        while live.dim() < m_i.dim():
            live = live.unsqueeze(-1)
        m_i = torch.where(live, m_i, torch.full_like(m_i, NEG_BIG))
        l_i = torch.where(live, l_i, torch.zeros_like(l_i))
        o_i = torch.where(live, o_i, torch.zeros_like(o_i))

        if acc_m is None:
            acc_m, acc_l, acc_o = m_i, l_i, o_i
            continue

        # Standard online-softmax merge (design.md C2's pseudocode).
        new_m = torch.maximum(acc_m, m_i)
        a = torch.exp(acc_m - new_m)
        b = torch.exp(m_i - new_m)
        wa = acc_l * a
        wb = l_i * b
        new_l = wa + wb
        # Both tiles empty => new_l == 0. Clamp only the divisor; the numerator is 0 too,
        # so the result is 0, which is the correct output for "no keys attended".
        denom = new_l.clamp(min=torch.finfo(compute_dtype).tiny)
        acc_o = (wa * acc_o + wb * o_i) / denom
        acc_m, acc_l = new_m, new_l

    assert acc_o is not None
    return acc_o.to(out_dtype)


class NeuronAttentionAdapter:
    """The single seam between LingBot-Map and ``attention_cte`` (design.md C2).

    Owns, and is the only place that owns:
      1. scale folding      (the kernel requires ``scale == 1.0``)
      2. layout             (q seq-major, k **d-major**, v seq-major, out seq-major)
      3. KV tiling + merge  (D1a; see module docstring)
      4. the padded-tail mask on the final live tile
      5. the two environment variables that make the kernel usable at all

    ``nkilib`` is imported lazily inside :meth:`attend` so that importing ``lingbot_map``
    on a GPU or CPU host cannot pull in a Neuron stack — mirroring how ``flashinfer`` is
    already gated in ``flashinfer_cache.py:48-52``.
    """

    #: Hard kernel ceiling (design.md F1). One tile is at most this many keys.
    TILE = MAX_SEQLEN

    def __init__(
        self,
        head_dim: int,
        max_seqlen_k: int,
        *,
        tile: int = MAX_SEQLEN,
        tail_mask_mode: str = "prior_used_len",
        check_env: bool = True,
        softmax_dtype: str = "float32",
    ):
        if tile > MAX_SEQLEN:
            raise ValueError(
                f"tile={tile} exceeds attention_cte's _MAX_SEQLEN={MAX_SEQLEN}; the kernel "
                f"hard-asserts on this during tracing (design.md F1)"
            )
        self.head_dim = head_dim
        self.tile = tile
        self.n_tiles = num_tiles(max_seqlen_k, tile)
        self.padded_seqlen_k = padded_seqlen_k(max_seqlen_k, tile)
        self.max_seqlen_k = max_seqlen_k
        self.softmax_dtype = softmax_dtype

        # Folded into q instead of passed to the kernel: attention_cte's precondition is
        # scale == 1.0, and the model's own non-fused path already pre-scales q
        # (attention.py). head_dim=64 -> 0.125.
        self.scale = head_dim ** -0.5

        if tail_mask_mode not in ("prior_used_len", "additive_bias"):
            raise ValueError(
                f"tail_mask_mode must be 'prior_used_len' or 'additive_bias', "
                f"got {tail_mask_mode!r}"
            )
        # Which mechanism excludes the padded tail of the last live tile. ODQ7 is exactly
        # the question of whether the preferred one composes with tiling; both are
        # implemented so the trn2 answer is a constructor argument, not a rewrite.
        self.tail_mask_mode = tail_mask_mode

        if check_env:
            self._assert_env()

    @staticmethod
    def _assert_env() -> None:
        """Fail loudly on the two env vars that each cost a real trn2 run to learn.

        Neither is set by the DLAMI defaults, and both failures are silent-ish: without
        ``PJRT_DEVICE`` the device never appears, and without the platform override the
        compiler targets the wrong generation.
        """
        missing = [
            f"{name}={want}"
            for name, want in (("PJRT_DEVICE", "NEURON"),
                               ("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2"))
            if os.environ.get(name) != want
        ]
        if missing:
            raise RuntimeError(
                "NeuronAttentionAdapter requires these environment variables: "
                + ", ".join(missing)
                + ". They are absent from the DLAMI defaults and must be exported before "
                  "the process starts (also activate the venv with '. <venv>/bin/activate' "
                  "— invoking <venv>/bin/python directly leaves the PJRT plugin path unset)."
            )

    # ------------------------------------------------------------------------------
    # tiling bookkeeping (pure, testable without Neuron)
    # ------------------------------------------------------------------------------

    def tile_valid_lens(self, valid_len: Tensor) -> List[Tensor]:
        return tile_valid_lens(valid_len, self.n_tiles, self.tile)

    def tile_slice(self, i: int) -> slice:
        """Key-axis slice of tile ``i`` in the padded buffer."""
        return slice(i * self.tile, (i + 1) * self.tile)

    # ------------------------------------------------------------------------------
    # the kernel seam
    # ------------------------------------------------------------------------------

    def attend(self, q: Tensor, k_pad: Tensor, v_pad: Tensor, valid_len: Tensor) -> Tensor:
        """Full-visibility attention over a padded, tiled KV region.

        Args:
            q:         [seqlen_q, H, D] NHD, exactly as the GPU manager passes it
                       (``FlashInferKVCacheManager.compute_attention``'s contract).
            k_pad:     [padded_seqlen_k, H, D]
            v_pad:     [padded_seqlen_k, H, D]
            valid_len: 0-d int tensor, live keys in ``k_pad`` (device-resident, per F2's
                       per-frame drift).

        Returns:
            [seqlen_q, H, D] in ``q``'s dtype.

        No causal mask, no sliding window, no sink, no dense bias: design.md F3 established
        that the 24-block trunk applies no attention mask on any path — visibility is
        enforced structurally by which pages the cache exposes. The eviction policy *is*
        the mask.
        """
        if k_pad.shape[0] != self.padded_seqlen_k or v_pad.shape[0] != self.padded_seqlen_k:
            raise ValueError(
                f"expected k/v padded to {self.padded_seqlen_k} keys "
                f"({self.n_tiles} tiles x {self.tile}), got k={tuple(k_pad.shape)} "
                f"v={tuple(v_pad.shape)}"
            )

        kernel = self._load_kernel()
        q_scaled = (q.to(torch.float32) * self.scale).to(q.dtype)

        # NHD -> the kernel's per-head layout. bs is the HEAD axis: V1's control ran
        # q=(16,1375,64) for 16 heads, and _MAX_BS=32 bounds it.
        q_k = q_scaled.permute(1, 0, 2).contiguous()          # [H, seqlen_q, D] seq-major
        valids = self.tile_valid_lens(valid_len)

        parts = []
        for i in range(self.n_tiles):
            sl = self.tile_slice(i)
            k_t = k_pad[sl].permute(1, 2, 0).contiguous()      # [H, D, tile]  d-major!
            v_t = v_pad[sl].permute(1, 0, 2).contiguous()      # [H, tile, D]  seq-major
            out_i, neg_max_i, sum_recip_i = self._call_kernel(
                kernel, q_k, k_t, v_t, valids[i]
            )
            parts.append((out_i, neg_max_i, sum_recip_i, valids[i]))

        merged = merge_online_softmax(parts)                   # [H, seqlen_q, D]
        return merged.permute(1, 0, 2).contiguous().to(q.dtype)

    def attend_groups(self, q: Tensor, groups: Sequence[tuple]) -> Tensor:
        """Attend several independently-tiled key groups and merge them as one softmax.

        This is what C1 calls. Each group is ``(K, V, valid_len, plan)``:
            K, V:      [plan.padded, H, D]
            valid_len: 0-d int device tensor, the group's live prefix length
            plan:      anything with ``.n_tiles`` and ``.tile`` (duck-typed, so C1's
                       ``TilePlan`` needs no import here and the modules stay acyclic)

        Merging across groups is sound for the same reason merging across tiles is: the
        online-softmax combine is associative and commutative over disjoint key sets, and
        this trunk applies no key-axis mask or bias (design.md F3), so the key ordering the
        cache happens to present is immaterial. What matters is that every live key appears
        in exactly one group and no padding slot is counted — which the per-group
        ``valid_len`` enforces, tile by tile.

        The GPU's single visible sequence therefore does not have to be reconstructed as one
        contiguous buffer just to be attended; see C1's module docstring for why not doing so
        matters for shape stability.

        Args:
            q: [seqlen_q, H, D] NHD.
        Returns:
            [seqlen_q, H, D] in ``q``'s dtype.
        """
        if not groups:
            raise ValueError("attend_groups needs at least one key group")

        kernel = self._load_kernel()
        q_scaled = (q.to(torch.float32) * self.scale).to(q.dtype)
        q_k = q_scaled.permute(1, 0, 2).contiguous()          # [H, seqlen_q, D] seq-major

        parts = []
        for g, (k_pad, v_pad, valid_len, plan) in enumerate(groups):
            if k_pad.shape[0] != plan.padded or v_pad.shape[0] != plan.padded:
                raise ValueError(
                    f"group {g}: expected k/v padded to {plan.padded} keys "
                    f"({plan.n_tiles} tiles x {plan.tile}), got k={tuple(k_pad.shape)} "
                    f"v={tuple(v_pad.shape)}"
                )
            if plan.tile > MAX_SEQLEN:
                raise ValueError(
                    f"group {g}: tile={plan.tile} exceeds attention_cte's "
                    f"_MAX_SEQLEN={MAX_SEQLEN} (design.md F1)"
                )
            valids = tile_valid_lens(valid_len, plan.n_tiles, plan.tile)
            for i in range(plan.n_tiles):
                sl = slice(i * plan.tile, (i + 1) * plan.tile)
                k_t = k_pad[sl].permute(1, 2, 0).contiguous()  # [H, D, tile]  d-major!
                v_t = v_pad[sl].permute(1, 0, 2).contiguous()  # [H, tile, D]  seq-major
                out_i, neg_max_i, sum_recip_i = self._call_kernel(
                    kernel, q_k, k_t, v_t, valids[i], tile=plan.tile
                )
                parts.append((out_i, neg_max_i, sum_recip_i, valids[i]))

        merged = merge_online_softmax(parts)                  # [H, seqlen_q, D]
        return merged.permute(1, 0, 2).contiguous().to(q.dtype)

    # ------------------------------------------------------------------------------

    def _load_kernel(self):
        """Import ``attention_cte`` lazily; call it directly, do not wrap it.

        It is a ``GenericKernel`` and already NKI-decorated — wrapping it in ``nki.jit``
        fails with ``OSError: could not get source code``.
        """
        try:
            from nkilib.core.attention import attention_cte
        except ImportError as exc:  # pragma: no cover - depends on the Neuron stack
            raise RuntimeError(
                "nkilib is unavailable. NeuronAttentionAdapter.attend() requires the "
                "Neuron stack (trn2). The pure-torch parts of this module "
                "(merge_online_softmax, tile_valid_lens) are importable anywhere and are "
                "what verify/neuron/ exercises at the desk."
            ) from exc
        return attention_cte

    def _call_kernel(self, kernel, q_k: Tensor, k_t: Tensor, v_t: Tensor,
                     tile_valid: Tensor, *, tile: Optional[int] = None):
        """One tile. Returns ``(out, out_neg_max, out_sum_recip)``.

        ``cache_softmax=True`` on **every** tile — the merge needs the statistics, and the
        port always tiles even when a single call would fit.

        ``tile`` is the key extent of THIS tile, which differs per group in
        :meth:`attend_groups` (patches tile at 30_208, specials at 6_784). Defaults to the
        adapter's own tile for :meth:`attend`.
        """
        tile = self.tile if tile is None else tile
        if self.tail_mask_mode == "prior_used_len":
            # Preferred mechanism. V1 proved prior_used_len is a bit-exact valid-length
            # bound (rel_err 0.0) for a single call; ODQ7 is whether it composes with
            # tiling. Passing the whole tile as `prior` with an explicit used-length keeps
            # the active length at zero so the bound is the only thing selecting keys.
            return kernel(
                q_k, k_t, v_t,
                scale=1.0,
                causal_mask=False,
                k_prior=k_t,
                v_prior=v_t,
                prior_used_len=tile_valid,
                cache_softmax=True,
                tp_q=True, tp_k=False, tp_out=False,
                softmax_dtype=self.softmax_dtype,
            )
        # Fallback: additive -inf bias on the padded columns. Forfeits the kernel's
        # compute-skipping (REQ-079) and costs a dense [seqlen_q, tile] tensor, so it is
        # only for the case where ODQ7 comes back negative.
        bias = self._tail_bias(q_k, tile_valid, tile)
        return kernel(
            q_k, k_t, v_t,
            scale=1.0,
            causal_mask=False,
            position_bias=bias,
            cache_softmax=True,
            tp_q=True, tp_k=False, tp_out=False,
            softmax_dtype=self.softmax_dtype,
        )

    def _tail_bias(self, q_k: Tensor, tile_valid: Tensor,
                   tile: Optional[int] = None) -> Tensor:
        """Additive mask: 0 for live keys, -BIG for padding. Fixed shape, branchless."""
        tile = self.tile if tile is None else tile
        cols = torch.arange(tile, device=q_k.device)
        live = cols < tile_valid
        return torch.where(
            live,
            torch.zeros((), dtype=torch.float32, device=q_k.device),
            torch.full((), NEG_BIG, dtype=torch.float32, device=q_k.device),
        ).view(1, 1, tile)
