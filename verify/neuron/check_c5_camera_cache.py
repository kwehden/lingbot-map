"""Desk-side verification of C5 (``NeuronCameraCache``) against the real camera head.

spec/neuron-port/design.md V5(a) and V6: the port's visible KV region must match
``CameraCausalHead``'s 16 dict-grown streams element for element, across keyframe and
non-keyframe frames, with ``_skip_append`` semantics and Gap A preserved.

THE REFERENCE IS THE REAL CODE, NOT A MODEL OF IT
-------------------------------------------------
``check_ring_cache.py`` runs the genuine ``FlashInferKVCacheManager`` because *"comparing
against a hand-written model of the GPU cache would only prove I can restate my own
assumptions."* C5 has it easier: the reference is ``CausalAttention.forward``'s cache-dict
mutation at ``attention.py:229-256``, reached through the real ``CameraCausalHead`` →
``CameraBlock`` → ``CausalAttention`` chain (``block.py:347``), and that runs on CPU with
**nothing stubbed at all**. So a real ``CameraCausalHead`` is constructed and driven frame by
frame, exactly as ``gct_stream_window.py`` drives it.

Nothing in ``attention.py`` or ``camera_head.py`` is edited. The two ``torch.cat`` sites
(``attention.py:239-240`` and ``:252-253``) stay byte-identical — the reference is only a valid
oracle while they do.

TWO INDEPENDENT ORACLES, NEITHER OF THEM CIRCULAR
-------------------------------------------------
1. **The attended set** (``camera_attended_kv_matches_gpu``). The tensors the model actually
   attends are *locals* inside ``CausalAttention.forward`` (``k`` after the reshape at
   ``:257-260``), so they are captured by shimming the ``F.scaled_dot_product_attention``
   the module calls at ``:284``: the shim records ``(q, k, v)`` and delegates to the real
   function. That is instrumentation of the reference, not a reimplementation of it, and it
   observes both branches — the keyframe's post-store cache clone (``:247-248``) and the
   non-keyframe's local ``cat`` (``:252-253``) — with no restatement of the view logic.

   C5's *input* is derived from that same recording (the tail beyond the previously-cached
   frame count is exactly this frame's ``k_reshaped``), which is why oracle 1 alone would be
   partly self-referential for the current frame. What it is NOT self-referential about is the
   **prefix**: rows ``[0, n_cached)`` came from earlier frames' writes, so a store bug shows up
   here one frame after it is committed. That delayed signature is the whole point of the
   ``store_on_non_keyframe`` injection below.

2. **The stored contents** (``camera_cache_contents_match_gpu``). Independently of oracle 1,
   the reference dict's ``k_{j}`` / ``v_{j}`` tensors are compared against C5's committed
   buffer prefix, per slot, per frame. This is what catches a divergence in what was *stored*
   rather than what was *presented*, and the two can be wrong separately: measured on a shared
   ``(cache, probe)`` pair, the keyframe and non-keyframe paths present element-for-identical
   tensors and differ only in the mutation, so a suite gating one but not the other is silent
   on half the bug space.

Both are ``torch.equal`` with no tolerance: the write is a copy, not arithmetic. ``rel_err``
with a ``1e-4`` gate is reserved for attention numerics.

WHAT V6 CONTRIBUTES
-------------------
design.md's F6 rests the whole no-eviction design on ``shape[3] == 1``. Asserting that alone is
vacuous on an unwritten cache, so ``_apply_kv_cache_eviction_causal`` (``attention.py:297``) is
wrapped to prove positively that the guard at ``:307`` was *evaluated* (call count > 0) and its
body never *ran* (no ``_special`` key ever materialised, no stream's ``shape[2]`` ever
decreased). The recomputation line at ``:236`` is also grepped, because that is what makes
``shape[3] == 1`` self-perpetuating — design.md attributes it to ``frame_seqlen`` instead, which
is not read in the cached branch at all.

**Both halves, GPU and PORT.** The wrapper and the reference-dict scan observe only untouched
upstream code, so they report ``shape[3] == 1`` regardless of what C5 does: measured, deleting
BOTH of C5's own ``shape[3]`` guards left the suite at PASS. Three further gates
(``f6_port_side_frame_axis_is_exactly_1_wide``, ``f6_ctor_refuses_frame_seqlen_not_1``,
``f6_write_refuses_shape3_gt_1_independently_of_config``) interrogate the live C5 instance, its
constructor and its write, and ``--inject_bug accept_shape3_gt_1`` proves they discriminate.

WHAT THE COMPARISON ORACLES CANNOT SEE, AND WHAT COVERS IT
----------------------------------------------------------
Both oracles drive :meth:`NeuronCameraCache.append` / :meth:`visible_kv`. They are silent on
three things by construction, each of which was a real defect found only after a gate was added
for it:

  1. **The graph-facing path.** ``compute_attention`` is the only method that reaches the NEFF,
     and the suite had zero occurrences of it — so it could be dead code (it was: it handed the
     adapter a ``max_total_frames``-key buffer, which no TILE_QUANTUM-aligned tile can accept)
     while all checks passed. Covered by
     ``compute_attention_feeds_the_adapter_a_tile_padded_buffer`` (adapter probe, no Neuron
     stack) and ``real_adapter_shape_contract_is_satisfiable_at_prod_capacity``.
  2. **Capacity, on the non-keyframe branch.** The replay never reaches capacity and the
     capacity loop only ever appends keyframes. Covered by the paired
     ``non_keyframe_at_capacity_does_not_raise`` / ``keyframe_at_capacity_still_raises``.
  3. **Source-level static discipline.** Those gates read text, so ``--inject_bug`` cannot reach
     them; ``static_discipline_gates_have_discriminating_power`` re-evaluates their predicates
     against synthetic mutants of C5's own source instead.

Run:
    docker exec -w /home/kwehden/lingbot-tier1/lingbot-map lingbot-tier1 \
        python verify/neuron/check_c5_camera_cache.py
    docker exec -w /home/kwehden/lingbot-tier1/lingbot-map lingbot-tier1 \
        python verify/neuron/check_c5_camera_cache.py --inject_bug store_on_non_keyframe

Or all seven injections plus the baseline:
    bash verify/neuron/run_c5_camera_cache_checks.sh
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)

from lingbot_map.heads.camera_head import CameraCausalHead  # noqa: E402
from lingbot_map.heads.neuron_camera_cache import NeuronCameraCache  # noqa: E402
from lingbot_map.layers import attention as attn_mod  # noqa: E402
from lingbot_map.layers.attention import CausalAttention  # noqa: E402

RESULTS: list[dict] = []


def emit(name: str, ok: bool, **kw) -> None:
    RESULTS.append({"check": name, "pass": bool(ok), **kw})
    extra = "  ".join(f"{k}={v}" for k, v in kw.items())
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {extra}", flush=True)


def _executable_source(path: str) -> str:
    """Source with comments and string literals (hence docstrings) removed.

    The static-discipline check below greps for ``.item()`` and ``torch.cat``, and
    ``neuron_camera_cache.py``'s docstrings *discuss* both at length. Scanning raw text would
    fire on the prose and say nothing about the code.
    """
    import io
    import tokenize

    with open(path, "rb") as fh:
        raw = fh.read()
    lines = raw.decode().splitlines()
    # Blank out every comment/string token in place, so line structure and the spacing inside
    # real statements both survive.
    for tok in tokenize.tokenize(io.BytesIO(raw).readline):
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        (r0, c0), (r1, c1) = tok.start, tok.end
        for r in range(r0, r1 + 1):
            ln = lines[r - 1]
            a = c0 if r == r0 else 0
            b = c1 if r == r1 else len(ln)
            lines[r - 1] = ln[:a] + " " * (b - a) + ln[b:]
    return "\n".join(lines)


def _class_node(path: str, cls: str):
    import ast

    with open(path) as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return node
    return None


def _method_span(path: str, cls: str, meth: str):
    """1-indexed ``(first_line, last_line)`` of ``cls.meth``, via ast rather than str.split.

    The point of using ast is that the answer does not depend on where in the class the method
    happens to be declared, which is what made the earlier positional gate hollow.
    """
    node = _class_node(path, cls)
    if node is None:
        return None
    for item in node.body:
        if getattr(item, "name", None) == meth:
            return (item.lineno, item.end_lineno)
    return None


def _host_int_attr_reads(path: str, cls: str, methods: tuple, attrs: tuple) -> list:
    """``['visible_kv:_n_pending', ...]`` for every ``self.<attr>`` read in ``methods``.

    Semantic complement to the ``.item()`` count: a host int reaching valid_len is a graph
    constant even though it contains no ``.item()`` at all.
    """
    import ast

    node = _class_node(path, cls)
    if node is None:
        return [f"{cls}: class not found"]
    found = []
    for item in node.body:
        if getattr(item, "name", None) not in methods:
            continue
        for sub in ast.walk(item):
            if (
                isinstance(sub, ast.Attribute)
                and sub.attr in attrs
                and isinstance(sub.value, ast.Name)
                and sub.value.id == "self"
            ):
                found.append(f"{item.name}:{sub.attr}@L{sub.lineno}")
    return found


def rel_err(a, b) -> float:
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-12)).item()


# ====================================================================================
# Instrumentation of the reference. Records, never reimplements.
# ====================================================================================
class SDPARecorder:
    """Shim ``attention.py``'s ``F`` so the attended k/v at ``:284-290`` are observable.

    Delegates every other attribute to the real ``torch.nn.functional``, so nothing else in
    ``attention.py`` changes behaviour. Records in call order, which for the camera head is
    ``for i in range(num_iterations): for idx in range(trunk_depth)``
    (``camera_head.py:344-358``) — so entry ``i * trunk_depth + idx`` is slot ``(i, idx)``.
    That mapping is cross-validated by ``sdpa_call_count_matches_slot_count``.
    """

    def __init__(self):
        self._real = attn_mod.F
        self.calls: list[tuple] = []
        self.enabled = False

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)

    def scaled_dot_product_attention(self, q, k, v, **kw):
        if self.enabled:
            self.calls.append((q.detach().clone(), k.detach().clone(), v.detach().clone()))
        return self._real.scaled_dot_product_attention(q, k, v, **kw)

    def __enter__(self):
        attn_mod.F = self
        return self

    def __exit__(self, *exc):
        attn_mod.F = self._real
        return False


class EvictionWatch:
    """Wrap ``_apply_kv_cache_eviction_causal`` to prove the F6 guard is reached, not assumed.

    Records, per call, the ``shape[3]`` the guard at ``attention.py:307`` tests and whether the
    stream's ``shape[2]`` decreased across the call (the body's only observable effect on the
    main stream). Calls through to the real method, so the reference's behaviour is unchanged.
    """

    def __init__(self):
        self.real = CausalAttention._apply_kv_cache_eviction_causal
        self.n_calls = 0
        self.shape3_seen: set[int] = set()
        self.n_shape2_decreases = 0

    def __enter__(self):
        watch = self

        def wrapped(self_, kv_cache, global_idx, camera_token_idx, scale_token_idx):
            t = kv_cache[f"k_{global_idx}"]
            watch.n_calls += 1
            watch.shape3_seen.add(int(t.shape[3]))
            before = int(t.shape[2])
            out = watch.real(self_, kv_cache, global_idx, camera_token_idx, scale_token_idx)
            if int(kv_cache[f"k_{global_idx}"].shape[2]) < before:
                watch.n_shape2_decreases += 1
            return out

        CausalAttention._apply_kv_cache_eviction_causal = wrapped
        return self

    def __exit__(self, *exc):
        CausalAttention._apply_kv_cache_eviction_causal = self.real
        return False


# ====================================================================================
# Discriminating power: quiet bugs that must turn the suite FAIL.
# ====================================================================================
def inject_bug(kind: str, cfg: dict) -> dict:
    """Return constructor/behaviour overrides implementing a subtly wrong C5.

    Follows ``check_c4_wiring.py:114-150``'s convention. Every bug here is *quiet*: none
    raises, none changes a buffer shape, all leave ``valid_len`` self-consistent and every
    structural check passing. A compare pass that reports PASS is worthless unless the same
    comparison reports FAIL on each of these.

    MEASURED at ``--n_frames 40``, ``MAX_TOTAL 96`` (see ``camera_cache_results.json`` per run;
    every one is caught by a *gate*, none by a traceback — see the note on ``MAX_TOTAL``):

      * ``store_on_non_keyframe``  -> FAIL 22/25. The highest-value discriminator: it is
        precisely Gap A's shape on GPU (the speculative frame stays resident) and is the
        mistake a correct-looking indexed write invites. The **frame of injection is
        unaffected**: the first non-keyframe is frame 5, and the earliest gate to notice is
        ``skip_append_semantics_match``'s depth half at frame 5 (stored 8 vs reference 7),
        while the *attended* set does not diverge until frame **6** (9 keys vs 8) — the
        delayed signature this suite exists to catch. Also trips
        ``frame_idx_diverges_from_depth_by_non_keyframe_count`` (drift 0, want 5), which is
        oracle checklist item (4) doing real work.
      * ``cursor_double_advance``  -> FAIL 22/25 (attended shape 6 vs 3 at frame 0, contents,
        and valid_len). Advances by n before the write as well as after, leaving a zero row
        inside the live prefix.
      * ``bf16_storage``           -> FAIL 23/25. Only a bit-exactness gate on *contents*
        catches it — 12 attended keys differ, plus the stored-contents oracle — and every
        structural, cursor and valid_len check still passes, so this is the one that proves the
        ``torch.equal`` gates are load-bearing. Worth injecting *because* C5's fp32
        deliberately differs from C1's bf16 (design.md :706, :741), so it is the divergence a
        C1-shaped copy-paste introduces.
      * ``valid_len_is_capacity``  -> FAIL 22/25. Attends ``max_total_frames - n`` zero rows
        and still produces plausible finite output; the C5 analogue of the empty-tile hazard.
      * ``accept_shape3_gt_1``     -> must FAIL the two F6 port-side gates. Stubs out BOTH of
        C5's ``shape[3] == 1`` guards, which is the whole foundation of the no-eviction design.
        Offered because the suite was measured to report **PASS 25/25** with both guards
        deleted: every shape[3] gate read only the GPU reference dict and the EvictionWatch,
        which are 1 regardless of what C5 does, and the constructor guard was never exercised
        at all (``grep frame_seqlen=`` over the harness returned 0 hits). Unlike the four above
        this one is not quiet in the replay -- the replay never feeds shape[3] > 1 -- so it is
        the port-side gates, not the comparison oracles, that must catch it.
      * ``alloc_raw_max_total_frames`` -> must FAIL the two graph-facing gates. Allocates the
        frame axis to ``max_total_frames`` rather than ``plan.padded``, i.e. C5's original
        allocation, under which ``compute_attention`` could never succeed at production geometry
        (no TILE_QUANTUM-aligned tile makes ``padded_seqlen_k == 1124``). Perfectly quiet in the
        replay and invisible to both comparison oracles, which is why the suite -- with zero
        occurrences of ``compute_attention`` -- reported PASS while the only method that feeds
        the NEFF was dead code.
      * ``charge_capacity_on_non_keyframe`` -> must FAIL
        ``non_keyframe_at_capacity_does_not_raise``. The pre-fix guard placement: capacity
        charged before the ``_skip_append`` branch, so C5 aborts on a non-keyframe at
        ``n_stored == max_total_frames`` while the GPU attends one extra key, keeps its dict
        depth, and raises nothing at all (``attention.py:249-256``'s cat is purely local). Quiet
        in the replay because the replay never reaches capacity.

    NOT offered: a key-ORDER bug. Presenting ``current ++ cached`` is numerically invisible
    here (all-ones mask at ``attention.py:279``, RoPE pre-applied at ``:214-222``), so it would
    report PASS on any output gate while being genuinely harmless. Offering it as a
    discriminator would teach the wrong lesson about this suite's power.
    """
    if kind is None:
        return {}

    class _Bugged(NeuronCameraCache):
        pass

    if kind == "store_on_non_keyframe":
        def append(self, i, j, k, v):
            saved = list(self._skip_append)
            self._skip_append = [False] * self.num_iterations   # store it anyway
            try:
                return NeuronCameraCache.append(self, i, j, k, v)
            finally:
                self._skip_append = saved
        _Bugged.append = append
    elif kind == "cursor_double_advance":
        def append(self, i, j, k, v):
            s = self.slot(i, j)
            n = k.shape[2]
            if not self._skip_append[i]:
                self._cursor[s] = self._cursor[s] + n     # advance BEFORE the write too
            return NeuronCameraCache.append(self, i, j, k, v)
        _Bugged.append = append
    elif kind == "bf16_storage":
        cfg = dict(cfg, dtype=torch.bfloat16)
    elif kind == "valid_len_is_capacity":
        def visible_kv(self, i, j):
            s = self.slot(i, j)
            cap = torch.full((), self.max_total_frames * self.frame_seqlen,
                             dtype=torch.int32, device=self.device)
            return self.k[s], self.v[s], cap
        _Bugged.visible_kv = visible_kv
    elif kind == "alloc_raw_max_total_frames":
        # Allocates the frame axis to max_total_frames instead of plan.padded -- the original
        # C5 allocation. Quiet in the replay (it is a legal buffer, just not a tileable one) and
        # invisible to every comparison oracle; only the compute_attention gate sees it, which
        # is why that gate had to exist. Patches the base class for the same reason
        # accept_shape3_gt_1 does: the graph-facing gates build NeuronCameraCache directly.
        _raw_init = NeuronCameraCache.__init__

        def _raw(self, *a, **kw):
            _raw_init(self, *a, **kw)
            self.spec_headroom = 0
            self.padded_frames = self.max_total_frames
            self.plan = self.plan._replace(
                n_keys=self.max_total_frames * self.frame_seqlen,
                n_tiles=1, tile=self.max_total_frames * self.frame_seqlen,
                padded=self.max_total_frames * self.frame_seqlen,
            )
            self.slot_shape = (self.batch_size, self.num_heads, self.padded_frames,
                               self.frame_seqlen, self.head_dim)
            self.k = [torch.zeros(*self.slot_shape, dtype=self.dtype, device=self.device)
                      for _ in range(self.num_slots)]
            self.v = [torch.zeros(*self.slot_shape, dtype=self.dtype, device=self.device)
                      for _ in range(self.num_slots)]
        NeuronCameraCache.__init__ = _raw
    elif kind == "charge_capacity_on_non_keyframe":
        # The pre-fix guard placement: capacity checked before the skip_append branch, so a
        # non-keyframe at n_stored == max_total_frames raises where the GPU cannot fail.
        _cap_append = NeuronCameraCache.append

        def _charged(self, i, j, k, v):
            s = self.slot(i, j)
            if self._n_stored[s] + int(k.shape[2]) > self.max_total_frames:
                raise AssertionError(
                    f"camera cache exhausted (max_total_frames={self.max_total_frames})")
            return _cap_append(self, i, j, k, v)
        NeuronCameraCache.append = _charged
    elif kind == "accept_shape3_gt_1":
        # Patches the BASE class, not _Bugged: the F6 port-side gates construct
        # NeuronCameraCache directly (they must, to test the constructor), so a subclass-only
        # injection would leave them measuring the unmodified class -- which is precisely the
        # vacuity being fixed here.
        def _permissive_check(self, s, k, v):
            return int(k.shape[2])          # guard 3 of 3 gone
        NeuronCameraCache._check_and_n_frames = _permissive_check

        _real_init = NeuronCameraCache.__init__

        def _permissive_init(self, *a, **kw):
            # Constructs at frame_seqlen=1, then widens the frame axis: exactly the state a
            # relaxed constructor guard (guards 1 and 2 gone) would produce.
            fs = kw.pop("frame_seqlen", 1)
            _real_init(self, *a, **kw)
            if fs != 1:
                self.frame_seqlen = fs
                self.slot_shape = (self.batch_size, self.num_heads, self.padded_frames,
                                   fs, self.head_dim)
                self.k = [torch.zeros(*self.slot_shape, dtype=self.dtype,
                                      device=self.device) for _ in range(self.num_slots)]
                self.v = [torch.zeros(*self.slot_shape, dtype=self.dtype,
                                      device=self.device) for _ in range(self.num_slots)]
        NeuronCameraCache.__init__ = _permissive_init
    else:
        raise ValueError(f"unknown --inject_bug kind: {kind}")

    print(f"*** INJECTED BUG: {kind} -- this pass MUST report FAIL ***", flush=True)
    return {"cls": _Bugged, "cfg": cfg}


# ====================================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject_bug", default=None)
    ap.add_argument("--n_frames", type=int, default=40)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emit("device", True, device=str(dev),
         gpu=(torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"),
         injected_bug=args.inject_bug or "none",
         expected_verdict="FAIL" if args.inject_bug else "PASS")

    # Scaled-down geometry with production STRUCTURE: 2 iterations x 2 blocks = 4 slots, a
    # multi-frame scale prefill, then single-frame streaming. Small enough to fill the buffer
    # well past the prefill inside n_frames.
    NUM_ITER, DEPTH, HEADS, HDIM = 2, 2, 4, 8
    DIM_IN = HEADS * HDIM
    SCALE_FRAMES = 3
    # Capacity must leave room for the INJECTED bugs to stay quiet, not just for the correct
    # replay. A correct run stores 37 frames at n_frames=40; `store_on_non_keyframe` stores 42
    # and `cursor_double_advance` writes as far as row ~74. At MAX_TOTAL=40 both hit the
    # capacity guard / an index_copy_ bounds assert and DIE inside the replay loop, so the
    # comparison gates never emit and the suite's exit 1 comes from a traceback rather than
    # from a check catching the bug. That is not discriminating power, it is a crash — and it
    # is what this constant being 40 originally produced. Sized to 96 so every injection is
    # observed by a gate. The capacity guard is exercised by its own small instances below.
    # NOTE the buffer's frame axis is plan_tiles(96+1).padded == 128, not 96: the frame axis is
    # tile-padded (TILE_QUANTUM=128) so C2's seam can accept it, with one row reserved above
    # capacity for the uncommitted speculative frame. Capacity is still MAX_TOTAL commits.
    MAX_TOTAL = 96
    dtype = torch.float32

    head = CameraCausalHead(
        dim_in=DIM_IN, trunk_depth=DEPTH, num_heads=HEADS, num_iterations=NUM_ITER,
        kv_cache_sliding_window=4, kv_cache_scale_frames=2,
    ).to(dev).eval()

    # V6, half 1: evaluated on the real INSTANCE, before the forward (design.md's wording).
    rollback_before = hasattr(head, "rollback_last_frame")

    bug = inject_bug(args.inject_bug, dict(
        num_iterations=NUM_ITER, trunk_depth=DEPTH, num_heads=HEADS, head_dim=HDIM,
        device=dev, batch_size=1, max_total_frames=MAX_TOTAL, dtype=dtype,
    ))
    C5 = bug.get("cls", NeuronCameraCache)
    c5 = C5(**bug.get("cfg", dict(
        num_iterations=NUM_ITER, trunk_depth=DEPTH, num_heads=HEADS, head_dim=HDIM,
        device=dev, batch_size=1, max_total_frames=MAX_TOTAL, dtype=dtype,
    )))

    emit("no_rollback_method_on_C5", not hasattr(c5, "rollback_last_frame"),
         detail="Gap A preserved, not fixed (design.md :715-718). THIS GATE is what holds that "
                "line: the hasattr guards at gct_stream_window.py:397 and v2:466 interrogate "
                "the HEAD, not the cache, so they are permanently false because "
                "CameraCausalHead lacks the method and stay false whatever the cache exposes "
                "(verified: a subclass with a rollback assigned as head.kv_cache does not make "
                "the guard fire). A rollback on C5 would only be REACHED if the wiring "
                "follow-on also forwarded one from CameraCausalHead")

    # ---- 1. lockstep replay in the real driver order -------------------------------
    # Deterministic mixed keyframe pattern so BOTH _skip_append branches fire, per
    # check_ring_cache.py:171's convention.
    N = args.n_frames
    KEEP = [not (i % 7 == 5) for i in range(N)]

    attended_mismatch = None
    contents_mismatch = None
    valid_mismatch = None
    depth_mismatch = None
    shape3_bad = None
    special_keys_seen: set[str] = set()

    shape_set: set[tuple] = set()
    cursor_kinds: set[tuple] = set()
    validlen_kinds: set[tuple] = set()
    sdpa_counts: set[int] = set()

    n_keyframe = n_nonkeyframe = 0
    max_depth = 0
    nonzero_seen = False
    frames_all_identical = True
    prev_attended = None
    n_compared_pairs = 0
    n_nonempty_pairs = 0

    recorder = SDPARecorder()
    with EvictionWatch() as ev, recorder:
        for f in range(N):
            n_frames = SCALE_FRAMES if f == 0 else 1
            skip = (not KEEP[f]) and f > 0     # the scale prefill is always a keyframe

            # _set_skip_append's two writes (gct_stream_window.py:365-367) and its C5 mirror.
            if head.kv_cache is not None:
                for d in head.kv_cache:
                    d["_skip_append"] = skip
            c5.set_skip_append(skip)

            # Cached frame counts BEFORE this forward, read off the reference dict itself.
            pre_n = {}
            for i in range(NUM_ITER):
                for j in range(DEPTH):
                    t = head.kv_cache[i][f"k_{j}"] if head.kv_cache is not None else None
                    pre_n[(i, j)] = 0 if t is None else int(t.shape[2])

            tokens = torch.randn(1, n_frames, 5, DIM_IN, dtype=dtype, device=dev)
            recorder.calls.clear()
            recorder.enabled = True
            with torch.no_grad():
                head([tokens], causal_inference=True, num_frame_per_block=n_frames)
            recorder.enabled = False
            sdpa_counts.add(len(recorder.calls))

            # The first frame's flag write happens after kv_cache exists, so re-apply for the
            # prefill frame's bookkeeping only (it is a keyframe either way).
            for i in range(NUM_ITER):
                for j in range(DEPTH):
                    call = recorder.calls[i * DEPTH + j]
                    a_k, a_v = call[1], call[2]          # [B, H, n_keys, D] attended
                    B, H, n_keys, D = a_k.shape
                    fs = 1
                    # This frame's k_reshaped, taken from the reference's own attended tensor:
                    # the tail beyond the previously-cached frames IS the current frame, on
                    # both branches (attention.py:239-240 vs :252-253).
                    cur_k = a_k[:, :, pre_n[(i, j)] * fs:].reshape(B, H, -1, fs, D)
                    cur_v = a_v[:, :, pre_n[(i, j)] * fs:].reshape(B, H, -1, fs, D)
                    if cur_k.shape[2] != n_frames and attended_mismatch is None:
                        attended_mismatch = (
                            f"frame {f} slot ({i},{j}): recovered {cur_k.shape[2]} current "
                            f"frames, expected {n_frames}")

                    ck, cv, cvalid = c5.append(i, j, cur_k, cur_v)
                    shape_set.add((tuple(ck.shape), tuple(cv.shape)))
                    cursor_kinds.add((
                        bool(torch.is_tensor(c5._cursor[c5.slot(i, j)])),
                        str(c5._cursor[c5.slot(i, j)].dtype),
                    ))
                    validlen_kinds.add((bool(torch.is_tensor(cvalid)), str(cvalid.dtype),
                                        int(cvalid.dim())))

                    # -- ORACLE 1: attended set, post-reshape, torch.equal ---------------
                    pk, pv, pvalid = c5.visible_kv_reference(i, j)
                    n_compared_pairs += 1
                    if pk.numel() > 0 and a_k.numel() > 0:
                        n_nonempty_pairs += 1
                    if attended_mismatch is None:
                        if pk.shape != a_k.shape:
                            attended_mismatch = (
                                f"frame {f} slot ({i},{j}): shape {tuple(a_k.shape)} vs "
                                f"{tuple(pk.shape)}")
                        elif not torch.equal(pk, a_k) or not torch.equal(pv, a_v):
                            nbad = int((pk != a_k).any(dim=-1).sum())
                            attended_mismatch = (
                                f"frame {f} slot ({i},{j}): {nbad} attended keys differ")
                    if valid_mismatch is None and int(pvalid) != n_keys:
                        valid_mismatch = (f"frame {f} slot ({i},{j}): valid_len "
                                          f"{int(pvalid)} != attended {n_keys}")

                    # -- ORACLE 2: stored contents, independent of oracle 1 --------------
                    ref = head.kv_cache[i][f"k_{j}"]
                    ref_v = head.kv_cache[i][f"v_{j}"]
                    n_ref = int(ref.shape[2])
                    max_depth = max(max_depth, n_ref)
                    if shape3_bad is None and (ref.shape[3] != 1 or ref_v.shape[3] != 1):
                        shape3_bad = (f"frame {f} slot ({i},{j}): shape[3] "
                                      f"k={ref.shape[3]} v={ref_v.shape[3]}")
                    special_keys_seen.update(
                        key for key in head.kv_cache[i] if key.endswith("_special"))

                    n_stored = c5.stored_frames(i, j)
                    if depth_mismatch is None and n_stored != n_ref:
                        depth_mismatch = (f"frame {f} slot ({i},{j}): stored {n_stored} "
                                          f"!= reference depth {n_ref}")
                    if contents_mismatch is None and n_stored == n_ref:
                        got_k = c5.k[c5.slot(i, j)][:, :, :n_stored]
                        got_v = c5.v[c5.slot(i, j)][:, :, :n_stored]
                        if not torch.equal(got_k, ref) or not torch.equal(got_v, ref_v):
                            contents_mismatch = (
                                f"frame {f} slot ({i},{j}): stored contents differ over "
                                f"{n_stored} frames")

                    if a_k.abs().max() > 0:
                        nonzero_seen = True
                    if (i, j) == (0, 0):
                        if prev_attended is not None and not (
                            prev_attended.shape == a_k.shape
                            and torch.equal(prev_attended, a_k)
                        ):
                            frames_all_identical = False
                        prev_attended = a_k

            if skip:
                n_nonkeyframe += 1
            else:
                n_keyframe += 1

    emit("camera_attended_kv_matches_gpu", attended_mismatch is None,
         frames=N, slots=NUM_ITER * DEPTH, keyframes=n_keyframe,
         non_keyframes=n_nonkeyframe,
         detail=attended_mismatch or "element-for-element identical, K and V, every slot, "
                                     "every frame, both _skip_append branches (torch.equal)")
    emit("camera_cache_contents_match_gpu", contents_mismatch is None,
         detail=contents_mismatch or "C5's committed buffer prefix == the reference dict's "
                                     "k_{j}/v_{j}, independent of the attended oracle")
    emit("skip_append_semantics_match", depth_mismatch is None and valid_mismatch is None,
         depth=depth_mismatch or "stored depth +1 on keyframe, +0 on non-keyframe",
         valid=valid_mismatch or "valid_len == n_stored + n_frames on both branches",
         detail="both halves gated: asserting only the depth misses the bug that drops the "
                "current frame from its own attention")

    # ---- counter divergence: frame_idx is NOT guarded by _skip_append ---------------
    depth0 = c5.stored_frames(0, 0)
    emit("frame_idx_diverges_from_depth_by_non_keyframe_count",
         head.frame_idx - depth0 == n_nonkeyframe,
         frame_idx=head.frame_idx, cache_depth=depth0, non_keyframes=n_nonkeyframe,
         drift=head.frame_idx - depth0,
         detail="camera_head.py:374-375 increments frame_idx UNCONDITIONALLY, unlike the "
                "aggregator's guarded total_frames_processed (stream.py:535). A "
                "reimplementation that 'fixes' this drift passes every other gate")

    # ---- VACUITY GUARDS (design.md :1075-1081). Their own emitted checks. -----------
    emit("sequence_contains_both_outcomes", n_keyframe > 0 and n_nonkeyframe > 0,
         n_keyframes=n_keyframe, n_non_keyframes=n_nonkeyframe,
         detail="if either is 0 the identity checks above are vacuous")
    emit("buffer_filled_past_the_scale_prefill", max_depth > SCALE_FRAMES + 1,
         max_depth=max_depth, scale_prefill=SCALE_FRAMES,
         detail="if this is <= the prefill, only the first-write path was exercised")
    emit("both_sides_nonempty", n_nonempty_pairs == n_compared_pairs > 0,
         compared=n_compared_pairs, nonempty=n_nonempty_pairs,
         detail="if any compared pair was empty, torch.equal on it is trivially True")
    emit("keys_not_identical_by_construction", nonzero_seen and not frames_all_identical,
         nonzero=nonzero_seen, varies_across_frames=not frames_all_identical,
         detail="random per-frame input: neither all-zeros nor constant, so equality is "
                "informative")
    emit("multiple_iterations_and_blocks_exercised",
         all(c5.stored_frames(i, j) > 0 for i in range(NUM_ITER) for j in range(DEPTH)),
         slots=NUM_ITER * DEPTH,
         detail="all iterations x blocks written; a per-slot indexing bug is invisible with "
                "one slot")
    emit("sdpa_call_count_matches_slot_count", sdpa_counts == {NUM_ITER * DEPTH},
         counts=sorted(sdpa_counts), expected=NUM_ITER * DEPTH,
         detail="cross-validates the recorder's call-order -> (iteration, block) mapping")

    # ---- static-shape / NEFF evidence, accumulated over EVERY frame -----------------
    emit("fixed_shapes_across_all_frames", len(shape_set) == 1,
         n_distinct_buffer_shapes=len(shape_set), shape=str(sorted(shape_set)[0][0]),
         detail="one shape for every frame -> one NEFF")
    emit("cursor_is_device_tensor_not_int",
         len(cursor_kinds) == 1 and sorted(cursor_kinds)[0][0]
         and sorted(cursor_kinds)[0][1] in ("torch.int32", "torch.int64"),
         kinds=sorted(cursor_kinds),
         detail="design.md :711-712 mandates a device-tensor cursor for C5 specifically; "
                "C1's host int is only tolerable at its 74-slot range")
    emit("valid_len_is_device_tensor_not_int",
         len(validlen_kinds) == 1 and sorted(validlen_kinds)[0] == (True, "torch.int32", 0),
         kinds=sorted(validlen_kinds),
         detail="0-d int32 tensor; as a Python int it would bake the frame length into the "
                "graph and recompile every frame (neuron_kv_cache.py:485-488)")

    # Executable code only: docstrings and comments in this module *discuss* .item() and
    # torch.cat at length, so a raw substring scan would fire on the prose. tokenize drops
    # both, leaving the statements that actually run.
    #
    # COUNTED and LOCATED, not "absent before a marker line". The earlier form split the
    # source at `def visible_kv_reference` and scanned only the text before it -- but that
    # method is the LAST of 16, so the "protected region" was everything except the final
    # method, and any .item() added in a new method appended after it (the most natural place
    # to add a helper) passed silently. Measured: appending a method containing
    # int(self._cursor[0].item()) left the old gate True. So the gate now asserts the module
    # contains EXACTLY ONE .item() and that it lies inside visible_kv_reference's own
    # ast-located line span -- invariant to method ordering.
    c5_path = os.path.join(_REPO, "lingbot_map", "heads", "neuron_camera_cache.py")
    code = _executable_source(c5_path)
    n_item = code.count(".item()")
    n_cat = code.count("torch.cat")
    ref_span = _method_span(c5_path, "NeuronCameraCache", "visible_kv_reference")
    item_lines = [i + 1 for i, ln in enumerate(code.splitlines()) if ".item()" in ln]
    item_outside = [] if ref_span is None else [
        ln for ln in item_lines if not (ref_span[0] <= ln <= ref_span[1])
    ]
    emit("exactly_one_item_and_it_is_inside_visible_kv_reference",
         n_item == 1 and n_cat == 0 and ref_span is not None and not item_outside,
         item_count_in_module=n_item, cat_count_in_module=n_cat,
         item_lines=item_lines,
         visible_kv_reference_span=str(ref_span),
         item_lines_outside_that_span=item_outside,
         detail="COUNT + ast SPAN, not position relative to a marker: the single .item() must "
                "be the one in visible_kv_reference (VERIFICATION ONLY, matching "
                "neuron_kv_cache.py:535-545) and there must be no torch.cat anywhere. A "
                "positional split let a host sync added in any later-declared method pass")

    # Semantic half of the same discipline: valid_len must be device arithmetic end to end.
    # `_n_pending` used to be a Python int read on the compute path, which made the
    # speculative frame's +1 a trace-time constant -- two NEFF specializations of one
    # streaming step. AST-walk visible_kv/visible_mask/visible_group for any read of a
    # host-int attribute.
    host_int_reads = _host_int_attr_reads(
        c5_path, "NeuronCameraCache",
        ("visible_kv", "visible_mask", "visible_group"),
        ("_n_stored", "_n_pending"),
    )
    emit("no_host_int_state_read_in_the_visible_path", not host_int_reads,
         offending_reads=host_int_reads,
         methods_scanned="visible_kv, visible_mask, visible_group",
         host_int_attrs="_n_stored, _n_pending",
         detail="valid_len = (_cursor + _pending) with BOTH device tensors. Reading the host "
                "mirror here would fold the keyframe/non-keyframe difference into the graph "
                "as a literal -- the exact coupling design.md :711-712's device cursor exists "
                "to remove")

    # The two gates above read SOURCE, so --inject_bug (which patches objects at runtime)
    # cannot reach them: a source-reading gate whose discriminating power is never shown is
    # exactly the vacuity class this suite is built against. So they are re-run here against
    # two synthetic mutants of C5's own source, in a temp file, and must report the mutation.
    # Both mutants are the real historical defects: a .item() added in a method declared AFTER
    # visible_kv_reference (which the earlier positional gate let through, measured), and
    # valid_len computed off the host int _n_pending.
    mutant_item_ok = mutant_host_ok = False
    mutant_detail = []
    with open(c5_path) as fh:
        c5_src = fh.read()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        m1 = os.path.join(td, "m_item.py")
        with open(m1, "w") as fh:
            fh.write(c5_src + "\n"
                     "    def _added_later(self, s):\n"
                     "        return int(self._cursor[s].item())\n")
        m1_code = _executable_source(m1)
        m1_span = _method_span(m1, "NeuronCameraCache", "visible_kv_reference")
        m1_lines = [i + 1 for i, ln in enumerate(m1_code.splitlines()) if ".item()" in ln]
        m1_outside = [ln for ln in m1_lines
                      if not (m1_span[0] <= ln <= m1_span[1])] if m1_span else ["no span"]
        # The gate's own predicate, applied to the mutant: it must be FALSE.
        mutant_item_ok = not (m1_code.count(".item()") == 1 and not m1_outside)
        mutant_detail.append(
            f"item_mutant: count={m1_code.count('.item()')} outside_span={m1_outside}")

        m2 = os.path.join(td, "m_host.py")
        with open(m2, "w") as fh:
            fh.write(c5_src.replace(
                "(self._cursor[s] + self._pending[s]) * self.frame_seqlen",
                "(self._cursor[s] + self._n_pending[s]) * self.frame_seqlen"))
        m2_reads = _host_int_attr_reads(
            m2, "NeuronCameraCache",
            ("visible_kv", "visible_mask", "visible_group"),
            ("_n_stored", "_n_pending"),
        )
        mutant_host_ok = bool(m2_reads)
        mutant_detail.append(f"host_int_mutant: reads={m2_reads}")

    emit("static_discipline_gates_have_discriminating_power",
         mutant_item_ok and mutant_host_ok,
         item_gate_rejects_late_method_mutant=mutant_item_ok,
         host_int_gate_rejects_n_pending_mutant=mutant_host_ok,
         evidence="; ".join(mutant_detail),
         detail="--inject_bug patches objects and cannot reach a source-reading gate, so the "
                "two above are re-evaluated against synthetic mutants of C5's own source and "
                "must report FALSE on both. Without this, a source gate reporting PASS proves "
                "only that it ran")

    # ---- V6: the F6 eviction guard, proved reached rather than assumed --------------
    emit("f6_guard_holds_shape3_is_1", shape3_bad is None and ev.shape3_seen == {1},
         shape3_values_seen=sorted(ev.shape3_seen),
         detail=shape3_bad or "all 16-equivalent streams, k AND v, every frame; iterating "
                              "BOTH axes (design.md writes kv_cache[f'k_{i}'], eliding the "
                              "list axis)")
    emit("f6_guard_was_actually_reached",
         ev.n_calls > 0 and not special_keys_seen and ev.n_shape2_decreases == 0,
         eviction_calls=ev.n_calls, body_entries=0, special_keys=sorted(special_keys_seen),
         shape2_decreases=ev.n_shape2_decreases,
         detail="positive proof the guard at attention.py:307 was EVALUATED and its body "
                "never ran. design.md omits this; without it the shape3 check above is "
                "vacuous on an unwritten cache")

    # ---- V6, PORT SIDE. The two gates above read only the GPU reference dict and the
    # EvictionWatch, both properties of untouched upstream code, so they are 1 no matter what
    # C5 does: measured, deleting BOTH of C5's shape[3] guards still reported 25/25 PASS. These
    # three interrogate C5 itself -- the live buffer, the constructor and the write -- and the
    # accept_shape3_gt_1 injection proves they discriminate. ---------------------------
    emit("f6_port_side_frame_axis_is_exactly_1_wide",
         c5.slot_shape[3] == 1 and c5.frame_seqlen == 1
         and all(t.shape[3] == 1 for t in c5.k) and all(t.shape[3] == 1 for t in c5.v),
         slot_shape=str(tuple(c5.slot_shape)), frame_seqlen=c5.frame_seqlen,
         k_shape3=sorted({int(t.shape[3]) for t in c5.k}),
         v_shape3=sorted({int(t.shape[3]) for t in c5.v}),
         detail="on the LIVE C5 instance and its allocated buffers. design.md's C5 row is "
                "[B,H,max_total_frames,1,head_dim]; this asserts that 1, which the whole "
                "no-eviction design rests on, rather than restating it")

    ctor_raised = ""
    try:
        NeuronCameraCache(num_iterations=1, trunk_depth=1, num_heads=1, head_dim=2,
                          device=dev, max_total_frames=4, frame_seqlen=2, dtype=dtype)
    except (ValueError, AssertionError) as e:
        ctor_raised = str(e)
    emit("f6_ctor_refuses_frame_seqlen_not_1", bool(ctor_raised),
         msg=ctor_raised[:110] or "(did not raise -- a frame_seqlen=2 buffer was CONSTRUCTED)",
         detail="never exercised before this gate existed: grep 'frame_seqlen=' over the "
                "harness returned 0 hits, so no non-default frame_seqlen was ever built")

    fs_small = NeuronCameraCache(
        num_iterations=1, trunk_depth=1, num_heads=1, head_dim=2, device=dev,
        max_total_frames=4, dtype=dtype,
    )
    wide = torch.randn(1, 1, 1, 2, 2, dtype=dtype, device=dev)
    write_raised, write_other = "", ""
    try:
        fs_small.append(0, 0, wide, wide)
    except (ValueError, AssertionError) as e:
        write_raised = str(e)
    except Exception as e:                                   # noqa: BLE001
        # An incidental index_copy_/shape error is NOT the guard doing its job; recorded
        # separately so it cannot be mistaken for a refusal, and never allowed to escape as a
        # traceback (which would exit 1 for the wrong reason -- see run_c5_camera_cache_checks).
        write_other = f"{type(e).__name__}: {e}"
    # And the write guard must survive a relaxed constructor: neuter the config value and the
    # check must still fire, because it compares against the LITERAL 1. This is the exact
    # single-line edit that was measured to make a frame_seqlen=2 write silently accepted.
    fs_small.frame_seqlen = 2
    write_raised_after, write_other_after = "", ""
    try:
        fs_small.append(0, 0, wide, wide)
    except (ValueError, AssertionError) as e:
        write_raised_after = str(e)
    except Exception as e:                                   # noqa: BLE001
        write_other_after = f"{type(e).__name__}: {e}"
    emit("f6_write_refuses_shape3_gt_1_independently_of_config",
         bool(write_raised) and bool(write_raised_after),
         raised_with_config_1=bool(write_raised),
         raised_with_config_relaxed_to_2=bool(write_raised_after),
         non_guard_errors=(write_other or write_other_after) or "(none)",
         msg=write_raised[:100] or "(did not raise)",
         detail="the append-side check must test shape[3] against the LITERAL 1, not against "
                "self.frame_seqlen. A config echo self-adjusts to whatever the constructor "
                "permitted and provides no independent protection -- measured: relaxing only "
                "the ctor made a [1,4,3,2,8] write ACCEPTED, storing 3 frames with valid_len 6 "
                "while the GPU entered the eviction body on 32/36 calls")

    attn_src = open(os.path.join(_REPO, "lingbot_map", "layers", "attention.py")).read()
    recompute = re.search(
        r"num_frame_per_block\s*=\s*k\.shape\[2\]\s*//\s*kv_cache\[f\"k_\{global_idx\}\"\]"
        r"\.shape\[3\]", attn_src)
    n_cat = len(re.findall(r"torch\.cat\(\(kv_cache\[f\"k_\{global_idx\}\"\]", attn_src))
    emit("shape3_invariant_is_self_perpetuating", recompute is not None,
         recompute_line_present=recompute is not None,
         detail="attention.py:236 re-derives num_frame_per_block from the cache's own "
                "shape[3], so 1 begets 1 forever. The fragile premise is nfpb == S on the "
                "FIRST prefill call, NOT frame_seqlen as design.md F6 states")
    emit("gpu_path_untouched_cat_still_present", n_cat >= 1,
         cat_sites=n_cat,
         detail="attention.py:239-240 must stay byte-identical; the reference is only a "
                "valid oracle while it does")

    rollback_after = hasattr(head, "rollback_last_frame")
    emit("hasattr_rollback_is_False_before_and_after",
         (not rollback_before) and (not rollback_after),
         before=rollback_before, after=rollback_after,
         detail="on the real CameraCausalHead INSTANCE, not the class (design.md V6)")

    # ---- capacity: ASYMMETRIC, and deliberately not a matched-failure test ----------
    small = NeuronCameraCache(
        num_iterations=1, trunk_depth=1, num_heads=1, head_dim=2, device=dev,
        max_total_frames=3, dtype=dtype,
    )
    one = torch.randn(1, 1, 1, 1, 2, dtype=dtype, device=dev)
    raised, at = "", 0
    try:
        for _ in range(10):
            small.append(0, 0, one, one)
            at += 1
    except AssertionError as e:
        raised = str(e)
    emit("capacity_raises_naming_max_total_frames",
         "max_total_frames" in raised and at == 3,
         at_frame=at, msg=raised[:110] or "(did not raise)",
         detail="ASYMMETRIC with the GPU by design: the camera head has NO assert at all "
                "(attention.py:239-240 cats until OOM), so design.md FM4's 'GPU raises no "
                "earlier' clause is inapplicable here. Deltas Owed #12. KEYFRAMES only -- see "
                "the non-keyframe gate below")

    # The other half of the capacity asymmetry, and the one the loop above cannot see because
    # it only ever appends KEYFRAMES. The GPU's non-keyframe branch (attention.py:249-256) is a
    # LOCAL torch.cat that never mutates kv_cache[k_j], so the GPU processes non-keyframes
    # indefinitely at any depth, attending n+1 keys and keeping its dict depth unchanged, with
    # no error. Charging capacity for that frame made C5 raise where the reference provably
    # cannot fail. Measured against the real head in this state: 7 keys attended, depth stays
    # 6, no error.
    cap = NeuronCameraCache(
        num_iterations=1, trunk_depth=1, num_heads=1, head_dim=2, device=dev,
        max_total_frames=6, dtype=dtype,
    )
    for _ in range(6):
        cap.append(0, 0, one, one)
    full_depth = cap.stored_frames(0, 0)
    cap.set_skip_append(True)
    nk_err, nk_valid, nk_depth = "", -1, -1
    try:
        _, _, nk_v = cap.append(0, 0, one, one)
        nk_valid, nk_depth = int(nk_v), cap.stored_frames(0, 0)
    except Exception as e:                                   # noqa: BLE001 - any raise fails
        nk_err = f"{type(e).__name__}: {e}"
    emit("non_keyframe_at_capacity_does_not_raise",
         not nk_err and nk_valid == full_depth + 1 and nk_depth == full_depth,
         n_stored_before=full_depth, valid_len_after=nk_valid, n_stored_after=nk_depth,
         error=nk_err or "(none)",
         detail="capacity is charged for COMMITS only, and one row above max_total_frames is "
                "reserved (spec_headroom) so the uncommitted frame has a physical row and "
                "valid_len still names where the write landed. Guarding before the "
                "skip_append branch aborted inference in a state the GPU cannot fail in")
    cap.set_skip_append(False)
    kf_err = ""
    try:
        cap.append(0, 0, one, one)
    except AssertionError as e:
        kf_err = str(e)
    emit("keyframe_at_capacity_still_raises",
         "max_total_frames" in kf_err and cap.stored_frames(0, 0) == full_depth,
         msg=kf_err[:110] or "(did not raise)", n_stored=cap.stored_frames(0, 0),
         detail="the paired half: moving the guard into the keyframe branch must not weaken "
                "the bound C5 exists to impose")

    # ---- the graph-facing path: compute_attention must actually be CALLABLE ----------
    # The suite previously had zero occurrences of 'compute_attention', so every check passed
    # while the sole method that feeds the NEFF was dead: it handed the adapter a
    # max_total_frames-key buffer, and attend()/attend_groups() require exactly plan.padded
    # keys. 1124 has NO divisor that is a multiple of TILE_QUANTUM=128
    # ([t for t in range(128, 36865, 128) if 1124 % t == 0] == []), so no legal quantized tile
    # could ever accept a raw-1124 buffer at any adapter configuration. Stub the adapter and
    # assert what it RECEIVES.
    class _AdapterProbe:
        """Records what compute_attention hands the C2 seam. No kernel, no Neuron stack."""

        def __init__(self):
            self.groups = None
            self.q = None

        def attend_groups(self, q, groups):
            self.q, self.groups = q, list(groups)
            k_pad = groups[0][0]
            return torch.zeros(q.shape[0], q.shape[1], q.shape[2],
                               dtype=k_pad.dtype, device=k_pad.device)

    probe = _AdapterProbe()
    ca = NeuronCameraCache(
        num_iterations=1, trunk_depth=1, num_heads=4, head_dim=8, device=dev,
        max_total_frames=1124, dtype=dtype, attention_adapter=probe,
    )
    ca.append(0, 0, torch.randn(1, 4, 1, 1, 8, dtype=dtype, device=dev),
              torch.randn(1, 4, 1, 1, 8, dtype=dtype, device=dev))
    ca_out = ca.compute_attention(0, 0, torch.randn(1, 4, 5, 8, dtype=dtype, device=dev))
    g_k, g_v, g_valid, g_plan = probe.groups[0]
    emit("compute_attention_feeds_the_adapter_a_tile_padded_buffer",
         len(probe.groups) == 1
         and tuple(g_k.shape) == (ca.plan.padded, 4, 8)
         and tuple(g_v.shape) == (ca.plan.padded, 4, 8)
         and g_plan.padded == ca.plan.padded == ca.padded_frames
         and ca.plan.tile % 128 == 0
         and ca.plan.padded >= ca.max_total_frames
         and torch.is_tensor(g_valid) and g_valid.dtype is torch.int32
         and g_valid.dim() == 0 and int(g_valid) == 1
         and tuple(ca_out.shape) == (1, 4, 5, 8),
         k_keys=int(g_k.shape[0]), plan=str(ca.plan),
         valid=f"{g_valid.dtype} dim={g_valid.dim()} value={int(g_valid)}",
         out_shape=str(tuple(ca_out.shape)),
         detail="plan.padded keys (NOT max_total_frames), a TILE_QUANTUM-aligned tile, and a "
                "0-d int32 device valid_len. Allocating the raw 1124 made this method "
                "unreachable at every adapter tile choice -- and the suite could not see it, "
                "because it never called compute_attention at all")

    # And the same buffer must satisfy the REAL adapter's shape check, not just the probe's.
    from lingbot_map.layers.neuron_attention import NeuronAttentionAdapter  # noqa: E402
    real_ad = NeuronAttentionAdapter(head_dim=8, max_seqlen_k=ca.plan.padded,
                                     tile=ca.plan.tile, check_env=False)
    emit("real_adapter_shape_contract_is_satisfiable_at_prod_capacity",
         real_ad.padded_seqlen_k == ca.plan.padded and real_ad.n_tiles == ca.plan.n_tiles
         and not [t for t in range(128, 36865, 128) if 1124 % t == 0],
         adapter_padded=real_ad.padded_seqlen_k, plan_padded=ca.plan.padded,
         quantized_tiles_dividing_1124=[t for t in range(128, 36865, 128) if 1124 % t == 0],
         detail="the empty divisor list is the proof that a raw-max_total_frames allocation "
                "is not merely wasteful but IMPOSSIBLE to attend: no legal tile makes "
                "padded_seqlen_k == 1124")

    # ---- reset: cursors only, buffers NOT reallocated -------------------------------
    ptr_before = c5.k[0].data_ptr()
    c5.reset()
    emit("reset_clears_cursors_without_reallocating",
         all(c5.stored_frames(i, j) == 0 for i in range(NUM_ITER) for j in range(DEPTH))
         and int(c5.visible_kv(0, 0)[2]) == 0
         and c5.k[0].data_ptr() == ptr_before,
         same_buffer=c5.k[0].data_ptr() == ptr_before,
         detail="buffers intentionally NOT reallocated; stale rows sit outside valid_len")

    # ---- production geometry: compute the design.md figure, do not trust it ---------
    prod = NeuronCameraCache(
        num_iterations=4, trunk_depth=4, num_heads=16, head_dim=128, device=dev,
        max_total_frames=1124, dtype=torch.float32,
    )
    rep = prod.memory_report()
    emit("prod_geometry_matches_design_C5", prod.num_slots == 16
         and tuple(prod.slot_shape) == (1, 16, 1152, 1, 128)
         and prod.max_total_frames == 1124 and prod.plan.padded == 1152
         and prod.plan.n_tiles == 1 and prod.plan.tile == 1152
         and prod.dtype is torch.float32,
         num_slots=prod.num_slots, slot_shape=str(tuple(prod.slot_shape)),
         capacity=prod.max_total_frames, plan=str(prod.plan), dtype=str(prod.dtype),
         detail="design.md :704-709 states the LIVE geometry, 16 slots of [1,16,1124,1,128] "
                "fp32 (NOT bf16); the frame axis is allocated 1152 = plan_tiles(1124+1).padded "
                "because C2's seam requires a TILE_QUANTUM-aligned key extent. Capacity is "
                "still 1124 committed frames. Deltas Owed #12")
    emit("prod_bytes_match_design_data_model_row",
         abs(rep["MB_k_only_live"] - 147.0) < 1.0
         and abs(rep["MB_k_and_v_live"] - 295.0) < 2.0
         and rep["pad_overhead_pct"] < 5.0,
         MB_k_only_live=rep["MB_k_only_live"], MB_k_and_v_live=rep["MB_k_and_v_live"],
         MB_k_and_v_allocated=rep["MB_k_and_v"],
         pad_overhead_pct=rep["pad_overhead_pct"],
         GB_k_and_v=round(rep["bytes_k_and_v"] / 1e9, 3),
         cat_calls_avoided_per_frame=rep["cat_calls_per_frame_avoided"],
         detail="computed from the live buffers, not hardcoded: design.md's Data Model row "
                "claims 2 x 147 MB ~= 0.30 GB for K+V at the LIVE extent, which *_live "
                "matches; the tile padding adds 2.49%, reported rather than hidden. C1's "
                "harness caught a stale table row exactly this way")

    n_fail = sum(1 for r in RESULTS if not r["pass"])
    out = {
        "suite": "neuron C5 camera cache vs CameraCausalHead (design.md V5(a)/V6)",
        "device": str(dev),
        "reference": "REAL CameraCausalHead -> CameraBlock -> CausalAttention, nothing "
                     "stubbed; attended k/v observed by shimming F.scaled_dot_product_"
                     "attention at attention.py:284",
        "injected_bug": args.inject_bug,
        "expected_verdict": "FAIL" if args.inject_bug else "PASS",
        "n_checks": len(RESULTS), "n_fail": n_fail,
        "verdict": "PASS" if n_fail == 0 else "FAIL",
        "prod_memory_report": rep,
        "results": RESULTS,
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "camera_cache_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n=== {out['verdict']}: {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    if args.inject_bug:
        print(f"=== injected bug {args.inject_bug!r}: wanted FAIL, got {out['verdict']}")
    print(f"=== wrote {path}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
