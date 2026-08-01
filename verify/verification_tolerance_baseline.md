# Numerical-tolerance baseline for the `KVCacheBackend` refactor

> **OUTCOME (2026-08-01): the tolerances derived below were never approached — the refactor is
> bit-exact.** Compare mode (job 21, `SUCCEEDED`, 29m36s) diffed all five configurations against
> this baseline and every `max_abs_diff` **and** `max_rel_diff` came back exactly `0.0`, with the
> windowed keyframe-decision sequence identical on all 613 frames (`differing=0/613`) and
> `alignment_mode` unchanged. The gate was therefore never the binding constraint.
>
> Keep the derivation on record regardless: it is what makes "0.0" a *verified* result rather than
> an unexamined one, and any future change that does perturb numerics will need it. The coverage
> caveats below still apply — bit-exactness is established for the keys that are **gated**, and the
> cross-window alignment and keyframe-decision keys only became gated in the run that produced this
> outcome (they had been saved to these baseline artifacts and silently never compared).

**Status:** measured and retained. Partial pass — 3 of 5 output keys (see
[Coverage gap](#coverage-gap--3-of-5-output-keys)).
**Task:** `TASK-003` ("Derive the numerical-tolerance acceptance gate from the unmodified-code
noise floor"), implementing `spec/design.md` Decision 4.
**Consumed by:** `TASK-021`, `TASK-022`, `TASK-023`, and `TASK-027`'s compare mode, which reads
these tolerances out of the baseline `results.json`.
**Measured:** 2026-07-31, on unmodified pre-refactor code.

> **Location note.** `TASK-003`'s write lease names `docs/verification_tolerance_baseline.md`, but
> `docs/` is gitignored in this repository (`.gitignore:18`), so a file there could never be
> committed or reviewed. This file therefore sits in `verify/`, beside the harness that produces the
> numbers, under the task's own "or equivalent location the team already uses for such notes"
> allowance. The lease pattern should be updated to match.

This file exists so that later parity checks compare against a *recorded, justified* tolerance
rather than an ad hoc one. Do not tighten or widen these numbers to make a downstream check pass;
per `TASK-021`'s "On FAIL" clause, a failing parity check is a finding, not a tolerance problem.

## How the numbers were produced

`verify/run_kv_cache_backend_checks.py --mode baseline`, run on **Tier 2** (SkyPilot, real
FlashInfer, CUDA 12) against the `freiburg1_desk` TUM fixture from `TASK-002` — 613 frames — with
the `TASK-001` checkpoint `lingbot-map-long.pt`, which loaded with `missing=0 unexpected=0` on
every model build. Tier 1 cannot produce these numbers: its CUDA 11.4 / A10G container cannot run
FlashInfer at all.

For each configuration the harness runs the model **twice, back-to-back**, on identical input and
takes the per-output-key max absolute difference between the two runs. That pairwise diff *is* the
noise floor; it is not recoverable after the fact from saved predictions, because only the first
run's output is retained on disk.

## Measured noise floor

All five configurations, for the three output keys the production configuration actually emits.
This is **not** full output coverage — see [Coverage gap](#coverage-gap--3-of-5-output-keys) below,
which is wider than the key count suggests.

**Only the absolute difference was measured.** Decision 4 step 2 (and `TASK-003` step 2) call for
max absolute **and** max relative difference per key; the harness computes only `max_abs`
(`verify/run_kv_cache_backend_checks.py:353`). With an exactly-zero absolute floor the relative
figure is zero or undefined, so nothing downstream changes — but the relative half of the required
measurement was not taken, and this table should not be read as fully satisfying that clause.

| Configuration | `pose_enc` | `depth` | `depth_conf` |
|---|---|---|---|
| `streaming_flashinfer` | 0.0 | 0.0 | 0.0 |
| `streaming_sdpa` | 0.0 | 0.0 | 0.0 |
| `windowed_flashinfer` | 0.0 | 0.0 | 0.0 |
| `windowed_sdpa` | 0.0 | 0.0 | 0.0 |
| `streaming_flashinfer_fp32` (`force_fp32=True`) | 0.0 | 0.0 | 0.0 |

**Reproduced independently three times.** These values were measured on three separate Tier 2 spot
instances (jobs 13, 15, and 16) that produced byte-identical floors. Three runs on different hardware
agreeing is a materially stronger basis than a single run's numbers. Job 16 is the run that completed
cleanly end-to-end (`SUCCEEDED`, 38m23s), so its `results.json` is the canonical baseline reference.

### Read 0.0 correctly: this is determinism, not an unexercised measurement

A zero floor is the *expected* outcome here, not a suspicious one, and it should not be presented
as though run-to-run nondeterminism was measured and found to be tiny. The harness runs inference
under `torch.no_grad()` on a `.eval()` model, with no dropout active and byte-identical inputs, on
fixed hardware. Two such runs are bitwise identical, so the difference is exactly zero by
construction.

What the measurement therefore establishes is: **this pipeline is deterministic on identical
input.** That is a stronger and cleaner basis for a parity gate than a small-but-nonzero floor
would have been — a later parity check that shows *any* difference is showing a real effect of the
refactor, not sampling noise. It does mean the floor carries no information about tolerance to
nondeterminism, because there is none to tolerate.

Note the harness does not set `torch.manual_seed` or the deterministic-algorithm flags. It does not
need to for this measurement (no RNG is consumed on the inference path under `eval()`/`no_grad()`),
but that also means these zeros are a property of this fixed configuration, not a guarantee that
would survive adding a stochastic op.

### Decision 4 step 5 escalation check: PASSES, no escalation warranted

Step 5 requires flagging a noise floor that is "surprisingly large (e.g., an order of magnitude
above the default)" to the tech lead rather than silently adopting it as the new bar. Zero is the
opposite of that trigger. Nothing to escalate; recorded here so the check is visibly performed
rather than skipped.

## Derived acceptance tolerances

Decision 4 sets the gate at `max(default, 3 × measured_noise_floor)`. Since the measured floor is
0.0, `3 × 0.0 = 0.0` and the tolerances **reduce exactly to Decision 4's stated defaults**:

| Path | `rtol` | `atol` |
|---|---|---|
| bf16 (`streaming_flashinfer`, `streaming_sdpa`, `windowed_flashinfer`, `windowed_sdpa`) | `1e-3` | `1e-5` |
| `force_fp32` (`streaming_flashinfer_fp32`) | `1e-5` | `1e-6` |

These are the values `TASK-021`/`022`/`023` must use. The floor did not widen the gate.

**Caveat for whoever re-measures this.** Compare mode widens **only `atol`** by
`3 × max(noise)` and never touches `rtol` (`run_kv_cache_backend_checks.py:485-490`, `:512-518`,
`:538-544`). With a zero floor that is identical to the defaults above, so it is currently harmless.
But if a future re-measurement returns a nonzero floor, `rtol` will silently stay at its default
while `atol` moves — which is not what Decision 4's `max(default, 3 × noise_floor)` says. Fix the
widening logic before trusting a nonzero floor.

## Coverage gap — 3 of 5 output keys

`TASK-003`'s verification clause asks for a recorded noise floor **per output key** across
`pose_enc`, `depth`, `depth_conf`, `world_points`, and `world_points_conf`. Only the first three
are present above.

`world_points` / `world_points_conf` are emitted only when `enable_point=True`. The production
configuration used by both `benchmark/methods/lingbot_map.py` and `demo.py` leaves it at its
`False` default, so the model never produces those tensors, and the comparison helper correctly
classifies absent-in-both as parity rather than as a divergence.

This is a **defensible partial pass, not a silent one**: the two missing keys are unmeasurable on
the production configuration, and measuring them would require a non-production `enable_point=True`
run that no task currently requests. Recorded so a future reader does not mistake the three-key
table for full coverage.

Resolve it one of two ways — a scope decision for the tech lead, not something to settle by editing
the table above:

1. Narrow `TASK-003`'s key list to the keys the production configuration actually emits, or
2. Add an explicit `enable_point=True` measurement covering the remaining two keys.

Until then, any `TASK-021`/`022`/`023` parity claim about `world_points`/`world_points_conf` rests
on no measured floor and should say so.

### The gap is wider than "3 of 5": three windowed outputs have no gate at all

`_OUTPUT_KEYS` (`run_kv_cache_backend_checks.py:55`) lists five keys, so "3 of 5" describes only the
*named* set. The model emits more than those five, and the unnamed ones are never diffed by compare
mode:

- `inference_streaming` also returns `images` (`gct_stream_window.py:659`) — visualization payload,
  low risk.
- `inference_windowed` additionally returns **`chunk_scales`**, **`chunk_transforms`**
  (`gct_stream_window.py:952-955`), and `alignment_mode`.

`chunk_scales` / `chunk_transforms` are the **cross-window alignment outputs** — arguably the
numerics most exposed to a KV-cache refactor in windowed mode, since they depend on per-window state.

**RESOLVED 2026-08-01 — this hole is now closed.** The omission was not intentional; it was an
oversight, found by loading a real staged `.pt` and listing its keys rather than reading the return
annotations. Compare mode now diffs all three (`_ALIGNMENT_KEYS`), and inspecting the artifacts
turned up **two more** ungated outputs that `_OUTPUT_KEYS` never named:

- **`is_keyframe`** `[1, 613]` (bool) and **`frame_type`** `[1, 613]` (uint8) — the **keyframe
  decision sequence**, produced by exactly the machinery this refactor moves (`_set_skip_append`,
  `_defer_eviction`, `rollback_last_frame`, `execute_deferred_eviction`). Gated as `_EXACT_KEYS`,
  compared element-wise rather than at `rtol`/`atol`: a keyframe decision is discrete, so "close"
  is meaningless, and *which* frames flipped is the useful diagnostic. This closes `design.md`
  verification **item 3** for the two entry points this harness drives — not all of `TASK-023`,
  which also covers flow-threshold-driven decisions never exercised here.

These keys had a measured floor of 0.0 available all along (the tensors were in the baseline
artifacts), so closing the gate required **no baseline re-run**. Job 21 reports parity on every one
of them. The gates were validated by perturbation against the real artifact: changing only
`chunk_scales`, or flipping one keyframe decision out of 613, now FAILS — and the old five-key gate
demonstrably PASSED both.

The same oversight existed in the **item 2 harness comparison**, which checked only `depth` while
the output carries `depth`, `pose`, `intrinsics` and `confidence` (613 ndarrays each). 613 camera
poses per config were ungated — a wider hole than this one, since pose is the primary output of a
reconstruction model. Now gated per key; validated by confirming a single perturbed camera pose
fails and names the frame, where the depth-only check reported PASS. That fix landed after job 21
launched, so it applies to the next run.

## Related gate that is weaker than it looks

Not this task's artifact, but recorded here because a reader deriving confidence from
`results.json` will see it pass: the **FlashInfer half of the `kv_cache_info_states` check is
vacuous**. `get_kv_cache_info` reads only the SDPA dict, which is `{}` for FlashInfer, so both
recorded states are `{0, 0.0}` and compare mode's equality gate passes even if a refactor breaks
FlashInfer cache accounting entirely. That zero-return is Decision 3's deliberately-preserved bug.
Details in `_check_get_kv_cache_info`'s docstring.

**Confirmed on hardware.** This was first derived by reading the code, then job 16's real
`results.json` matched the prediction exactly: FlashInfer `[{0, 0.0}, {0, 0.0}]` versus SDPA
`[{24 blocks, 0.0 MB}, {24 blocks, 25.78 MB}]`. The SDPA leg's values change across inference; the
FlashInfer leg's cannot.
