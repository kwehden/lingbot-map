# Numerical-tolerance baseline for the `KVCacheBackend` refactor

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

All five configurations, all recorded output keys:

| Configuration | `pose_enc` | `depth` | `depth_conf` |
|---|---|---|---|
| `streaming_flashinfer` | 0.0 | 0.0 | 0.0 |
| `streaming_sdpa` | 0.0 | 0.0 | 0.0 |
| `windowed_flashinfer` | 0.0 | 0.0 | 0.0 |
| `windowed_sdpa` | 0.0 | 0.0 | 0.0 |
| `streaming_flashinfer_fp32` (`force_fp32=True`) | 0.0 | 0.0 | 0.0 |

**Reproduced independently twice.** These values were measured on two separate Tier 2 spot
instances (jobs 13 and 15) that produced byte-identical floors. Two runs on different hardware
agreeing is a materially stronger basis than a single run's numbers.

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
