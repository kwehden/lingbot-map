# Tier 2: the version-controlled trn2 verification job

`spec/neuron-port/tasks.md` TASK-N18.

## What this replaces

Every Neuron run on this project so far went out on a hand-rolled SkyPilot YAML in a scratch
directory — `_v1.yaml`, `_np2.yaml` … `_np4.yaml`, `_r.yaml` — none of it in any repository. The
recipe works. The problem is that nine separate details of it each cost a real run to learn, and
none of those nine were written down anywhere a second person could find them. A Tier 2 result
produced that way is also unreproducible by anyone who is not the person who ran it: the artifact
records no revision, and the job that made it exists on one desk.

This directory is that recipe, in version control, with each of the nine details encoded as
something that fails loudly rather than as a line in a runbook.

```
job_tier2.yaml.tmpl      the job. A template: it is not launchable by hand, on purpose.
launch_tier2.py          renders it, validates the rendered text, launches, collects, tears down.
site_config.example.json every account-shaped value, as placeholders. Nothing real is committed.
stages/                  the embedded scripts, readable. The job carries them as base64.
test_tier2_job.py        the constraints, exercised in the direction that fails.
```

## Why the job is a template

A rendered job contains a bucket name, a prefix, a region, a zone list and a remote URL. This
repository has public remotes, so none of those can be committed. Every one arrives as a
doubled-at-sign token substituted from `--site-config`, validated by *shape* — `subnet_id` must
look like a subnet id — and never against an expected value. A validator that knows the right
subnet id has the subnet id in it.

The other reason: handing the template straight to `sky launch` renders its tokens literally, and
the resulting failure looks like a capacity failure rather than an operator error. Two of the nine
constraints below were learned exactly that way.

## Why `stages/` is separate from the job

The scratch YAMLs base64-encode their embedded Python inline, which is correct (see constraint 1)
and also means the only committed form of those scripts is unreadable. Nobody can review a base64
blob, so nobody did. Here the readable original is the committed artifact and `launch_tier2.py`
encodes at render time.

Each stage script resolves `rec.py` through `LINGBOT_T2_REC` before falling back to its on-instance
path, which is what makes them runnable at the desk under `test_tier2_job.py`. A stage whose only
exercise is trn2 spot time is a stage nobody exercises.

## Where the nine constraints are encoded

TASK-N18 lists nine operational constraints. Each one is a run somebody already paid for.

| # | Constraint | Encoded as |
|---|---|---|
| 1 | Base64-encode embedded scripts to survive SSM + YAML quoting | The template embeds `stages/*` as base64 tokens only. `check_rendered` refuses any heredoc or inline `python -c` in the rendered text. |
| 2 | Keep `sleep` ≤ 115 s inside a single command | `check_rendered` refuses a `sleep > 115` that is not inside a detached watcher. The scope is a *single command* — an SSM `AWS-RunShellScript` body — so a long `timeout` on a foreground stage is fine; a long `sleep` is not. |
| 3 | Use `setsid nohup … < /dev/null` for watchers | `stages/spot_watch.sh` is launched that way, and `check_rendered` refuses any line that starts a watcher with any of the three parts missing. Checked over *logical* lines, so the backslash continuation the real job uses is not a false positive. |
| 4 | Verify `git ls-remote` matches local HEAD before launching | `check_revision` refuses on mismatch, on a dirty tree, on a missing branch, and on an `ls-remote` *failure* — not a skip, because an unreachable remote is when a stale local ref is most likely to be what gets cloned. The job's `repo` stage then re-checks the cloned sha against the one that was validated, because an instance had to be acquired in between. |
| 5 | Activate venvs with `source <venv>/bin/activate` | The `venv` stage sources it; `check_rendered` refuses `$VENV/bin/python` anywhere and refuses a job with no `activate` at all. Invoking the interpreter directly leaves the PJRT plugin path unset, the device count reads zero, and the run looks like a hardware finding. That broke Phase 0 run 1. |
| 6 | Export both `PJRT_DEVICE=NEURON` and `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` | Re-exported in every stage that touches the runtime; `check_rendered` refuses a job missing either. `PJRT_DEVICE` alone is why the first `neuron_probe.json` recorded no devices on a 16-device host. |
| 7 | Pin and record the SDK/compiler version (REQ-088) | `stages/sdk_versions.py`, and `check_rendered` refuses any profile that does not include that stage. A tolerance derived under one pinned set is not valid under another — NKI-1559 is the reason. |
| 8 | Flush to S3 after each stage (REQ-092) | Every stage's `flush` sits *outside* its `if stage …` guard, so a skipped stage still records and still flushes, and the property is checkable by counting: `check_rendered` refuses unless the guard set equals the flush set. Phase 0 run 2 was preempted mid-compile and was recoverable only because of this. |
| 9 | Parameterize the capacity category; do not hard-code one | `--category` is required, validated against REQ-090's categories, and cross-checked against the profile's own `invokes_nki_kernel` declaration. Two combinations are refused outright (below). |

## What constraint 9 refuses, and what it deliberately does not decide

`check_category` refuses:

- **a kernel check on REQ-090's cheap capacity.** `attention_cte` dropped trn1/inf2 support
  (REQ-081), so `inf2.xlarge` is not a cheap answer to that question — it is a different question.
  This is the load-bearing half of "parameterize the category".
- **a kernel-free check on trn2**, unless `--allow-expensive-venue` is passed, in which case the
  override is recorded in the manifest with the flag named. REQ-090 says run kernel-free checks on
  the cheapest sufficient capacity, and the gap is roughly two orders of magnitude.
- **a `--category` that contradicts the profile.** The profile declares whether it invokes an NKI
  kernel; a contradicting category is an operator error, not a preference.

It does **not** resolve the venue conflict `design.md`'s V4 row holds open. Parameterizing makes
both outcomes executable from tracked artifacts; choosing one is not this file's call. REQ-091 also
leaves open whether the trn1/inf2 exclusion is `attention_cte`-specific or blanket, so until that
is resolved the narrow reading governs and REQ-090 applies only to kernel-free checks.

## Rendering proves nothing about capacity

`--render` is the default and produces artifacts only. REQ-087 rejects four kinds of evidence that
a Neuron instance is obtainable: a `run-instances --dry-run` result, a service-quota figure, a
`describe-instance-type-offerings` entry, and a spot-price record. That is not caution — the same
dry-run returned "Request would have succeeded" for a `trn1.2xlarge` in an AZ where
`describe-instance-type-offerings` says the type is not offered at all, and for a `p5.48xlarge` on
a night when H100 was provably unobtainable in all six configured regions. The only admissible
evidence is an actual launch that reached a usable state.

Add a fifth rejected class from finding 6 below: **the remediation sentence inside a capacity error**
("you can currently get capacity by choosing …"). Three of those arrived within nine seconds naming
each other's zones, so they cannot all have been true, and one named a zone that had refused four
seconds earlier. It reads like an answer and is not one.

So the launcher writes `"capacity_evidence": null` into its manifest and *cannot* make it true.
The only writer of the run's `capacity` block is `stages/identity.py`, which runs on the instance
and populates it because an instance answered IMDS with its own id — the one thing none of the four
rejected evidence classes can do. A test asserts the launcher never assigns a truthy value there.

The same reasoning governs the category: `identity.py` derives `category_reached` from the IMDS
`instance-type`, and the launcher's request is recorded separately as `category_requested`. REQ-090
scores which category the check *fell into*, and a spot `any_of` list can hand back a different
zone or size than the first preference. A request is not a result.

## How an interrupted run is classified

`trn2.48xlarge` is spot-only on current evidence, so preemption is the common case, not the exotic
one. REQ-092: an infrastructure-level interruption is recorded as
inconclusive-infrastructure-interrupted with cause and elapsed time, **never** as a phase failure.

The mechanism is structural rather than detective. `terminal_state` is written in exactly one
place — `stages/verdict.py`, in the job's last stage. A run that is killed never reaches that line,
so its last flushed artifact has no `terminal_state` at all, and `--collect` reads that absence as
INCOMPLETE. Nothing has to have *detected* the preemption for the classification to come out right.
`stages/spot_watch.sh` only supplies the cause and the elapsed time when a notice does arrive.

The watcher writes a **sidecar** (`tier2_interrupt.json`), not the run's own artifact, because
`rec.py` is read-modify-write and the one moment the watcher matters is the one moment the main
script is also writing. Separate S3 objects cannot collide; `verdict.py` merges the sidecar at the
end, when there is exactly one writer left, and only a sidecar with a `cause` counts — a spot
*rebalance recommendation* is recorded as a warning and is not an interruption, so a run that
receives one and then finishes is not mislabelled.

A compile that ran out of time is `INCONCLUSIVE_COMPILE_TIMEOUT`, never a kernel failure
(NKI-1590), and `inconclusive_stages` is recorded even when the overall verdict is FAIL so that a
timeout sitting beside an unrelated failure keeps its own attribution instead of being absorbed.

Per REQ-089, an out-of-tolerance or non-finite comparison is not a finding about LingBot-Map's
cache design until it has been checked against the `attention_cte` defects known for the pinned
version — the NKI-1559 NaN class specifically. `verdict.py` therefore writes an `attribution` block
with `pending_req089` set rather than implying an attribution the run cannot support.

## Teardown

`--teardown` refuses any name this runner did not generate, and emits `sky jobs cancel -n` for that
one job and nothing else — no `terminate-instances`, no `delete-volume`, no filter that could
match a resource this initiative did not create. Surviving volumes are **reported, not deleted**:
REQ-064 forbids deleting or terminating any capacity-allocation resource without explicit user
direction, and an orphan of unknown provenance is precisely that case.

Teardown is mostly a fallback now, because the run is a **managed job**: SkyPilot terminates the
job's cluster when the job ends, including when it fails. That is a stronger form of REQ-064 than a
follow-up command — the earlier `sky down` shape only ran if somebody remembered to run it.

## Two things authoring this surfaced

1. **The validated watchdog does not use the form constraint 3 requires.** `_v1.yaml` starts its
   timeout guard as `( sleep 7800; … ) &` — no `setsid`, no `nohup`, no `< /dev/null`. It survived
   because SkyPilot's `run:` block does not SIGHUP its children. It would not survive the SSM path,
   which is how the job now travels. Recorded rather than quietly "fixed", because the constraint
   as written is right and the precedent that appears to contradict it is the thing that needs
   explaining.

2. **The subnet does not reach SkyPilot through the task YAML.** SkyPilot is not installed on the
   desk; it runs on the controller, and the subnet it places instances in comes from the
   controller's own config. `subnet_id` and `vpc_name` are therefore shape-validated and recorded
   in the manifest but not rendered into the job. Making them take effect is a controller-side
   change.

   This paragraph used to end "the one part of the job that cannot be verified without a launch."
   That was wrong, and reading the controller's config instead of launching against it is what
   showed why: a config that pins `aws.vpc_names` selects the VPC **by name**, and SkyPilot then
   picks a subnet inside it that sits in the zone the job asked for. `zones` *is* rendered. So a
   subnet added to that VPC in a new zone becomes eligible with no controller change at all, and
   the placement mechanism is readable in advance — only whether capacity exists in it is not.

## Two things launching against a real controller surfaced

3. **`sky launch` could never have worked here.** The controller carries an admin policy that
   raises on `CLUSTER_LAUNCH` and `CLUSTER_EXEC`: direct cluster launches are refused, `sky jobs
   launch` is the accepted entry point, and managed jobs auto-terminate to stop idle clusters
   accumulating cost. The first version of this runner emitted `sky launch -c NAME`, which the
   controller would have rejected before requesting any capacity — and `sky jobs launch` has no
   `-c` flag at all. Found by reading the installed policy, not by paying for a launch.

4. **The controller's region and the job's region are different facts.** `--launch` sent
   `ssm send-command --region` the *job's* region while its target is the controller, which sits in
   another one; a controller with `use_ssm` reaches instances in regions it does not live in. There
   is now a `controller_region` key, and `--launch` checks the controller is `Online` to SSM there
   before sending anything — because an instance id is not a durable name for a controller either,
   and the one this runner was written against no longer exists in any state.

## Three things the verifying run surfaced

5. **An unpinned `image_id` yields an instance with the Neuron driver and nothing else.** The
   verifying run left `image_id` unset, so SkyPilot chose its own AMI. `neuron_ls` enumerated the
   device and `/dev/neuron0` existed — the driver was there — and the `venv` stage found no venv at
   any searched path, and all eight recorded modules were unimportable: `torch`, `torch_xla`,
   `torch_neuronx`, `neuronxcc`, `neuronx_distributed`, `neuronx_distributed_inference`, `nkilib`,
   `nkilib.core.attention`. The run still passed, because `noop` asks for nothing that needs them.
   No other profile is runnable on that AMI. `image_id` is therefore not the optional convenience
   its note used to describe; leaving it unset is how you pay for an instance that cannot do the
   work. Constraints 5, 6 and 7 all went unexercised for exactly this reason.

6. **The capacity error's own remediation advice was falsified in all three directions within
   seconds.** First pass: `us-east-2b` refused with `InsufficientInstanceCapacity` and said to try
   2a or 2c; 2a refused four seconds later and said to try 2b or 2c; 2c refused five seconds after
   that and said to try 2a or 2b. Every zone named the other two as having capacity while refusing
   itself. SkyPilot then retried the list and 2b — the zone that had refused first — succeeded. So a
   refusal does not persist twenty seconds, and the *remediation text inside an AWS capacity error*
   joins the four evidence classes REQ-087 already rejects. That is the fourth independent
   confirmation on this project, and the sharpest: the two claims were simultaneous and
   contradictory, so at least one was wrong at the moment it was printed.

7. **The rebalance-recommendation path is exercised, not just asserted.** The run received a spot
   rebalance recommendation and finished anyway. `tier2_interrupt.json` carries a
   `rebalance_recommended_utc` and no `cause`, so `verdict.py` merged it as a warning; the verdict
   is `PASS` and `inconclusive_stages` is empty. The distinction the interruption section describes
   between a recommendation and an actual interruption now has a run behind it.

## Status

**Verified.** TASK-N18's verification is discharged by run `20260817T160414Z-0a536bd-3bf89d`: it
launched from the committed job, obtained spot capacity (one `inf2.xlarge` in `us-east-2b`; the
instance id is in the run's own artifact, not here), reached a usable state, flushed its artifact to
S3 after every stage, wrote
`terminal_state: reached_complete_stage` with verdict `PASS`, and its cluster was terminated by the
managed-job mechanism — leaving no running instance and no volume in `available` state.

Two limits on what that verifies. It ran on **`inf2.xlarge`, not `trn2.48xlarge`**, which is
REQ-090's correct venue for a kernel-free check and about 113× cheaper; whether trn2 capacity is
obtainable is a separate question this run does not touch. And it landed in **`us-east-2b`**, so it
says nothing about placement in `us-east-2c` / the private-c subnet — that was a secondary goal, and
finding 6 is why it was given up rather than retried.

The ordering trap the task records is also discharged. Constraint 4 requires `git ls-remote` to
match local HEAD, and committing this runner moved HEAD past what TASK-N00 published; TASK-N00's
publish steps were re-run for the new revision first, so the run's `revision` block records
`local_head == published_head == 0a536bd`.

## Usage

```bash
# Render and validate only. Prints the manifest and the SSM command bodies. Costs nothing.
python3 verify/neuron/tier2/launch_tier2.py \
    --profile noop --category no-nki-kernel --instance-type <cheap-type> \
    --site-config ~/.config/lingbot/tier2_site.json

# Launch (spends money; trn2 is spot-only at $8.596/hr in the validated region).
… --profile c5-parity --category nki-kernel --instance-type trn2.48xlarge \
  --c5-tolerance <tolerance> --launch

# Collect and classify a finished or interrupted run.
… --collect <run-id>

# Tear down. Refuses any cluster name it did not generate.
… --teardown lingbot-t2-<profile>-<run-id>
```

`--c5-tolerance` has no default and must not acquire one: the arm exits 2 `MEASURED_NOT_SCORED`
without a tolerance, and 2 is not a pass. Where the tolerance comes from is TASK-N08 owed step 3's
question, not this runner's.
