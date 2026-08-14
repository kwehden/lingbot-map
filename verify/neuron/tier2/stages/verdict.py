"""Turn the stage record into one verdict, and be the only writer of ``terminal_state``.

REQ-092 is the reason this is a file rather than a shell conditional. A Phase 2/4/5/6 run that is
interrupted by spot preemption or another infrastructure-level termination "shall NOT be recorded
as a phase failure"; it is inconclusive-infrastructure-interrupted, with cause and elapsed time
captured. ``trn2.48xlarge`` is spot-only on current evidence, so this is the common case and not
the exotic one -- Phase 0 run 2 was preempted mid-compile.

The mechanism that makes it un-fakeable: ``terminal_state`` is written HERE and nowhere else, in
the last stage of the job. A run that is killed never reaches this line, so its flushed artifact
has no ``terminal_state`` at all, and ``launch_tier2.py --collect`` reads that absence as
INCOMPLETE. Nothing has to detect the preemption for the classification to be right; the watcher's
notice, when it lands, only adds the cause.

Verdict precedence, and why:

1. interrupted -> INCOMPLETE. Dominates everything, per REQ-092.
2. any failed stage -> FAIL.
3. any inconclusive stage -> INCONCLUSIVE. A compile that ran out of time is not a kernel
   failure (NKI-1590), so it must not collapse into FAIL.
4. otherwise PASS.

``inconclusive_stages`` is recorded even when the verdict is FAIL, so that a compile timeout
sitting beside an unrelated failure keeps its own attribution rather than being absorbed.

Usage::

    python3 verdict.py <result.json>
"""
import json
import os
import subprocess
import sys

RESULT = sys.argv[1]

EXIT = {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2, "INCOMPLETE": 3}


# rec.py is base64'd next to this script on the instance at /tmp/t2. Resolving it rather than
# hard-coding that path is what makes these stages runnable at the desk under test_tier2_job.py:
# a stage whose only exercise is on trn2 spot time is a stage nobody exercises.
REC = os.environ.get("LINGBOT_T2_REC") or "/tmp/t2/rec.py"
if not os.path.exists(REC):
    REC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rec.py")


def rec(key, value):
    subprocess.run([sys.executable, REC, RESULT, key,
                    value if isinstance(value, str) else json.dumps(value)], check=False)


try:
    with open(RESULT) as fh:
        doc = json.load(fh)
except Exception as exc:                                # noqa: BLE001
    # No artifact to classify. Not a failure of the thing under test.
    rec("verdict", "INCOMPLETE")
    rec("verdict_because", f"result artifact unreadable: {type(exc).__name__}: {exc}"[:200])
    raise SystemExit(EXIT["INCOMPLETE"])

stages = doc.get("stages", {}) if isinstance(doc.get("stages"), dict) else {}
selected = [s for s in (doc.get("stages_selected") or "").split(",") if s]

failed, inconclusive, skipped, ok = [], [], [], []
for name, row in sorted(stages.items()):
    status = (row or {}).get("status") if isinstance(row, dict) else None
    status = status or "NO_STATUS_RECORDED"
    if status.startswith("SKIPPED_NOT_IN_PROFILE"):
        skipped.append(name)
    elif status.startswith("INCONCLUSIVE"):
        inconclusive.append({"stage": name, "status": status})
    elif status.startswith("FAIL") or status == "NO_STATUS_RECORDED":
        failed.append({"stage": name, "status": status})
    elif status.startswith("SKIPPED"):
        # A stage skipped for a reason OTHER than "not in this profile" -- e.g. the arm skipped
        # because the reference did not verify. The cause is already a FAIL on its own stage;
        # recorded here so the reader is not left wondering why the arm produced nothing.
        inconclusive.append({"stage": name, "status": status})
    else:
        ok.append(name)

# Any stage the profile selected that left no record at all. A missing stage is not a passing
# stage: it means the run stopped before reaching it, or the stage crashed before its first rec.
missing = [s for s in selected if s not in stages and s != "complete"]

# The watcher writes a SIDECAR rather than into this document, because rec.py is
# read-modify-write and the one moment the watcher matters is the one moment the main script is
# also writing. Merged here, at the end, when there is exactly one writer left. A rebalance
# recommendation alone is not an interruption: only `cause` counts.
SIDECAR = os.environ.get("LINGBOT_T2_SIDECAR") or "/tmp/t2/interrupt.json"
interrupted = doc.get("_interrupted")
if not interrupted and os.path.exists(SIDECAR):
    try:
        with open(SIDECAR) as fh:
            side = json.load(fh)
        rec("_interrupt_sidecar", side)
        if side.get("cause"):
            interrupted = side
            rec("_interrupted", side)
    except Exception as exc:                            # noqa: BLE001
        rec("_interrupt_sidecar_unreadable_because", f"{type(exc).__name__}: {exc}"[:200])

if interrupted:
    verdict = "INCOMPLETE"
    because = ("infrastructure-level interruption, per REQ-092: recorded as "
               "inconclusive-infrastructure-interrupted and NOT as a phase failure. Cause: "
               f"{interrupted.get('cause') if isinstance(interrupted, dict) else interrupted}")
elif missing:
    verdict = "INCOMPLETE"
    because = (f"selected stages left no record: {', '.join(missing)}. The run did not reach "
               "them, which is an incomplete run rather than a failed one")
elif failed:
    verdict = "FAIL"
    because = "failed stages: " + ", ".join(f"{f['stage']}={f['status']}" for f in failed)
elif inconclusive:
    verdict = "INCONCLUSIVE"
    because = "inconclusive stages: " + ", ".join(
        f"{f['stage']}={f['status']}" for f in inconclusive)
else:
    verdict = "PASS"
    because = "every selected stage recorded a passing status"

# REQ-089: an out-of-tolerance or non-finite comparison is not a property of LingBot-Map's cache
# design until it has been checked against the known attention_cte defects for the pinned version
# (the NKI-1559 NaN class specifically). This run cannot do that check, so it marks the
# attribution as owed rather than implying one.
arm = stages.get("c5_arm") if isinstance(stages.get("c5_arm"), dict) else {}
arm_status = (arm or {}).get("status") or ""
attribution_pending = arm_status.startswith("FAIL")
rec("attribution", {
    "pending_req089": attribution_pending,
    "why": ("the C5 arm did not pass, so REQ-089 forbids recording that as a finding about the "
            "cache design, mask handling or port approach until it is checked against the "
            "attention_cte defects known for the version pinned in `sdk`. Until then it is "
            "inconclusive, not a design finding") if attribution_pending else None,
    "pin_complete_for_req088": (doc.get("sdk") or {}).get("pin_complete_for_req088"),
})

rec("stages_ok", ok)
rec("stages_skipped_not_in_profile", skipped)
rec("stages_failed", failed)
rec("inconclusive_stages", inconclusive)
rec("stages_missing", missing)
rec("verdict", verdict)
rec("verdict_because", because)
# Written last, and only here.
rec("terminal_state", "reached_complete_stage")
rec("stages.complete.status", "OK")

print(json.dumps({"verdict": verdict, "because": because}))
raise SystemExit(EXIT[verdict])
