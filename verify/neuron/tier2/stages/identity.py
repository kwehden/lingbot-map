"""Record what instance this actually is -- the only admissible evidence that one was obtainable.

REQ-087 rejects four kinds of evidence for "a Neuron instance is obtainable": a
``run-instances --dry-run`` result, a service-quota figure, a
``describe-instance-type-offerings`` entry, and a spot-price record. It is not being cautious.
The same dry-run returned ``Request would have succeeded`` for ``trn1.2xlarge`` in an AZ where
``describe-instance-type-offerings`` says that type is not offered at all, and for
``p5.48xlarge`` on a night when H100 was provably unobtainable in all six configured regions.
The only evidence accepted is an actual launch that reached a usable state.

So this file is the *only* writer of the ``capacity`` block, and it runs on the instance. The
launcher cannot populate it, does not try, and a test asserts it never assigns to those keys.

It also derives the REQ-090 category from the instance type IMDS reports rather than from what
the launcher requested, because REQ-090 scores "which category the check falls into" and a
request is not a result -- a spot ``any_of`` list can hand back a different zone or size than
the first preference.

Usage::

    python3 identity.py <result.json>
"""
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

RESULT = sys.argv[1]
IMDS = "http://169.254.169.254/latest"

# Kept in step with REQ-090's named cheap capacity and REQ-081's trn2 requirement. Prefix match,
# because sizes vary and the generation is what the requirements turn on.
CHEAP_PREFIXES = ("inf2.", "trn1.")
KERNEL_CAPABLE_PREFIXES = ("trn2.",)


# rec.py is base64'd next to this script on the instance at /tmp/t2. Resolving it rather than
# hard-coding that path is what makes these stages runnable at the desk under test_tier2_job.py:
# a stage whose only exercise is on trn2 spot time is a stage nobody exercises.
REC = os.environ.get("LINGBOT_T2_REC") or "/tmp/t2/rec.py"
if not os.path.exists(REC):
    REC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rec.py")


def rec(key, value):
    subprocess.run([sys.executable, REC, RESULT, key,
                    value if isinstance(value, str) else json.dumps(value)], check=False)


def imds_token():
    req = urllib.request.Request(f"{IMDS}/api/token", method="PUT",
                                headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"})
    with urllib.request.urlopen(req, timeout=5) as fh:
        return fh.read().decode()


def imds(path, token):
    req = urllib.request.Request(f"{IMDS}/meta-data/{path}",
                                headers={"X-aws-ec2-metadata-token": token})
    try:
        with urllib.request.urlopen(req, timeout=5) as fh:
            return fh.read().decode()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def category_for(instance_type):
    """Which REQ-090 category the capacity this run actually reached belongs to."""
    if instance_type is None:
        return None, "instance type unknown: IMDS did not answer"
    if instance_type.startswith(KERNEL_CAPABLE_PREFIXES):
        return "nki-kernel-capable", (
            "trn2: the only generation on which nkilib.core.attention.attention_cte is "
            "supported, per REQ-081 -- attention_cte dropped trn1/inf2")
    if instance_type.startswith(CHEAP_PREFIXES):
        return "no-nki-kernel-only", (
            "REQ-090's cheap capacity. Admissible only for checks that invoke NO NKI kernel; "
            "REQ-091 leaves open whether the exclusion is attention_cte-specific or blanket, "
            "and until it is resolved REQ-090 applies only to kernel-free checks")
    return "unrecognized", (
        f"{instance_type} is named by neither REQ-090 nor REQ-081; recorded as unrecognized "
        "rather than assumed equivalent to either")


try:
    token = imds_token()
except Exception as exc:                                # noqa: BLE001
    # No IMDS means no evidence. Recorded as absent, never inferred from the fact that this
    # script is running -- a container or a stubbed host would also run it.
    rec("capacity", {
        "obtainable_evidence": None,
        "unavailable_because": f"{type(exc).__name__}: {exc}"[:200],
        "note": "REQ-087: with no instance identity there is no admissible evidence that an "
                "instance was obtained, whatever else this run produced",
    })
    rec("stages.identity.status", "FAIL_NO_IMDS")
    raise SystemExit(1)

fields = {
    "instance_id": "instance-id",
    "instance_type": "instance-type",
    "availability_zone": "placement/availability-zone",
    "region": "placement/region",
    "ami_id": "ami-id",
    "instance_life_cycle": "instance-life-cycle",
}
found, errors = {}, {}
for name, path in fields.items():
    try:
        found[name] = imds(path, token)
    except Exception as exc:                            # noqa: BLE001
        found[name] = None
        errors[name] = f"{type(exc).__name__}: {exc}"[:120]

category, why = category_for(found.get("instance_type"))

rec("capacity", {
    # The positive form on purpose: this key is true because an instance answered IMDS with its
    # own id, which is the one thing none of REQ-087's four rejected evidence classes can do.
    "obtainable_evidence": "actual_launch_reached_usable_state",
    "instance_id": found.get("instance_id"),
    "instance_type": found.get("instance_type"),
    "availability_zone": found.get("availability_zone"),
    "region": found.get("region"),
    "ami_id": found.get("ami_id"),
    "spot": found.get("instance_life_cycle") == "spot",
    "instance_life_cycle": found.get("instance_life_cycle"),
    "imds_errors": errors or None,
    "recorded_by": "stages/identity.py, on the instance",
})
rec("category_reached", {
    "category": category,
    "derived_from": "IMDS instance-type",
    "instance_type": found.get("instance_type"),
    "why": why,
    "note": "REQ-090 scores the category the check actually fell into. The launcher's "
            "--category is recorded separately as category_requested; if the two disagree, the "
            "spot any_of list handed back something other than the first preference and the "
            "requested one is not the answer",
})
rec("stages.identity.status", "OK" if found.get("instance_id") else "FAIL_NO_INSTANCE_ID")
print(json.dumps({"instance_type": found.get("instance_type"), "category": category}))
raise SystemExit(0 if found.get("instance_id") else 1)
