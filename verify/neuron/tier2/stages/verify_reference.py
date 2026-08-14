"""Verify owed step 2's staged reference tensors against the sha256 committed in the manifest.

TASK-N08 owed step 2 produced the GPU reference tensors on the desk and retained them; getting
them onto a trn2 host is a transport problem, and transport confers no integrity. So every staged
file is checked against the digest recorded in ``verify/neuron/c5_reference_floor_results.json``,
which is in version control, before anything reads a byte of it as a reference.

The manifest is read the same way ``check_c5_attention_parity.py``'s Neuron arm reads it -- keyed
by basename off ``cells[].retained_tensors`` -- so a file this stage accepts is a file the arm
will also accept, and a divergence between the two shows up here rather than 40 minutes later.

Three cases are failures, not warnings:

* a file present but not in the manifest -- an unknown reference is not a reference;
* a digest mismatch -- the bytes changed in transit or at rest;
* nothing verified at all -- the vacuous run. The arm exits 1 on this too, so a run that gets
  past this stage with an empty directory would only fail later and more expensively.

Usage::

    python3 verify_reference.py <result.json> <staged-dir> <manifest.json>
"""
import hashlib
import json
import os
import subprocess
import sys

RESULT, STAGED, MANIFEST = sys.argv[1], sys.argv[2], sys.argv[3]


# rec.py is base64'd next to this script on the instance at /tmp/t2. Resolving it rather than
# hard-coding that path is what makes these stages runnable at the desk under test_tier2_job.py:
# a stage whose only exercise is on trn2 spot time is a stage nobody exercises.
REC = os.environ.get("LINGBOT_T2_REC") or "/tmp/t2/rec.py"
if not os.path.exists(REC):
    REC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rec.py")


def rec(key, value):
    subprocess.run([sys.executable, REC, RESULT, key,
                    value if isinstance(value, str) else json.dumps(value)], check=False)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest, manifest_why = {}, None
expected_cells = None
try:
    with open(MANIFEST) as fh:
        art = json.load(fh)
    manifest = {os.path.basename(c["retained_tensors"]["path"]): c
                for c in art["cells"] if c.get("retained_tensors")}
    expected_cells = len(manifest)
except Exception as exc:                                # noqa: BLE001
    manifest_why = f"{type(exc).__name__}: {exc}"[:200]

staged = sorted(f for f in os.listdir(STAGED) if f.endswith(".pt")) \
    if os.path.isdir(STAGED) else []

verified, rejected = [], []
for name in staged:
    cell = manifest.get(name)
    got = sha256(os.path.join(STAGED, name))
    if cell and got == cell["retained_tensors"]["sha256"]:
        verified.append({"file": name, "sha256": got,
                         "frames": cell.get("frames"), "seqlen_q": cell.get("seqlen_q")})
    else:
        rejected.append({
            "file": name, "sha256": got,
            "expected": (cell or {}).get("retained_tensors", {}).get("sha256"),
            "why": "not in the manifest" if not cell else "sha256 mismatch"})

ok = bool(verified) and not rejected
rec("reference", {
    "staged_dir": STAGED,
    "manifest": MANIFEST,
    "manifest_unreadable_because": manifest_why,
    "cells_in_manifest": expected_cells,
    "files_verified": verified,
    "files_rejected": rejected,
    "full_geometry_present": (expected_cells is not None
                              and len(verified) == expected_cells and not rejected),
    "note": "a subset is legitimate for a shallow-first run (TASK-N10's ordering); it is "
            "recorded rather than gated here, because the arm's --require-full-geometry is "
            "what decides whether partial coverage is acceptable for a given check",
})
# Rejection is checked FIRST, because it is the more specific cause. A single tampered file leaves
# `verified` empty as well, and reporting that as FAIL_NOTHING_VERIFIED would name the vacuous-run
# case for a run that was not vacuous -- it staged a reference and the bytes were wrong, which is a
# different problem with a different fix.
rec("stages.reference.status", "OK" if ok else (
    "FAIL_REJECTED_FILES" if rejected else "FAIL_NOTHING_VERIFIED"))
print(json.dumps({"verified": len(verified), "rejected": len(rejected), "ok": ok}))
raise SystemExit(0 if ok else 1)
