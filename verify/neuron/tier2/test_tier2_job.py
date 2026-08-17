"""Desk-verify the Tier 2 job and runner: every constraint exercised in the direction that FAILS.

TASK-N18 lists nine operational constraints, each of which cost a real trn2 run to learn. A test
that only renders the job and asserts it looks right would pass with every one of those checks
deleted. So each constraint here is exercised NEGATIVELY -- the template is mutated to violate it
and the runner must refuse -- with the passing render kept only as a positive control, so that a
refusal-on-everything bug cannot masquerade as a green run.

The four requirement-level properties are tested the same way:

* **REQ-087** -- the launcher must never be able to claim an instance was obtainable. Tested by
  asserting it writes ``capacity_evidence: null`` and never writes the ``capacity`` block at all,
  and by running ``stages/identity.py`` with IMDS stubbed to fail: no instance identity, no
  evidence, whatever else the run produced.
* **REQ-092** -- an interrupted run is inconclusive-infrastructure-interrupted, never a phase
  failure. Tested through ``classify()`` and through ``stages/verdict.py`` end to end, including the
  case that matters: an interrupted run whose stages contain a genuine FAIL is still INCOMPLETE.
* **REQ-090/REQ-081** -- the two venue combinations that are wrong for a reason rather than for a
  price, plus the profile/category contradictions.
* **REQ-064** -- teardown refuses any cluster name the runner did not generate, and its command
  contains no destructive verb beyond ``sky jobs cancel -n`` on that one job name.

Finally a secret scan over every file in ``tier2/``, because this repository has public remotes.
The scan has an emptiness guard -- a scan that examined nothing passes trivially, which is the
failure mode of every secret scan ever written -- and a positive control that feeds it assembled
identifiers and requires each pattern to fire.

Nothing here launches anything, reaches any network, or spends anything. Run::

    python3 verify/neuron/tier2/test_tier2_job.py
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = os.path.join(HERE, "stages")
sys.path.insert(0, HERE)
import launch_tier2 as L                                   # noqa: E402

FAILS = []
TOTAL = []


def check(name, cond, detail=""):
    ok = bool(cond)
    TOTAL.append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  <- {detail}" if detail and not ok
                                                      else ""))
    if not ok:
        FAILS.append(name)
    return ok


def refuses(name, fn, expect=None):
    """The runner must raise Refused, and for the stated reason.

    Matching on a substring of the message matters: a check that refuses for the wrong reason is
    a check that will stop refusing the moment the unrelated cause is fixed.
    """
    try:
        fn()
    except L.Refused as exc:
        if expect and expect not in str(exc):
            return check(name, False, f"refused, but not for the expected reason: {exc}")
        return check(name, True)
    except Exception as exc:                                # noqa: BLE001
        return check(name, False, f"raised {type(exc).__name__} instead of Refused: {exc}")
    return check(name, False, "did NOT refuse")


# --------------------------------------------------------------------------------------------
# Fixtures. Shape-valid and entirely fictional: no value here names anything real, and the two
# that would otherwise match the secret scan below are ASSEMBLED at runtime so that the committed
# text of this file contains neither a concrete s3 URI nor anything shaped like an account id.
# --------------------------------------------------------------------------------------------
FAKE_BUCKET = "tier2-test-bucket-not-real"
FAKE_REF_URI = "s3" + "://" + FAKE_BUCKET + "/reference"
FAKE_HEAD = "a" * 40

SITE = {
    "s3_bucket": FAKE_BUCKET,
    "s3_prefix": "tier2/desk-test",
    "region": "us-west-1",
    "zones": ["us-west-1a", "us-west-1b"],
    "repo_url": "https://example.invalid/fake/repo.git",
    "reference_uri": FAKE_REF_URI,
    "venv_preference": "nxd_inference",
}

TEMPLATE_SRC = open(L.TEMPLATE).read()


def make_opts(**over):
    o = L.build_parser().parse_args(
        ["--profile", "noop", "--category", "no-nki-kernel", "--instance-type", "inf2.xlarge"])
    for key, val in over.items():
        setattr(o, key, val)
    return o


def render(template=None, profile="noop", site=None, **over):
    o = make_opts(**over)
    return L.render(site or SITE, profile, "20260814T000000Z-aaaaaaa-000001", o.category,
                    FAKE_HEAD, o, template=template)


def mutate(old, new, src=None, count=1):
    src = TEMPLATE_SRC if src is None else src
    if old not in src:
        raise AssertionError(f"the template no longer contains {old!r}; this test mutates text "
                             "that must exist, and a silent no-op mutation would make the "
                             "negative case pass for the wrong reason")
    return src.replace(old, new, count)


# ============================================================================================
print("\n=== positive control: the committed template renders and validates =================")
# First, because every negative case below is only meaningful if the unmutated path is clean.
GOOD = render()
check("the committed template renders clean for the noop profile", bool(GOOD))
check("...and for c5-parity", bool(render(profile="c5-parity", category="nki-kernel",
                                          instance_type="trn2.48xlarge", c5_tolerance=1e-5)))
check("...and for env-probe", bool(render(profile="env-probe")))
check("no doubled-at token survives rendering", not L.ANY_TOKEN.search(GOOD))
check("the rendered job carries the stage scripts as base64", GOOD.count("| base64 -d >") == 7)
for name in ("rec.py", "identity.py", "sdk_versions.py", "verify_reference.py", "verdict.py",
             "get.py", "spot_watch.sh"):
    src = open(os.path.join(STAGES, name), "rb").read()
    check(f"...and the {name} blob decodes to the committed file",
          base64.b64encode(src).decode() in GOOD)
check("the noop profile's job contains no c5 arm invocation",
      "check_c5_attention_parity.py" in GOOD)     # present as a stage, guarded off by the profile
check("the rendered job names the fictional bucket, not a default",
      FAKE_BUCKET in GOOD and "REPLACE" not in GOOD)

print("\n=== constraint 1: embedded scripts travel as base64, never as a heredoc or -c ======")
refuses("a heredoc in the job is refused",
        lambda: render(template=mutate("  rec() {", "  python3 - <<'PYEOF'\nprint(1)\nPYEOF\n"
                                                    "  rec() {")),
        expect="heredoc")
refuses("an inline `python3 -c` is refused",
        lambda: render(template=mutate('  rec() {', '  python3 -c "import os"\n  rec() {')),
        expect="python -c")
refuses("a base64 token naming a file that does not exist is refused",
        lambda: render(template=mutate("@@B64:stages/get.py@@", "@@B64:stages/nonexistent.py@@")),
        expect="does not exist")
refuses("an unsubstituted token left in the job is refused",
        lambda: render(template=mutate("name: @@RUN_NAME@@", "name: @@RUN_NAME@@ @@UNKNOWN@@")),
        expect="unsubstituted")

print("\n=== constraint 2: no foreground sleep past 115 s ==================================")
refuses("a foreground `sleep 116` is refused",
        lambda: render(template=mutate("  rec() {", "  sleep 116\n  rec() {")),
        expect="115 s cap")
check("a foreground `sleep 115` is accepted (the cap is inclusive, and 115 is the documented one)",
      bool(render(template=mutate("  rec() {", "  sleep 115\n  rec() {"))))
check("a long sleep inside a properly detached watcher is accepted",
      bool(render(template=mutate(
          "  rec() {",
          "  setsid nohup sh -c 'sleep 7800; true' > /tmp/w.out 2>&1 < /dev/null &\n  rec() {"))))
check("a long `timeout` on a foreground stage is NOT flagged (the cap is about sleep in one "
      "command, not about a stage's wall clock)", "timeout 3000" in GOOD)

print("\n=== constraint 3: watchers use setsid nohup ... < /dev/null, all three ============")
WATCH = ('setsid nohup /tmp/t2/spot_watch.sh "$JSON" "$S3" "$LOG" "$START_TS" \\\n'
         "    > /tmp/t2/spot_watch.out 2>&1 < /dev/null &")
check("the template's watcher launch is the text this section mutates", WATCH in TEMPLATE_SRC)
refuses("a watcher without setsid is refused",
        lambda: render(template=mutate(WATCH, WATCH.replace("setsid ", ""))),
        expect="'setsid'")
refuses("a watcher without nohup is refused",
        lambda: render(template=mutate(WATCH, WATCH.replace("nohup ", ""))),
        expect="'nohup'")
refuses("a watcher without `< /dev/null` is refused",
        lambda: render(template=mutate(WATCH, WATCH.replace(" < /dev/null", ""))),
        expect="'< /dev/null'")
check("the check reads across the backslash continuation the real watcher uses",
      # If it did not, the committed template would have failed the positive control above with
      # `< /dev/null` reported missing from the line that has it on its second half.
      len(L.logical_lines(WATCH)) == 1)

print("\n=== constraint 5: activate the venv; never invoke its interpreter =================")
refuses("`$VENV/bin/python` anywhere in the job is refused",
        lambda: render(template=mutate('  rec() {', '  $VENV/bin/python -V\n  rec() {')),
        expect="PJRT plugin path")
refuses("a hard-coded venv interpreter path is refused too",
        lambda: render(template=mutate(
            '  rec() {', '  /opt/aws_neuron_venv_pytorch/bin/python -V\n  rec() {')),
        expect="PJRT plugin path")
refuses("a job with no `activate` at all is refused",
        lambda: render(template=mutate('. "$VENV/bin/activate"', "true")),
        expect="constraint 5")

print("\n=== constraint 6: both Neuron environment variables ===============================")
refuses("removing PJRT_DEVICE is refused",
        lambda: render(template=TEMPLATE_SRC.replace("export PJRT_DEVICE=NEURON", "true")),
        expect="PJRT_DEVICE=NEURON")
refuses("removing NEURON_PLATFORM_TARGET_OVERRIDE is refused",
        lambda: render(template=TEMPLATE_SRC.replace(
            "export NEURON_PLATFORM_TARGET_OVERRIDE=trn2", "true")),
        expect="NEURON_PLATFORM_TARGET_OVERRIDE=trn2")

print("\n=== constraint 7: every profile records the version pin (REQ-088) =================")
L.PROFILES["_test_no_sdk"] = {"stages": ("identity", "noop", "complete"),
                              "invokes_nki_kernel": False, "purpose": "test fixture"}
refuses("a profile without the sdk_versions stage is refused",
        lambda: render(profile="_test_no_sdk"),
        expect="REQ-088")
L.PROFILES["_test_bad_stage"] = {"stages": ("identity", "sdk_versions", "not_a_stage"),
                                 "invokes_nki_kernel": False, "purpose": "test fixture"}
refuses("a profile naming a stage the job does not implement is refused",
        lambda: render(profile="_test_bad_stage"),
        expect="KNOWN_STAGES")

print("\n=== constraint 8: one flush per stage, counted =====================================")
refuses("deleting a flush is refused",
        lambda: render(template=mutate("\n  flush noop\n", "\n")),
        expect="constraint 8")
refuses("adding a stage guard with no flush is refused",
        lambda: render(template=mutate("  if stage noop; then",
                                       "  if stage devices2; then\n    true\n  fi\n"
                                       "  if stage noop; then")),
        expect="constraint 8")
GUARDS = re.findall(r"^\s*if stage (\w+); then\s*$", GOOD, re.M)
check("the job implements every known stage, so a profile change needs no template change",
      sorted(GUARDS) == sorted(L.KNOWN_STAGES), str(sorted(GUARDS)))
check("a stage the profile skipped still flushes (the guard is outside the flush)",
      GOOD.count("\n  flush ") == len(L.KNOWN_STAGES), str(GOOD.count("\n  flush ")))

print("\n=== constraint 9 / REQ-090 / REQ-081: the venue is parameterized, and two "
      "combinations are refused ===")
refuses("a kernel check on REQ-090's cheap capacity is refused",
        lambda: L.check_category("inf2.xlarge", "nki-kernel", "c5-parity", False),
        expect="attention_cte dropped")
refuses("...on trn1 too",
        lambda: L.check_category("trn1.2xlarge", "nki-kernel", "c5-parity", False),
        expect="attention_cte dropped")
refuses("a kernel-free check on trn2 is refused without the explicit override",
        lambda: L.check_category("trn2.48xlarge", "no-nki-kernel", "noop", False),
        expect="--allow-expensive-venue")
ALLOWED = L.check_category("trn2.48xlarge", "no-nki-kernel", "noop", True)
check("...and permitted with it, recorded as an override rather than silently",
      ALLOWED["expensive_venue_override"] is True)
refuses("an unclassified instance type is refused",
        lambda: L.check_category("m5.large", "no-nki-kernel", "noop", False),
        expect="neither REQ-090's cheap capacity")
refuses("a category REQ-090 does not name is refused",
        lambda: L.check_category("inf2.xlarge", "cheap-ish", "noop", False),
        expect="REQ-090 recognizes")
refuses("a category contradicting a kernel profile is refused",
        lambda: L.check_category("trn2.48xlarge", "no-nki-kernel", "c5-parity", True),
        expect="false on its face")
refuses("a category overstating a kernel-free profile is refused",
        lambda: L.check_category("inf2.xlarge", "nki-kernel", "noop", False),
        expect="overstates")
OK_CAT = L.check_category("inf2.xlarge", "no-nki-kernel", "noop", False)
check("a correct pairing is accepted", OK_CAT["instance_type_class"] == "cheap")
check("...and records that this is the REQUEST, not the category reached",
      "REQUEST" in OK_CAT["note"] and "IMDS" in OK_CAT["note"], OK_CAT["note"])
check("the template hard-codes no instance type: the category travels as a token",
      "@@INSTANCE_TYPE@@" in TEMPLATE_SRC and "@@CATEGORY@@" in TEMPLATE_SRC)

print("\n=== constraint 4: git ls-remote must match local HEAD =============================")


class FakeGit:
    """Stands in for subprocess. Returns whatever the case under test needs, per subcommand."""

    def __init__(self, head, remote, rc=0, dirty=False):
        self.head, self.remote, self.rc, self.dirty = head, remote, rc, dirty

    def __call__(self, args):
        out, rc, err = "", 0, ""
        if args[0] == "rev-parse":
            out = self.head
        elif args[0] == "status":
            out = " M verify/neuron/tier2/launch_tier2.py\n" if self.dirty else ""
        elif args[0] == "ls-remote":
            out, rc = self.remote, self.rc
            err = "fatal: could not read from remote repository" if rc else ""
        return argparse.Namespace(returncode=rc, stdout=out, stderr=err)


MATCH = f"{FAKE_HEAD}\trefs/heads/neuron-port\n"
OKREV = L.check_revision("/tmp", "https://example.invalid/r.git", "neuron-port",
                         git=FakeGit(FAKE_HEAD, MATCH))
check("a published HEAD equal to local HEAD passes", OKREV["published_head"] == FAKE_HEAD)
refuses("a published HEAD different from local HEAD is refused",
        lambda: L.check_revision("/tmp", "https://example.invalid/r.git", "neuron-port",
                                 git=FakeGit(FAKE_HEAD, "b" * 40 + "\trefs/heads/neuron-port\n")),
        expect="clone code nobody can match")
refuses("an ls-remote FAILURE is refused, not skipped",
        lambda: L.check_revision("/tmp", "https://example.invalid/r.git", "neuron-port",
                                 git=FakeGit(FAKE_HEAD, "", rc=128)),
        expect="unreachable remote")
refuses("a remote with no such branch is refused",
        lambda: L.check_revision("/tmp", "https://example.invalid/r.git", "neuron-port",
                                 git=FakeGit(FAKE_HEAD, "")),
        expect="TASK-N00's publish step has not run")
refuses("a dirty tree is refused even when HEAD matches",
        lambda: L.check_revision("/tmp", "https://example.invalid/r.git", "neuron-port",
                                 git=FakeGit(FAKE_HEAD, MATCH, dirty=True)),
        expect="uncommitted changes")
check("the job re-checks the cloned sha on the instance, because an instance had to be acquired "
      "in between", "FAIL_HEAD_MISMATCH" in TEMPLATE_SRC)

print("\n=== REQ-087: the launcher cannot claim capacity; only the instance can ============")
LSRC = open(os.path.join(HERE, "launch_tier2.py")).read()
check("the launcher writes capacity_evidence as null", '"capacity_evidence": None' in LSRC)
ASSIGNED = set(re.findall(r'"capacity_evidence":\s*([A-Za-z0-9_"\']+)', LSRC))
check("...and assigns no other value to it anywhere", ASSIGNED == {"None"}, str(ASSIGNED))
check("the launcher never writes the capacity block at all", 'rec("capacity"' not in LSRC)
ISRC = open(os.path.join(STAGES, "identity.py")).read()
OTHER_STAGES = {f: open(os.path.join(STAGES, f)).read()
                for f in sorted(os.listdir(STAGES))
                if f != "identity.py" and os.path.isfile(os.path.join(STAGES, f))
                and f.endswith((".py", ".sh"))}
check("no file other than stages/identity.py writes the capacity block",
      not [f for f, s in OTHER_STAGES.items() if 'rec("capacity"' in s],
      str([f for f, s in OTHER_STAGES.items() if 'rec("capacity"' in s]))
check("identity.py's two writes are the IMDS-failed and IMDS-answered branches, and nothing else",
      ISRC.count('rec("capacity"') == 2, str(ISRC.count('rec("capacity"')))
check("...and it is conditioned on IMDS answering with an instance id",
      'rec("stages.identity.status", "OK" if found.get("instance_id")' in ISRC)
check("...and derives the category from IMDS, not from the launcher's request",
      '"derived_from": "IMDS instance-type"' in ISRC)
check("the two prefix lists agree with the launcher's",
      L.CHEAP_PREFIXES == ("inf2.", "trn1.") and L.KERNEL_CAPABLE_PREFIXES == ("trn2.",)
      and 'CHEAP_PREFIXES = ("inf2.", "trn1.")' in ISRC
      and 'KERNEL_CAPABLE_PREFIXES = ("trn2.",)' in ISRC)

# Behavioural half: run identity.py with IMDS stubbed to fail. This is the case that matters --
# on a host that cannot prove it is an instance, REQ-087's answer must be "no evidence", not
# "well, something ran". Stubbed rather than reaching the real IMDS both because the desk may
# itself be EC2 (which would make the test's outcome depend on where it runs) and because a real
# answer would put this desk's instance id in the test output.
_orig_urlopen = urllib.request.urlopen


def _no_imds(*_a, **_k):
    raise OSError("IMDS unreachable (stubbed by test_tier2_job.py)")


urllib.request.urlopen = _no_imds
IDENT = None
with tempfile.TemporaryDirectory() as tmp:
    res = os.path.join(tmp, "r.json")
    os.environ["LINGBOT_T2_REC"] = os.path.join(STAGES, "rec.py")
    argv, sys.argv = sys.argv, ["identity.py", res]
    rc = None
    try:
        spec = importlib.util.spec_from_file_location("t2_identity",
                                                      os.path.join(STAGES, "identity.py"))
        IDENT = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(IDENT)
    except SystemExit as exc:
        rc = exc.code
    finally:
        sys.argv = argv
        urllib.request.urlopen = _orig_urlopen
    doc = json.load(open(res)) if os.path.exists(res) else {}
check("with no IMDS, identity.py exits non-zero", rc == 1, f"rc={rc}")
check("...and records NO obtainable evidence rather than inferring it from having run",
      doc.get("capacity", {}).get("obtainable_evidence") is None)
check("...and says why", "unavailable_because" in doc.get("capacity", {}))
check("...and fails its stage as FAIL_NO_IMDS",
      doc.get("stages", {}).get("identity", {}).get("status") == "FAIL_NO_IMDS")
check("category_for maps trn2 to the kernel-capable category",
      IDENT and IDENT.category_for("trn2.48xlarge")[0] == "nki-kernel-capable")
check("...inf2/trn1 to the kernel-free-only category",
      IDENT and IDENT.category_for("inf2.xlarge")[0] == "no-nki-kernel-only"
      and IDENT.category_for("trn1.2xlarge")[0] == "no-nki-kernel-only")
check("...an unknown type to 'unrecognized' rather than to either",
      IDENT and IDENT.category_for("m5.large")[0] == "unrecognized")
check("...and an unknown instance type to no category at all",
      IDENT and IDENT.category_for(None)[0] is None)

print("\n=== REQ-092: an interrupted run is INCOMPLETE, never a phase failure ==============")
FULL = {"terminal_state": "reached_complete_stage", "verdict": "PASS", "_elapsed_s": 100}
check("a completed passing run classifies PASS", L.classify(dict(FULL))["verdict"] == "PASS")
check("a completed failing run classifies FAIL",
      L.classify(dict(FULL, verdict="FAIL"))["verdict"] == "FAIL")
CUT = L.classify({"_elapsed_s": 42, "_last_stage_flushed": "venv"})
check("a run with no terminal_state classifies INCOMPLETE", CUT["verdict"] == "INCOMPLETE")
check("...and reports how far it got, so a rerun knows where to start",
      CUT["last_stage_flushed"] == "venv")
INT = L.classify({"_elapsed_s": 42, "verdict": "FAIL", "stages": {"c5_arm": {"status": "FAIL_X"}}},
                 {"cause": "spot_preemption", "elapsed_s": 42})
check("an interrupted run is INCOMPLETE even with a FAIL among its stages",
      INT["verdict"] == "INCOMPLETE", str(INT))
check("...and records the cause and the elapsed time REQ-092 requires",
      INT["cause"] == "spot_preemption" and INT["elapsed_s"] == 42)
check("no artifact at all classifies INCOMPLETE, not FAIL",
      L.classify({})["verdict"] == "INCOMPLETE")
REB = L.classify(dict(FULL), {"rebalance_recommended_utc": "2026-08-14T00:00:00Z"})
check("a rebalance recommendation alone does not make a finished run interrupted",
      REB["verdict"] == "PASS", str(REB))

print("\n=== stages/verdict.py end to end ==================================================")


def run_verdict(stages, selected, sidecar=None):
    tmp = tempfile.mkdtemp()
    res = os.path.join(tmp, "r.json")
    with open(res, "w") as fh:
        json.dump({"stages": stages, "stages_selected": ",".join(selected)}, fh)
    side = os.path.join(tmp, "interrupt.json")
    if sidecar is not None:
        with open(side, "w") as fh:
            json.dump(sidecar, fh)
    env = dict(os.environ, LINGBOT_T2_REC=os.path.join(STAGES, "rec.py"),
               LINGBOT_T2_SIDECAR=side)
    proc = subprocess.run([sys.executable, os.path.join(STAGES, "verdict.py"), res],
                          capture_output=True, text=True, env=env, timeout=120)
    return proc.returncode, json.load(open(res))


ALL_OK = {"identity": {"status": "OK"}, "devices": {"status": "RECORDED"},
          "venv": {"status": "ACTIVATED"}, "sdk_versions": {"status": "OK"},
          "noop": {"status": "OK"}}
SEL = ["identity", "devices", "venv", "sdk_versions", "noop", "complete"]

rc, doc = run_verdict(ALL_OK, SEL)
check("all stages passing -> PASS, exit 0", rc == 0 and doc["verdict"] == "PASS", str(rc))
check("...and terminal_state is written exactly once, only here",
      doc.get("terminal_state") == "reached_complete_stage")

rc, doc = run_verdict(dict(ALL_OK, noop={"status": "FAIL_SOMETHING"}), SEL)
check("a failed stage -> FAIL, exit 1", rc == 1 and doc["verdict"] == "FAIL", str(rc))

rc, doc = run_verdict(dict(ALL_OK, c5_arm={"status": "INCONCLUSIVE_COMPILE_TIMEOUT"}),
                      SEL + ["c5_arm"])
check("a compile timeout -> INCONCLUSIVE, exit 2, and never FAIL (NKI-1590)",
      rc == 2 and doc["verdict"] == "INCONCLUSIVE", str(rc))

rc, doc = run_verdict(dict(ALL_OK, noop={"status": "FAIL_SOMETHING"},
                           c5_arm={"status": "INCONCLUSIVE_COMPILE_TIMEOUT"}), SEL + ["c5_arm"])
check("a timeout beside an unrelated failure keeps its own attribution",
      rc == 1 and any(s["stage"] == "c5_arm" for s in doc["inconclusive_stages"]),
      str(doc.get("inconclusive_stages")))

rc, doc = run_verdict(ALL_OK, SEL + ["repo"])
check("a selected stage that left no record -> INCOMPLETE, exit 3, not a pass",
      rc == 3 and doc["verdict"] == "INCOMPLETE" and doc["stages_missing"] == ["repo"], str(rc))

rc, doc = run_verdict(dict(ALL_OK, noop={"status": "FAIL_SOMETHING"}), SEL,
                      sidecar={"cause": "spot_preemption", "elapsed_s": 55})
check("the watcher's sidecar wins over a FAIL: interrupted -> INCOMPLETE, exit 3 (REQ-092)",
      rc == 3 and doc["verdict"] == "INCOMPLETE", str(rc) + " " + str(doc.get("verdict_because")))
check("...and the sidecar is merged into the artifact rather than left in a second file",
      doc.get("_interrupted", {}).get("cause") == "spot_preemption")

rc, doc = run_verdict(ALL_OK, SEL, sidecar={"rebalance_recommended_utc": "2026-08-14T00:00:00Z"})
check("a sidecar with no cause does not make a finished run interrupted",
      rc == 0 and doc["verdict"] == "PASS", str(rc))
check("...but the recommendation is still on the record",
      "rebalance_recommended_utc" in (doc.get("_interrupt_sidecar") or {}))

rc, doc = run_verdict(dict(ALL_OK, c5_arm={"status": "FAIL_EXIT_1"}), SEL + ["c5_arm"])
check("a failing arm marks REQ-089 attribution as owed, not as a design finding",
      doc["attribution"]["pending_req089"] is True and "NKI" not in str(rc))
check("...and names what has to be checked before it becomes one",
      "attention_cte defects" in (doc["attribution"]["why"] or ""))
rc, doc = run_verdict(dict(ALL_OK, c5_arm={"status": "OK"}), SEL + ["c5_arm"])
check("a passing arm owes no attribution", doc["attribution"]["pending_req089"] is False)

print("\n=== stages/verify_reference.py: transport confers no integrity ====================")


def run_verify_reference(files, manifest_cells):
    tmp = tempfile.mkdtemp()
    staged = os.path.join(tmp, "staged")
    os.makedirs(staged)
    for name, body in files.items():
        with open(os.path.join(staged, name), "wb") as fh:
            fh.write(body)
    man = os.path.join(tmp, "manifest.json")
    with open(man, "w") as fh:
        json.dump({"cells": manifest_cells}, fh)
    res = os.path.join(tmp, "r.json")
    env = dict(os.environ, LINGBOT_T2_REC=os.path.join(STAGES, "rec.py"))
    proc = subprocess.run([sys.executable, os.path.join(STAGES, "verify_reference.py"),
                           res, staged, man], capture_output=True, text=True, env=env, timeout=120)
    return proc.returncode, (json.load(open(res)) if os.path.exists(res) else {})


BODY = b"pretend this is a retained tensor blob"
DIGEST = hashlib.sha256(BODY).hexdigest()
CELL = {"frames": 2, "seqlen_q": 4,
        "retained_tensors": {"path": "/somewhere/cell_f2.pt", "sha256": DIGEST}}

rc, doc = run_verify_reference({"cell_f2.pt": BODY}, [CELL])
check("a staged file matching the committed digest verifies",
      rc == 0 and doc["stages"]["reference"]["status"] == "OK", str(rc))
rc, doc = run_verify_reference({"cell_f2.pt": BODY + b"!"}, [CELL])
check("a tampered file is REJECTED, not warned about",
      rc == 1 and doc["stages"]["reference"]["status"] == "FAIL_REJECTED_FILES", str(rc))
check("...and the reason is the digest, not absence",
      doc["reference"]["files_rejected"][0]["why"] == "sha256 mismatch")
rc, doc = run_verify_reference({"stranger.pt": BODY}, [CELL])
check("a file absent from the manifest is rejected: an unknown reference is not a reference",
      rc == 1 and doc["reference"]["files_rejected"][0]["why"] == "not in the manifest")
rc, doc = run_verify_reference({}, [CELL])
check("an empty staged directory FAILS rather than vacuously passing",
      rc == 1 and doc["stages"]["reference"]["status"] == "FAIL_NOTHING_VERIFIED", str(rc))
DEEPER = {"frames": 4, "seqlen_q": 8,
          "retained_tensors": {"path": "/somewhere/cell_f4.pt", "sha256": "0" * 64}}
rc, doc = run_verify_reference({"cell_f2.pt": BODY}, [CELL, DEEPER])
check("a partial subset verifies but is recorded as partial (shallow-first is TASK-N10's order)",
      rc == 0 and doc["reference"]["full_geometry_present"] is False,
      f"rc={rc} {doc.get('reference', {}).get('full_geometry_present')}")
rc, doc = run_verify_reference({"cell_f2.pt": BODY}, [CELL])
check("...and full coverage is recorded as full",
      doc["reference"]["full_geometry_present"] is True)
check("the arm's own manifest accessor is reused verbatim, so acceptance here means acceptance "
      "there",
      'for c in art["cells"] if c.get("retained_tensors")'
      in open(os.path.join(STAGES, "verify_reference.py")).read()
      and 'for c in art["cells"] if c.get("retained_tensors")'
      in open(os.path.join(HERE, "..", "check_c5_attention_parity.py")).read())

print("\n=== the arm refuses to score without a tolerance or a verified reference ==========")
check("the job skips the arm when the reference did not verify",
      "SKIPPED_REFERENCE_NOT_VERIFIED" in TEMPLATE_SRC)
check("the job fails the arm rather than defaulting a tolerance",
      "FAIL_NO_TOLERANCE_SUPPLIED" in TEMPLATE_SRC)
check("no tolerance literal is hard-coded anywhere in the job or the runner",
      not re.search(r"tolerance[^\n]*=\s*[0-9]", TEMPLATE_SRC)
      and "--c5-tolerance" in LSRC and 'default=None' in LSRC)
check("the runner's --c5-tolerance has no default", make_opts().c5_tolerance is None)

print("\n=== stages/sdk_versions.py: REQ-088's pin, and absences typed rather than fatal ===")
# The desk has no Neuron stack, so this run exercises the branch that matters: every probe fails,
# and the stage must still finish and still type each absence. TASK-N19's REQ-095 gate turns on
# exactly one such absence, and a stage that died on the first ImportError would deliver nothing
# for it to read.
with tempfile.TemporaryDirectory() as tmp:
    res = os.path.join(tmp, "r.json")
    env = dict(os.environ, LINGBOT_T2_REC=os.path.join(STAGES, "rec.py"))
    proc = subprocess.run([sys.executable, os.path.join(STAGES, "sdk_versions.py"), res],
                          capture_output=True, text=True, env=env, timeout=300)
    doc = json.load(open(res)) if os.path.exists(res) else {}
    sdk = doc.get("sdk", {})
    check("the version stage completes on a host with no Neuron stack at all",
          proc.returncode in (0, 1) and bool(sdk), proc.stderr[-300:])
    mods = sdk.get("modules", {})
    check("...and types each missing module rather than dying on the first ImportError",
          mods.get("torch_neuronx", {}).get("importable") is False, str(mods.get("torch_neuronx")))
    check("...and records WHY it was missing, which is what a gate can read",
          bool(mods.get("torch_neuronx", {}).get("why")))
    check("...and probes the exact module TASK-N19's REQ-095 gate needs, separately",
          sdk.get("nxd_kvcache_manager_gate", {}).get("module")
          == "neuronx_distributed_inference.modules.kvcache.kv_cache_manager")
    check("...and says what REQ-095 still needs beyond this probe",
          "install ATTEMPT" in (sdk.get("nxd_kvcache_manager_gate", {}).get("note") or ""))
    check("...and probes attention_cte specifically, not just nkilib",
          "nkilib.core.attention" in mods)
    check("an incomplete pin is recorded as incomplete, not assumed complete (REQ-088)",
          sdk.get("pin_complete_for_req088") is False)
    check("...and refuses a tolerance derived from it, in words the reader cannot miss",
          "no tolerance may be derived or reused" in (sdk.get("pin_incomplete_because") or ""))
    check("...and the stage says so in its status",
          doc.get("stages", {}).get("sdk_versions", {}).get("status")
          == "RECORDED_INCOMPLETE_PIN")
    check("the pin covers all four items REQ-088 names",
          all(k in sdk for k in ("dlami_image_id", "neuronx_cc", "modules",
                                 "neuron_driver_dpkg")),
          str(sorted(sdk)))

print("\n=== stages/rec.py and get.py =====================================================")
with tempfile.TemporaryDirectory() as tmp:
    res = os.path.join(tmp, "r.json")
    for key, val in (("stages.a.status", "OK"), ("stages.b.status", "FAIL_X"),
                     ("top", '{"n": 3}'), ("num", "7")):
        subprocess.run([sys.executable, os.path.join(STAGES, "rec.py"), res, key, val], check=True)
    doc = json.load(open(res))
    check("dotted keys nest instead of colliding in one namespace",
          doc["stages"]["a"]["status"] == "OK" and doc["stages"]["b"]["status"] == "FAIL_X")
    check("JSON-typed values stay structured", doc["top"] == {"n": 3} and doc["num"] == 7)
    got = subprocess.run([sys.executable, os.path.join(STAGES, "get.py"), res,
                          "stages.b.status"], capture_output=True, text=True)
    check("get.py reads a dotted path", got.stdout.strip() == "FAIL_X", got.stdout)
    got = subprocess.run([sys.executable, os.path.join(STAGES, "get.py"), res, "stages.zz.status"],
                         capture_output=True, text=True)
    check("get.py prints empty for an absent path, so the shell's test is a comparison and not "
          "an error", got.stdout.strip() == "", got.stdout)

print("\n=== REQ-064: teardown touches only what this runner created =======================")
check("a cluster name this runner generated is eligible",
      L.teardown_guard("lingbot-t2-noop-20260814t000000z-aaaaaaa-000001"))
for bad in ("prod-inference-cluster", "sky-controller", "", "lingbot", "lingbot-t2-",
            "trn2-cluster-1"):
    refuses(f"refuses to tear down {bad!r}", lambda b=bad: L.teardown_guard(b),
            expect="REQ-064")
DOWN = L.ssm_commands("lingbot-t2-noop-x123456", "", make_opts())["down"]
for verb in ("terminate-instances", "delete-volume", "delete-network-interface", "aws ec2 delete",
             "--force", "rm -rf"):
    check(f"the teardown command contains no {verb!r}", verb not in DOWN)
check("the teardown command names exactly one job",
      DOWN.count("lingbot-t2-noop-x123456") == 2 and "sky jobs cancel -n" in DOWN)
check("orphaned volumes are reported, not deleted", "do not delete them" in LSRC)
check("teardown cancels a named managed job rather than downing a cluster, because a cluster name "
      "can be reused and a job name cannot", "sky down" not in DOWN)

print("\n=== the launcher's own refusals ===================================================")
with tempfile.TemporaryDirectory() as tmp:
    cfg = os.path.join(tmp, "site.json")
    with open(cfg, "w") as fh:
        json.dump(SITE, fh)
    refuses("--skip-revision-check with --launch is refused",
            lambda: L.main(["--profile", "noop", "--category", "no-nki-kernel",
                            "--instance-type", "inf2.xlarge", "--site-config", cfg,
                            "--skip-revision-check", "--launch"]),
            expect="unidentifiable revision")
    refuses("a site config with a placeholder left in it is refused",
            lambda: L.load_site(os.path.join(HERE, "site_config.example.json")),
            expect="example placeholder")
    for key, bad in (("s3_bucket", "Not A Bucket"), ("region", "nowhere"),
                     ("subnet_id", "subnet-nothex"), ("controller_instance_id", "i-nope"),
                     ("zones", ["not-a-zone"])):
        broken = os.path.join(tmp, f"bad_{key}.json")
        with open(broken, "w") as fh:
            json.dump(dict(SITE, **{key: bad}), fh)
        refuses(f"a malformed {key} is refused", lambda p=broken: L.load_site(p))
    for key in ("s3_bucket", "s3_prefix", "region", "zones"):
        missing = os.path.join(tmp, f"missing_{key}.json")
        stripped = {k: v for k, v in SITE.items() if k != key}
        with open(missing, "w") as fh:
            json.dump(stripped, fh)
        refuses(f"a missing required {key} is refused", lambda p=missing: L.load_site(p),
                expect="is missing")

    # A full --render pass, which is also the REQ-087 assertion that the launcher writes no
    # capacity evidence of its own.
    out = os.path.join(tmp, "render")
    rc = L.main(["--profile", "noop", "--category", "no-nki-kernel",
                 "--instance-type", "inf2.xlarge", "--site-config", cfg,
                 "--skip-revision-check", "--out-dir", out])
    man = json.load(open(os.path.join(out, "manifest.json")))
    check("--render exits 0 and writes a manifest", rc == 0 and bool(man))
    check("--render does not launch", man["launched"] is False)
    check("the manifest's capacity_evidence is null (REQ-087)",
          man["capacity_evidence"] is None)
    check("...and says what would establish it instead",
          "actual launch that reached a usable state" in man["capacity_evidence_note"])
    check("the manifest records the rendered job's digest, so the artifact is traceable to text",
          len(man["rendered_sha256"]) == 64)
    check("the manifest records the venue REQUEST separately from anything measured",
          man["category"]["requested"] == "no-nki-kernel")
    check("a skipped revision check is recorded as skipped, not as passed",
          man["revision"]["skipped"] is True)
    check("the SSM launch body is emitted for a human to run", "sky jobs launch -n" in
          man["next"]["ssm_command_launch"])
    check("...detached, because acquiring capacity outlasts a single SSM command",
          all(t in man["next"]["ssm_command_launch"] for t in ("setsid", "nohup", "< /dev/null")))
    # The controller carries an admin policy that raises on CLUSTER_LAUNCH, so `sky launch` is
    # refused there before any capacity is requested. This assertion is the whole reason the first
    # version of this runner could not have worked, and it is cheap to keep.
    check("the launch is a managed job, never a direct cluster launch: the controller's admin "
          "policy refuses CLUSTER_LAUNCH outright",
          "sky launch" not in man["next"]["ssm_command_launch"])
    check("the poll reads the managed-job queue, not cluster status",
          "sky jobs queue" in man["next"]["ssm_command_poll"]
          and "sky status" not in man["next"]["ssm_command_poll"])

print("\n=== the controller is a fact to check, not a field to trust =======================")


class _Proc:
    def __init__(self, rc=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = rc, stdout, stderr


_ONLINE = json.dumps({"InstanceInformationList": [{"PingStatus": "Online",
                                                   "AgentVersion": "3.3.4851.0"}]})
check("an Online controller passes and its ping status is returned for the manifest",
      L.check_controller("i-fixture", "us-east-1",
                         runner=lambda cmd: _Proc(0, _ONLINE))["ping_status"] == "Online")
refuses("a controller id SSM does not know is refused, naming BOTH candidate causes",
        lambda: L.check_controller("i-fixture", "us-east-2",
                                   runner=lambda cmd: _Proc(0, '{"InstanceInformationList": []}')),
        expect="controller_region")
refuses("a controller that is ConnectionLost is refused, because send-command would be accepted "
        "and never delivered",
        lambda: L.check_controller(
            "i-fixture", "us-east-1",
            runner=lambda cmd: _Proc(0, json.dumps(
                {"InstanceInformationList": [{"PingStatus": "ConnectionLost"}]}))),
        expect="not Online")
# The bug this guards: send-command's --region was the JOB's region, while its target is the
# controller. A controller in us-east-1 launching a trn2 job into us-east-2 is the normal case,
# not an exotic one, and it failed with an InvalidInstanceId that named neither field.
_seen = []
L.check_controller("i-fixture", "us-west-2",
                   runner=lambda cmd: (_seen.append(cmd), _Proc(0, _ONLINE))[1])
check("the check asks the region it was given, not a region it inferred",
      "us-west-2" in _seen[0] and "--region" in _seen[0], _seen[0])

print("\n=== run ids ======================================================================")
RID = L.make_run_id("noop", FAKE_HEAD, now=1_776_000_000)
check("a run id carries the revision that produced it (Phase 0 recorded neither -- finding #40)",
      FAKE_HEAD[:7] in RID and RID.startswith("2026"), RID)
check("two profiles at the same instant get different ids",
      L.make_run_id("noop", FAKE_HEAD, now=1_776_000_000)
      != L.make_run_id("c5-parity", FAKE_HEAD, now=1_776_000_000))
check("the same profile, instant and revision is reproducible",
      L.make_run_id("noop", FAKE_HEAD, now=1_776_000_000) == RID)

print("\n=== secret scan over every committed file in tier2/ ===============================")
# This repository has public remotes. Everything account-, VPC-, bucket- or instance-shaped has to
# arrive from --site-config at runtime.
SCANNED, TOTAL_BYTES, HITS = [], 0, []
for root, dirs, names in os.walk(HERE):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for name in sorted(names):
        if name.endswith((".pyc", ".json.gz")):
            continue
        path = os.path.join(root, name)
        body = open(path, encoding="utf-8", errors="replace").read()
        SCANNED.append(os.path.relpath(path, HERE))
        TOTAL_BYTES += len(body)
        clean = L.scrub_allowed(body)
        for pattern, desc in L.FORBIDDEN_IN_REPO:
            for match in pattern.finditer(clean):
                HITS.append(f"{os.path.relpath(path, HERE)}: {desc} -- {match.group(0)[:32]!r}")

# The emptiness guard. A scan that examined nothing passes, which is how a secret scan fails
# without anyone noticing; the repo-level scans on this project were extended for exactly this
# reason. ABORT rather than FAIL: a scan that did not run is not a result to be weighed.
if len(SCANNED) < 8 or TOTAL_BYTES < 20_000:
    print(f"\nABORT: the secret scan examined {len(SCANNED)} files / {TOTAL_BYTES} bytes, which "
          "is too little to be a scan of this directory. Not reported as a pass.")
    raise SystemExit(3)
check(f"the scan examined the directory ({len(SCANNED)} files, {TOTAL_BYTES} bytes)", True)
check("no committed tier2 file contains an account-, VPC-, bucket- or instance-shaped identifier",
      not HITS, "\n        " + "\n        ".join(HITS))

# Positive control: the patterns must actually fire. Assembled at runtime so that the committed
# text of this file contains none of them -- otherwise this control would trip the scan above.
CONTROL = " ".join([
    "acct=" + "9876" + "5432" + "1098",
    "subnet-" + "0" * 17,
    "rtb-" + "0abcdef12",
    "i-" + "0abcdef12",
    "vpc-" + "0abcdef12",
    "s3" + "://" + "some-real-bucket/key",
    "arn" + ":aws:iam::role/x",
    "10.100" + ".3.0/24",
])
FIRED = {desc for pattern, desc in L.FORBIDDEN_IN_REPO if pattern.search(CONTROL)}
check("the scan's patterns all fire on assembled identifiers (positive control)",
      len(FIRED) == len(L.FORBIDDEN_IN_REPO),
      f"only fired: {sorted(FIRED)}")
check("the IMDS link-local constant is the one dotted quad allowed, and only that one",
      not any(p.search(L.scrub_allowed("http://" + "169.254" + ".169.254/latest"))
              for p, _ in L.FORBIDDEN_IN_REPO)
      and any(p.search("10.100" + ".3.0/24") for p, _ in L.FORBIDDEN_IN_REPO))
check("the example site config ships placeholders only",
      "REPLACE" in open(os.path.join(HERE, "site_config.example.json")).read())

print(f"\n===== {'ALL PASS' if not FAILS else 'FAILURES: ' + '; '.join(FAILS)} "
      f"({len(FAILS)} failed of {len(TOTAL)} checks)")
print("      Nothing here launched anything. TASK-N18's verification needs a real trn2 run:\n"
      "      rendering, a dry-run, a quota and a spot price all establish nothing (REQ-087).\n")
raise SystemExit(1 if FAILS else 0)
