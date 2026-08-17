"""Render, validate and launch the Tier 2 Neuron verification job (TASK-N18).

Every trn2 run so far went out on a hand-rolled YAML in ``~/lingbot-tier1`` -- ``_v1.yaml``,
``_np2..4.yaml``, ``_r.yaml`` -- none of it in any repository. The recipe worked, and nine
separate details of it each cost a real run to learn. This file exists so that the tenth person
does not pay for them again, and so that a Tier 2 result can be reproduced by someone who is not
the person who ran it.

WHAT THIS DOES NOT DO
---------------------
It does not decide whether to spend money. ``--render`` is the default and produces artifacts
only; an actual launch needs ``--launch`` and an SSM-reachable SkyPilot controller. And
``--render`` establishes NOTHING about capacity: REQ-087 rejects dry-runs, quota figures,
offerings entries and spot prices as evidence that an instance is obtainable, having been
disproved twice on this project. The only writer of the capacity block is ``stages/identity.py``,
running on the instance -- this file cannot populate it and a test asserts it never tries.

SITE VALUES
-----------
Nothing account-, VPC-, bucket- or instance-shaped is committed here: this repository has public
remotes. Values arrive from ``--site-config`` (see ``site_config.example.json``) or from
``LINGBOT_T2_*`` environment variables, and are validated by SHAPE, never against an expected
value -- a validator that knows the right subnet id has the subnet id in it.

THE ORDERING TRAP
-----------------
Constraint 4 requires ``git ls-remote`` to match local HEAD before launching, which is why
TASK-N00 gates this task. Committing this file moves HEAD past what TASK-N00 published, so
completing TASK-N18 breaks the very check it encodes. Re-run TASK-N00's publish steps before any
task uses this runner. The check here fails loudly rather than warning, so the trap surfaces at
the first attempted launch instead of in the artifact's provenance six weeks later.
"""
import argparse
import base64
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "job_tier2.yaml.tmpl")
STAGES_DIR = os.path.join(HERE, "stages")

# --------------------------------------------------------------------------------------------
# Profiles. A profile is a stage list plus one declaration that matters: whether the profile
# invokes an NKI kernel. REQ-090/REQ-091 turn on that single fact, so it is data on the profile
# rather than a judgement the operator makes at the command line.
# --------------------------------------------------------------------------------------------
ALWAYS = ("identity", "devices", "venv", "sdk_versions")

PROFILES = {
    "noop": {
        "stages": ALWAYS + ("noop", "complete"),
        "invokes_nki_kernel": False,
        "needs_neuron_python_stack": False,
        "purpose": "TASK-N18's own verification: launch, usable state, artifact to S3, teardown. "
                   "Computes no numbers on purpose -- a figure here would get cited.",
    },
    "c5-parity": {
        "stages": ALWAYS + ("repo", "reference", "c5_arm", "complete"),
        "invokes_nki_kernel": True,
        "needs_neuron_python_stack": True,
        "purpose": "TASK-N14a / Phase 4 V5: run check_c5_attention_parity.py --arm neuron against "
                   "owed step 2's staged GPU reference. Invokes attention_cte, so trn2 only.",
    },
    "env-probe": {
        "stages": ALWAYS + ("repo", "complete"),
        "invokes_nki_kernel": False,
        # False on purpose, and not by oversight: this profile's output IS the importability of the
        # stack, so an instance with no stack produces a valid env-probe result rather than a wasted
        # run. It is the one profile for which an unpinned AMI is a legitimate question.
        "needs_neuron_python_stack": False,
        "purpose": "Environment and version capture with the tree cloned but nothing computed: "
                   "REQ-088's pin, and the REQ-095 module/venv facts TASK-N19's gate needs.",
    },
}

# Every stage the template knows about. Kept here so a stage added to one and not the other is a
# refusal rather than a silently skipped step.
KNOWN_STAGES = tuple(ALWAYS) + ("noop", "repo", "reference", "c5_arm", "complete")

# --------------------------------------------------------------------------------------------
# REQ-090 / REQ-081 capacity categories.
# --------------------------------------------------------------------------------------------
CHEAP_PREFIXES = ("inf2.", "trn1.")
KERNEL_CAPABLE_PREFIXES = ("trn2.",)

CATEGORIES = {
    "no-nki-kernel": {
        "description": "a compile or execution check that invokes NO NKI kernel (REQ-090's own "
                       "examples: Phase 1's cache design compiled without attention, Phase 3's "
                       "control-flow rewrite under REQ-026)",
        "cheap_capacity_required": True,
    },
    "nki-kernel": {
        "description": "a check that invokes an NKI kernel, attention_cte in practice. REQ-081: "
                       "attention_cte dropped trn1/inf2 support, so trn2 is not a preference",
        "cheap_capacity_required": False,
    },
}

SITE_KEYS = {
    # key: (regex the value must match, human description, required?)
    "s3_bucket": (r"^[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9]$", "destination S3 bucket name", True),
    "s3_prefix": (r"^[A-Za-z0-9_\-./]{1,256}$", "key prefix under that bucket", True),
    "region": (r"^[a-z]{2}-[a-z]+-\d$", "AWS region", True),
    "zones": (None, "ordered list of AZ names to try, most-likely first", True),
    "subnet_id": (r"^subnet-[0-9a-f]{8,17}$", "subnet the controller places instances in", False),
    "vpc_name": (r"^[A-Za-z0-9_\-]{1,64}$", "VPC name as SkyPilot's config names it", False),
    "controller_instance_id": (r"^i-[0-9a-f]{8,17}$", "SkyPilot controller, reached over SSM",
                               False),
    # Where the controller *is*, which is not where the job *runs*. Conflating the two sent
    # send-command at the job's region and got InvalidInstanceId, because the controller sits in
    # one region and launches into others; `use_ssm` is what makes that work.
    "controller_region": (r"^[a-z]{2}-[a-z]+-\d$",
                          "region the SkyPilot controller runs in; defaults to `region`", False),
    "repo_url": (r"^(https://|git@)[A-Za-z0-9._:/\-]+$", "git remote the instance clones", False),
    "image_id": (r"^ami-[0-9a-f]{8,17}$", "DLAMI to pin (REQ-088)", False),
    "reference_uri": (r"^s3://[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9]/.*$",
                      "s3:// location of owed step 2's staged reference tensors", False),
    "venv_preference": (r"^[A-Za-z0-9_\-]{1,64}$",
                        "substring identifying the venv to activate, e.g. the NxD inference one",
                        False),
}

DEFAULT_VENV_PREFERENCE = "nxd_inference"

# The venv names a Neuron DLAMI actually reported, from the recorded runtime venv list. The job's
# discovery is checked against these before anything is launched: the discovery it replaced globbed
# `aws_neuron_venv*`, every name the DLAMI reports spells it `neuronx`, and so nothing matched. That
# failure is silent in the worst place -- the venv stage records
# NO_VENV_FOUND_USING_SYSTEM_PYTHON3, nkilib is unimportable under system python3, and a kernel arm
# exits neuron_kernel_unavailable having answered nothing on an instance already paid for. The older
# single-`neuron` spelling stays in the list so a discovery narrowed to only the current DLAMI is
# refused too.
# The venvs a real Neuron DLAMI reported, from the runtime list the trn2 probe captured
# (context.md's "DLAMI venvs discovered at runtime"). All four carry the `neuronx` spelling; the
# discovery this check replaced searched three `aws_neuron_venv*` globs and matched none of them.
#
# This tuple is a CONJUNCTION -- the discovery must be able to match every entry -- so nothing may
# be added to it that a DLAMI did not report. An earlier version carried `aws_neuron_venv_pytorch`
# "for older DLAMIs", which made the refusal below assert that the DLAMI reports a venv it does
# not, and blocked the correct narrowing `aws_neuronx_venv*`. A name we merely *tolerate* at
# runtime does not belong in a list of names we *require* the discovery to reach.
DLAMI_VENV_NAMES = (
    "aws_neuronx_venv_pytorch_2_8_nxd_inference",       # the venv REQ-095 names
    "aws_neuronx_venv_pytorch_2_8_nxd_training",        # and the one the probe actually used
    "aws_neuronx_venv_pytorch_2_8",
    "aws_neuronx_venv_jax_0_7",
)


def strip_shell_comment(line):
    """Drop a trailing `# ...`, respecting quotes, so a check reads what a line DOES.

    Only used by the render checks. A `#` inside single or double quotes is data, not a comment.
    """
    quote = None
    for i, ch in enumerate(line):
        if quote is not None:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#":
            return line[:i]
    return line


class Refused(Exception):
    """A refusal, not a warning. Every one of these is a run somebody already paid for."""


def refuse(msg):
    raise Refused(msg)


# --------------------------------------------------------------------------------------------
# site config
# --------------------------------------------------------------------------------------------
def load_site(path):
    """Site values from JSON, then environment overrides, then shape validation.

    Shape, never value. `subnet_id` must look like a subnet id; which subnet is correct is a
    property of the account and belongs in a file this repository does not contain.
    """
    site = {}
    if path:
        with open(path) as fh:
            site = json.load(fh)
        if not isinstance(site, dict):
            refuse(f"{path} does not contain a JSON object")
    for key in SITE_KEYS:
        env = os.environ.get("LINGBOT_T2_" + key.upper())
        if env:
            site[key] = json.loads(env) if key == "zones" and env.strip().startswith("[") else env

    # A placeholder left in place is the failure mode the example file invites, so it is named
    # rather than passed through to a launch that fails opaquely.
    for key, val in list(site.items()):
        if isinstance(val, str) and ("REPLACE" in val or "<" in val or "EXAMPLE" in val):
            refuse(f"site config key {key!r} still holds the example placeholder {val!r}")

    for key, (pattern, desc, required) in SITE_KEYS.items():
        val = site.get(key)
        if val in (None, "", []):
            if required:
                refuse(f"site config is missing {key!r} ({desc}); supply it in --site-config or "
                       f"LINGBOT_T2_{key.upper()}")
            continue
        if key == "zones":
            if not isinstance(val, list) or not val:
                refuse("site config 'zones' must be a non-empty list of AZ names")
            for zone in val:
                if not re.match(r"^[a-z]{2}-[a-z]+-\d[a-z]$", str(zone)):
                    refuse(f"zone {zone!r} is not shaped like an availability zone")
            continue
        if pattern and not re.match(pattern, str(val)):
            refuse(f"site config {key!r}={val!r} is not shaped like a {desc}")
    site.setdefault("venv_preference", DEFAULT_VENV_PREFERENCE)
    return site


# --------------------------------------------------------------------------------------------
# constraint 9: the venue decision
# --------------------------------------------------------------------------------------------
def check_category(instance_type, category, profile_name, allow_expensive):
    """Constraint 9 and REQ-090/REQ-091, as a refusal rather than a note in a runbook.

    The corrected form of TASK-N18 step 9: the runner takes instance type, region/AZ and subnet as
    inputs, validates that the requested category is one REQ-090 recognizes, and refuses the two
    combinations that are wrong for a reason rather than for a price.

    This is deliberately NOT a resolution of the venue conflict design.md's V4 row holds open.
    TASK-N12 records that conflict as a conflict; parameterizing makes both outcomes executable
    from tracked artifacts instead of pre-empting a decision only the tech lead can make.
    """
    if category not in CATEGORIES:
        refuse(f"--category {category!r} is not one REQ-090 recognizes. Choose from: "
               + ", ".join(sorted(CATEGORIES)))

    profile = PROFILES[profile_name]
    if profile["invokes_nki_kernel"] and category == "no-nki-kernel":
        refuse(f"profile {profile_name!r} invokes an NKI kernel, so --category no-nki-kernel is "
               "false on its face. REQ-090 applies only to checks that invoke no NKI kernel at "
               "all (REQ-091 leaves the scope of the trn1/inf2 exclusion open, and until it is "
               "resolved the narrow reading governs)")
    if not profile["invokes_nki_kernel"] and category == "nki-kernel":
        refuse(f"profile {profile_name!r} invokes no NKI kernel, so --category nki-kernel "
               "overstates what it needs and would send a kernel-free check to trn2 against "
               "REQ-090. If that is deliberate, say so with a profile that declares it")

    cheap = instance_type.startswith(CHEAP_PREFIXES)
    kernel_capable = instance_type.startswith(KERNEL_CAPABLE_PREFIXES)
    if not cheap and not kernel_capable:
        refuse(f"instance type {instance_type!r} is named by neither REQ-090's cheap capacity "
               "(inf2.*, trn1.*) nor REQ-081's kernel-capable generation (trn2.*). Add it to "
               "one of those lists with the requirement that justifies it, rather than letting "
               "an unclassified venue produce an artifact TASK-N12 cannot score")

    if category == "nki-kernel" and cheap:
        refuse(f"{instance_type} cannot run an NKI-kernel check: attention_cte dropped "
               "trn1/inf2 support (REQ-081), so this would not be a cheap answer to the "
               "question -- it would be a different question. This refusal is the load-bearing "
               "half of step 9's 'parameterize the category': the cheap venue is the wrong "
               "answer for any check that invokes a kernel, however much cheaper it is")

    if category == "no-nki-kernel" and kernel_capable and not allow_expensive:
        refuse(f"{instance_type} is ~two orders of magnitude more expensive than REQ-090's "
               "cheap capacity, and this check invokes no NKI kernel, so REQ-090 says run it on "
               "the cheapest sufficient capacity. If the venue conflict design.md's V4 row holds "
               "open is being resolved deliberately in trn2's favour, pass "
               "--allow-expensive-venue and the reason is recorded in the manifest")

    return {
        "requested": category,
        "requested_description": CATEGORIES[category]["description"],
        "instance_type": instance_type,
        "instance_type_class": "cheap" if cheap else "kernel_capable",
        "profile_invokes_nki_kernel": profile["invokes_nki_kernel"],
        "expensive_venue_override": bool(allow_expensive and category == "no-nki-kernel"
                                         and kernel_capable),
        "note": "this records the REQUEST. The category actually reached is derived on the "
                "instance from IMDS by stages/identity.py and recorded as category_reached; "
                "REQ-090 scores that one",
    }


def check_image_pin(site, profile_name, allow_unpinned):
    """Refuse a profile that needs the Neuron Python stack on an AMI nobody chose.

    TASK-N18's verifying run is why this exists. It left ``image_id`` unset, SkyPilot picked its own
    AMI, and the instance came up with the Neuron *driver* working -- ``neuron_ls`` enumerated the
    device, ``/dev/neuron0`` was there -- and no Python stack whatsoever: no venv at any searched
    path, and torch, torch_xla, torch_neuronx, neuronxcc, neuronx_distributed,
    neuronx_distributed_inference and nkilib all unimportable. The ``noop`` profile passed anyway
    because it asks for none of them, so the gap is invisible in exactly the run that proves the
    plumbing works, and would have surfaced on the first expensive one.

    A driver without a stack is the failure mode this refusal is for: it looks like a working
    instance right up to the first import, which on trn2 is several dollars and a provisioning wait
    after the money starts. The override exists because "does this AMI have a stack?" is a legitimate
    question -- but it has to be asked on purpose, and it is recorded when it is.
    """
    profile = PROFILES[profile_name]
    pinned = bool(site.get("image_id"))
    if profile["needs_neuron_python_stack"] and not pinned and not allow_unpinned:
        refuse(f"profile {profile_name!r} runs a model, and site config pins no image_id. The AMI "
               "SkyPilot then chooses is not guaranteed to be a Neuron DLAMI: TASK-N18's verifying "
               "run got one with the driver present and torch, torch_neuronx, neuronxcc, "
               "neuronx_distributed and nkilib all unimportable, which is an instance that cannot "
               "run this profile at all. Set image_id to a Neuron DLAMI -- which also satisfies "
               "REQ-088's fourth item -- or pass --allow-unpinned-image if discovering what the "
               "default AMI carries is the actual intent, and the override is recorded")
    return {
        "image_id": site.get("image_id"),
        "pinned": pinned,
        "profile_needs_neuron_python_stack": profile["needs_neuron_python_stack"],
        "unpinned_image_override": bool(allow_unpinned and profile["needs_neuron_python_stack"]
                                        and not pinned),
        "note": "REQ-088's pin is recorded either way by stages/sdk_versions.py, from the AMI id "
                "the instance reports. Recording which AMI ran is not the same as choosing it",
    }


# --------------------------------------------------------------------------------------------
# constraint 4: the revision check
# --------------------------------------------------------------------------------------------
def check_revision(repo_root, repo_url, ref, git=None):
    """`git ls-remote` must match local HEAD before launching (constraint 4).

    Failing loudly on an ls-remote error rather than skipping the check: an unreachable remote is
    exactly when a stale local ref is most likely to be what gets cloned, and the whole point is
    that the instance runs the revision somebody can look at.
    """
    git = git or (lambda args: subprocess.run(["git"] + args, cwd=repo_root, capture_output=True,
                                              text=True, timeout=60))

    head = git(["rev-parse", "HEAD"])
    if head.returncode != 0:
        refuse(f"cannot read local HEAD in {repo_root}: {head.stderr.strip()[:200]}")
    local = head.stdout.strip()

    status = git(["status", "--porcelain"])
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None

    remote = git(["ls-remote", repo_url, f"refs/heads/{ref}"])
    if remote.returncode != 0:
        refuse(f"git ls-remote {repo_url} refs/heads/{ref} failed: "
               f"{remote.stderr.strip()[:200]}. Not skipped: an unreachable remote is when a "
               "stale local ref is most likely to be what the instance clones")
    lines = [ln for ln in remote.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        refuse(f"the remote has no refs/heads/{ref}. TASK-N00's publish step has not run for "
               "this branch, so there is nothing for the instance to clone")
    published = lines[0].split()[0]

    if published != local:
        refuse(
            f"local HEAD {local[:12]} != published refs/heads/{ref} {published[:12]}. The "
            "instance would clone code nobody can match to this desk (constraint 4). Note the "
            "ordering trap TASK-N18 records: committing the runner itself moves HEAD past what "
            "TASK-N00 published, so re-run TASK-N00's publish steps 1/2/4 and then retry")
    if dirty:
        refuse(f"{repo_root} has uncommitted changes. HEAD matches the remote, so the instance "
               "would clone something that is not what is on this desk -- which is the same "
               "defect as a mismatch, wearing a passing check")
    return {"local_head": local, "published_head": published, "ref": ref, "dirty": False}


# --------------------------------------------------------------------------------------------
# rendering, and the constraint checks over the rendered text
# --------------------------------------------------------------------------------------------
B64_TOKEN = re.compile(r"@@B64:([A-Za-z0-9_./\-]+)@@")
ANY_TOKEN = re.compile(r"@@[A-Z0-9_:./\-]+@@")

# Values that must never appear in a committed file in this repository, which has public remotes.
# `s3://` is matched only when a concrete bucket character follows it: `s3://${VAR}` is the form
# every committed reference has to take, and forbidding the scheme outright would forbid that too.
FORBIDDEN_IN_REPO = (
    (re.compile(r"\b\d{12}\b"), "a 12-digit AWS account id"),
    (re.compile(r"\bsubnet-[0-9a-f]{8,}\b"), "a subnet id"),
    (re.compile(r"\brtb-[0-9a-f]{8,}\b"), "a route table id"),
    (re.compile(r"\bi-[0-9a-f]{8,}\b"), "an instance id"),
    (re.compile(r"\bvpc-[0-9a-f]{8,}\b"), "a VPC id"),
    (re.compile(r"s3://[a-z0-9]"), "a concrete S3 URI"),
    # `arn:aws` alone would match this very list. Requiring the colon that closes the partition
    # field matches every real ARN (aws, aws-cn, aws-us-gov) and not a pattern definition.
    (re.compile(r"arn:aws[a-z\-]*:"), "an ARN"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b"), "an IPv4 address or CIDR block"),
)
# The link-local IMDS address is the one dotted quad allowed: a fixed, published constant,
# identical on every EC2 instance, which identifies nothing about this account. Elided before the
# scan rather than special-cased inside it, so the scan itself has no exceptions to get wrong.
IMDS_LITERAL = "169.254.169.254"


def scrub_allowed(text):
    return text.replace(IMDS_LITERAL, "IMDS_LINK_LOCAL")


def logical_lines(text):
    """Join backslash continuations before checking anything line-oriented.

    The watcher launch in the template spans two physical lines: `setsid nohup ... \\` then
    `> out 2>&1 < /dev/null &`. Checking constraint 3 per physical line would report the redirect
    as missing on a line that has it, which is the kind of validator that gets deleted rather than
    fixed. Line numbers are the FIRST physical line of each logical one, so a refusal still points
    somewhere real.
    """
    out, buf, start = [], "", 1
    for i, line in enumerate(text.split("\n"), 1):
        if not buf:
            start = i
        if line.rstrip().endswith("\\"):
            buf += line.rstrip()[:-1] + " "
            continue
        out.append((start, buf + line))
        buf = ""
    if buf:
        out.append((start, buf))
    return out


def render(site, profile_name, run_id, category, repo_head, opts, template=None):
    """Substitute every token, base64 every embedded script, then check the result."""
    src = template if template is not None else open(TEMPLATE).read()
    text = src
    profile = PROFILES[profile_name]

    for stage in profile["stages"]:
        if stage not in KNOWN_STAGES:
            refuse(f"profile {profile_name!r} names stage {stage!r}, which KNOWN_STAGES does not")

    # constraint 1: base64, and only base64. A heredoc or a `python -c` in the template is the
    # SSM+YAML quoting hazard this constraint exists for, so it is refused at render time.
    encoded = {}
    for match in B64_TOKEN.finditer(text):
        rel = match.group(1)
        path = os.path.join(HERE, rel)
        if not os.path.exists(path):
            refuse(f"the template embeds {rel!r}, which does not exist")
        with open(path, "rb") as fh:
            raw = fh.read()
        encoded[rel] = base64.b64encode(raw).decode()
    text = B64_TOKEN.sub(lambda m: encoded[m.group(1)], text)

    extra = []
    if site.get("image_id"):
        extra.append(f"  image_id: {site['image_id']}")
    else:
        extra.append("  # no image_id pinned: REQ-088 then depends on the DLAMI id the instance "
                     "reports, which stages/sdk_versions.py records")

    any_of = "\n".join(
        f"    - {{region: {site['region']}, zone: {zone}, use_spot: {str(bool(opts.spot)).lower()}}}"
        for zone in site["zones"])

    values = {
        "@@RUN_NAME@@": f"lingbot-t2-{profile_name}-{run_id}",
        "@@INSTANCE_TYPE@@": opts.instance_type,
        "@@DISK_SIZE@@": str(opts.disk_size),
        "@@RESOURCES_EXTRA@@": "\n".join(extra),
        "@@ANY_OF@@": any_of,
        "@@S3_BUCKET@@": site["s3_bucket"],
        "@@S3_PREFIX@@": site["s3_prefix"],
        "@@RUN_ID@@": run_id,
        "@@PROFILE@@": profile_name,
        "@@STAGES@@": ",".join(profile["stages"]),
        "@@CATEGORY@@": category,
        "@@REPO_URL@@": site.get("repo_url", ""),
        "@@REPO_REF@@": opts.ref,
        "@@REPO_HEAD@@": repo_head or "",
        "@@REFERENCE_URI@@": site.get("reference_uri", ""),
        "@@VENV_PREFERENCE@@": site["venv_preference"],
        "@@C5_TOLERANCE@@": "" if opts.c5_tolerance is None else repr(opts.c5_tolerance),
        "@@C5_REQUIRE_FULL@@": "1" if opts.c5_require_full_geometry else "",
    }
    for token, val in values.items():
        text = text.replace(token, val)

    left = ANY_TOKEN.findall(text)
    if left:
        refuse(f"unsubstituted tokens remain after rendering: {sorted(set(left))}")

    check_rendered(text, profile, src)
    return text


def check_rendered(text, profile, template_src=None):
    """The nine constraints, checked over the text that will actually run.

    Over the rendered text and not only the template, because rendering is where a supplied value
    could reintroduce what the template forbids.
    """
    lines = logical_lines(text)

    # 1 -- no embedded script may travel any way but base64.
    for i, line in lines:
        if re.search(r"<<-?\s*['\"]?\w*(EOF|PYEOF|PY)\b", line):
            refuse(f"line {i}: a heredoc embeds a script. Constraint 1: base64-encode embedded "
                   "scripts to survive SSM + YAML quoting -- the scratch YAMLs used heredocs and "
                   "then had to be rewritten to base64 mid-project")
        if re.search(r"\bpython3?\s+-c\b", line):
            refuse(f"line {i}: an inline `python -c` is an embedded script with worse quoting "
                   "odds than a file. Put it in stages/ and embed it as base64 instead")

    # 2 and 3 -- the sleep cap, and the one exemption from it.
    for i, line in lines:
        for match in re.finditer(r"\bsleep\s+\"?\$?\{?(\d+)", line):
            seconds = int(match.group(1))
            if seconds <= 115:
                continue
            # A watcher may wait longer, because its wait is not a command anybody is timing out.
            # It has to actually be detached to qualify.
            detached = ("setsid" in line and "nohup" in line and "< /dev/null" in line)
            if not detached:
                refuse(f"line {i}: sleep {seconds}s exceeds the 115 s cap for a single command "
                       "(constraint 2) and is not a detached watcher. An SSM document that "
                       "blocks past its own timeout is killed with its stage half-done: "
                       f"{line.strip()[:90]}")
    for i, line in lines:
        if re.search(r"\bsetsid\b", line) or ("nohup" in line and line.rstrip().endswith("&")):
            missing = [tok for tok in ("setsid", "nohup", "< /dev/null") if tok not in line]
            if missing:
                refuse(f"line {i}: a watcher is launched without {missing} (constraint 3). "
                       "`setsid nohup ... < /dev/null` is the whole form; a watcher that dies "
                       "with its session records no preemption, which is the one case REQ-092 "
                       f"exists for: {line.strip()[:90]}")

    # 5 -- source the venv; never invoke its interpreter directly.
    for i, line in lines:
        if re.search(r"\$\{?VENV\}?/bin/python", line) or re.search(r"venv[\w./]*/bin/python\b",
                                                                    line):
            refuse(f"line {i}: invoking a venv's python directly leaves the PJRT plugin path "
                   "unset, the device count reads zero, and the run looks like a hardware "
                   "finding. That broke Phase 0 run 1. Use `. \"$VENV/bin/activate\"` "
                   "(constraint 5)")
    if not re.search(r'\.\s+"\$VENV/bin/activate"', text):
        refuse("no `. \"$VENV/bin/activate\"` anywhere in the rendered job (constraint 5)")

    # 5b -- and the discovery must be able to FIND a venv to activate. Checked by matching every
    # name the DLAMI reports against the patterns the job actually searches, because the discovery
    # this replaced was three plausible-looking globs that matched none of them and said nothing.
    #
    # Comments are stripped first, and that is not a detail: tokenizing the raw line let PROSE
    # satisfy the check. `VENVS=$(ls -d /opt/aws_neuron_venv*)  # was -name 'aws_neuron*venv*'`
    # rendered clean while searching for exactly the pattern that found nothing on a real DLAMI.
    # A comment cannot find a venv, and a `VENVS=` that survives only inside one is not a discovery.
    discovery = [s for s in (strip_shell_comment(line) for _, line in lines) if "VENVS=" in s]
    patterns = [os.path.basename(tok.strip("'\"()")) for line in discovery
                for tok in line.split() if "venv" in tok]
    for venv_name in DLAMI_VENV_NAMES:
        if not any(fnmatch.fnmatchcase(venv_name, pat) for pat in patterns):
            refuse(f"the job's venv discovery cannot match {venv_name}, a venv the DLAMI reports: "
                   f"it searches for {patterns or 'nothing at all'}. Not a warning -- the venv "
                   "stage then records NO_VENV_FOUND_USING_SYSTEM_PYTHON3, nkilib is unimportable "
                   "under system python3, and the arm exits neuron_kernel_unavailable with the "
                   "instance paid for (constraint 5)")

    # 6 -- both variables, always. One without the other and the kernel cannot identify a platform.
    for var in ("PJRT_DEVICE=NEURON", "NEURON_PLATFORM_TARGET_OVERRIDE=trn2"):
        if f"export {var}" not in text:
            refuse(f"the rendered job never exports {var} (constraint 6). PJRT_DEVICE alone is "
                   "why the first neuron_probe.json recorded no devices on a 16-device host")

    # 7 -- the version capture is not optional.
    if "sdk_versions" not in profile["stages"]:
        refuse("every profile must include the sdk_versions stage: REQ-088 requires the SDK, "
               "compiler, kernel and DLAMI versions recorded for every comparison, and a "
               "tolerance derived under one pinned set is not valid under another")

    # 8 -- one flush per stage, checked by counting rather than by trusting a code path.
    guards = re.findall(r"^\s*if stage (\w+); then\s*$", text, re.M)
    flushes = re.findall(r"^\s*flush (\w+)\s*$", text, re.M)
    if sorted(guards) != sorted(flushes):
        refuse(f"stage guards {sorted(guards)} do not match flush calls {sorted(flushes)} "
               "(constraint 8). Phase 0 run 2 was preempted mid-compile and was recoverable only "
               "because every stage had already flushed")
    for stage in profile["stages"]:
        if stage not in guards:
            refuse(f"profile stage {stage!r} has no `if stage {stage}; then` guard in the job")

    # 9 -- nothing site-specific may be committed in the template itself. Values the operator
    # passed in are expected in the RENDERED text; a hard-coded one in the template is not, and
    # this repository has public remotes.
    src = template_src if template_src is not None else open(TEMPLATE).read()
    for pattern, desc in FORBIDDEN_IN_REPO:
        match = pattern.search(scrub_allowed(src))
        if match:
            refuse(f"the job template contains {desc} ({match.group(0)[:24]!r}). This repository "
                   "has public remotes; site values belong in --site-config, reached through a "
                   "token")
    return True


# --------------------------------------------------------------------------------------------
# launching, and tearing down only what we made
# --------------------------------------------------------------------------------------------
def ssm_commands(cluster, yaml_b64, opts):
    """The SSM command bodies handed to the SkyPilot controller.

    SkyPilot lives on the controller, not on the desk, so the job travels as base64 through
    ``AWS-RunShellScript`` -- constraint 1's original motivation. Two commands, because
    constraint 2's 115 s cap is about a single command: the launch is started detached and polled,
    never waited on inline.

    A managed job, not a cluster launch. The controller carries an admin policy that raises on
    CLUSTER_LAUNCH and CLUSTER_EXEC -- `sky launch` is refused there before any capacity is
    requested, so the first version of this function could not have launched anything on the
    controller it was written for. `sky jobs launch` is the accepted entry point, and it takes no
    `-c` at all: the name below is the *job* name, and SkyPilot derives its own cluster name from
    it. That also moves REQ-064's teardown into a property of the launch rather than a command
    somebody has to remember -- a managed job terminates its cluster when the job ends, including
    when it fails.
    """
    remote_yaml = f"/tmp/lingbot_t2/{cluster}.yaml"
    launch = "\n".join([
        "set -eu",
        "mkdir -p /tmp/lingbot_t2",
        f"echo {yaml_b64} | base64 -d > {remote_yaml}",
        # Detached, so this command returns immediately instead of blocking past the document's
        # timeout while capacity is acquired. -d also asks SkyPilot itself to detach.
        f"setsid nohup sky jobs launch -n {cluster} -y -d {remote_yaml} "
        f"> /tmp/lingbot_t2/{cluster}.launch.log 2>&1 < /dev/null &",
        "sleep 5",
        f"echo started; tail -n 5 /tmp/lingbot_t2/{cluster}.launch.log || true",
    ])
    poll = "\n".join([
        "set -u",
        # -a so a job that has already finished still shows: for a run this short, the interesting
        # states (SUCCEEDED, FAILED_NO_RESOURCE) are terminal ones that the default view hides.
        f"sky jobs queue -a 2>&1 | grep -E 'ID|{cluster}' | head -n 20 || true",
        f"tail -n 40 /tmp/lingbot_t2/{cluster}.launch.log 2>/dev/null || true",
    ])
    down = "\n".join([
        "set -u",
        # REQ-064: this names the job this run created, and nothing else. No terminate-instances,
        # no delete-volume, no filter that could match a resource this initiative did not create.
        # `sky jobs cancel -n` is narrower than the `sky down` it replaces -- it can only reach a
        # job submitted under this name, where a cluster name could in principle be reused.
        f"sky jobs cancel -n {cluster} -y 2>&1 | tail -n 20 || true",
        f"sky jobs queue -a 2>&1 | grep -E 'ID|{cluster}' | head -n 10 || true",
    ])
    return {"launch": launch, "poll": poll, "down": down, "remote_yaml": remote_yaml}


CLUSTER_RE = re.compile(r"^lingbot-t2-[a-z0-9\-]+-[0-9a-z]{6,}$")


def teardown_guard(cluster):
    """REQ-064, as a name check rather than a promise.

    'Do not delete or terminate any capacity-allocation resource without explicit user direction'
    is not satisfiable by a runner that will cancel whatever string it is handed. Only names this
    runner generates are eligible.
    """
    if not CLUSTER_RE.match(cluster or ""):
        refuse(f"refusing to tear down {cluster!r}: it is not a name this runner generates "
               "(lingbot-t2-<profile>-<runid>). REQ-064 forbids deleting or terminating anything "
               "this initiative did not create, and a cluster nobody here named is exactly that")
    return True


def check_controller(instance_id, region, runner=None):
    """Refuse a launch at a controller that is not there, and say which fact is wrong.

    Two site values can each be stale in a way ``send-command`` reports identically. The recorded
    instance may have been replaced -- an instance id is not a durable name for a controller, and
    the one this runner was written against no longer exists in any state. Or the region may be the
    job's rather than the controller's, which is the same conflation that used to be in the call
    below. Both surface as one opaque ``InvalidInstanceId`` after the manifest has already been
    built, so ask first and name both candidates in the refusal.
    """
    runner = runner or (lambda cmd: subprocess.run(cmd, capture_output=True, text=True,
                                                   timeout=60))
    proc = runner(["aws", "ssm", "describe-instance-information", "--region", region,
                   "--filters", f"Key=InstanceIds,Values={instance_id}", "--output", "json"])
    if proc.returncode != 0:
        refuse(f"could not ask SSM about controller {instance_id} in {region}: "
               f"{(proc.stderr or '').strip()[:200]}")
    try:
        info = json.loads(proc.stdout)["InstanceInformationList"]
    except Exception:                                   # noqa: BLE001
        refuse(f"unreadable describe-instance-information response for {instance_id}")
    if not info:
        refuse(f"controller {instance_id} is not an SSM-managed instance in {region}. Either the "
               "recorded 'controller_instance_id' has been replaced, or 'controller_region' is "
               "wrong -- it is the region the CONTROLLER runs in, not the region the job launches "
               "into. Those are different facts and this runner used to have only one field")
    ping = (info[0] or {}).get("PingStatus")
    if ping != "Online":
        refuse(f"controller {instance_id} is {ping!r} to SSM, not Online. send-command would be "
               "accepted and never delivered, which reads exactly like a launch that quietly did "
               "nothing")
    return {"instance_id": instance_id, "region": region, "ping_status": ping,
            "agent_version": (info[0] or {}).get("AgentVersion")}


# --------------------------------------------------------------------------------------------
# collecting, and the REQ-092 classification
# --------------------------------------------------------------------------------------------
def classify(result, interrupt=None):
    """Map a collected artifact to a verdict. An interrupted run is INCOMPLETE, never FAIL.

    The mechanism: ``terminal_state`` is written only by ``stages/verdict.py``, in the job's last
    stage. A run that was killed never reached that line, so its absence *is* the evidence of an
    incomplete run -- nothing has to have detected the preemption for the classification to come
    out right. The watcher's sidecar, when it exists, supplies the cause and the elapsed time
    REQ-092 requires captured.
    """
    if not isinstance(result, dict) or not result:
        return {"verdict": "INCOMPLETE",
                "because": "no result artifact was retrieved; a run that produced nothing is "
                           "not a run that failed"}
    interrupted = interrupt if (interrupt or {}).get("cause") else result.get("_interrupted")
    reached = result.get("terminal_state") == "reached_complete_stage"

    if interrupted:
        return {
            "verdict": "INCOMPLETE",
            "because": "REQ-092: infrastructure-level interruption is recorded as "
                       "inconclusive-infrastructure-interrupted, not as a phase failure",
            "cause": interrupted.get("cause"),
            "elapsed_s": interrupted.get("elapsed_s") or result.get("_elapsed_s"),
            "last_stage_flushed": result.get("_last_stage_flushed"),
            "req092_rerun_from": result.get("_last_stage_flushed"),
        }
    if not reached:
        return {
            "verdict": "INCOMPLETE",
            "because": "the job never reached its complete stage, so no verdict was written. "
                       "REQ-092: an interrupted run is not a failed one, and the absence of a "
                       "terminal state cannot be told apart from a preemption whose notice never "
                       "arrived -- so it is classified the same way, not as FAIL",
            "elapsed_s": result.get("_elapsed_s"),
            "last_stage_flushed": result.get("_last_stage_flushed"),
        }
    return {
        "verdict": result.get("verdict") or "INCOMPLETE",
        "because": result.get("verdict_because") or "the job reached its complete stage but wrote "
                                                    "no verdict",
        "elapsed_s": result.get("_elapsed_s"),
        "capacity": result.get("capacity"),
        "category_reached": result.get("category_reached"),
        "attribution": result.get("attribution"),
    }


def collect(site, run_id, out_dir, runner=None):
    """Pull the run's artifacts from S3 and classify them."""
    runner = runner or (lambda cmd: subprocess.run(cmd, capture_output=True, text=True,
                                                   timeout=300))
    base = f"s3://{site['s3_bucket']}/{site['s3_prefix']}/{run_id}"
    os.makedirs(out_dir, exist_ok=True)
    got = {}
    for name in ("tier2_result.json", "tier2_interrupt.json", "c5_arm.json"):
        dest = os.path.join(out_dir, name)
        proc = runner(["aws", "s3", "cp", f"{base}/{name}", dest, "--only-show-errors"])
        if proc.returncode == 0 and os.path.exists(dest):
            try:
                with open(dest) as fh:
                    got[name] = json.load(fh)
            except Exception as exc:                    # noqa: BLE001
                got[name] = {"_unreadable": f"{type(exc).__name__}: {exc}"[:200]}
        else:
            got[name] = None
    verdict = classify(got.get("tier2_result.json") or {}, got.get("tier2_interrupt.json") or {})
    return {"run_id": run_id, "s3_base": base, "artifacts_present":
            {k: v is not None for k, v in got.items()}, "classification": verdict,
            "result": got.get("tier2_result.json")}


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------
def make_run_id(profile, head, now=None):
    """Short, reproducible-ish, and traceable to a revision.

    The revision goes in so that two runs of the same profile are told apart by what they ran,
    not only by when. Phase 0's artifacts recorded neither, which is finding #40.
    """
    now = int(now if now is not None else time.time())
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))
    short = (head or "nohead")[:7]
    return f"{stamp}-{short}-{hashlib.sha256(f'{profile}{now}{head}'.encode()).hexdigest()[:6]}"


def build_parser():
    ap = argparse.ArgumentParser(
        description="Render, validate and (with --launch) run the Tier 2 Neuron job.",
        epilog="--render is the default and proves nothing about capacity (REQ-087).")
    ap.add_argument("--profile", choices=sorted(PROFILES), required=True)
    ap.add_argument("--category", choices=sorted(CATEGORIES), required=True,
                    help="which REQ-090 category this check falls into. Required: the runner "
                         "will not guess, because the artifact records it and TASK-N12's "
                         "verification scores it")
    ap.add_argument("--instance-type", required=True,
                    help="e.g. a trn2 size for kernel checks, or REQ-090's cheap capacity")
    ap.add_argument("--site-config", help="JSON of site values; see site_config.example.json")
    ap.add_argument("--repo-root", default=os.path.abspath(os.path.join(HERE, "..", "..", "..")))
    ap.add_argument("--ref", default="neuron-port", help="branch the instance clones")
    ap.add_argument("--disk-size", type=int, default=256)
    ap.add_argument("--spot", action="store_true", default=True)
    ap.add_argument("--no-spot", dest="spot", action="store_false",
                    help="trn2 has no on-demand offer in the validated region; this exists for "
                         "REQ-090's cheap venues, which do")
    ap.add_argument("--allow-expensive-venue", action="store_true",
                    help="run a no-NKI-kernel check on trn2 anyway, against REQ-090's cheapest-"
                         "sufficient rule. Recorded in the manifest with this flag named")
    ap.add_argument("--allow-unpinned-image", action="store_true",
                    help="run a model-running profile without a pinned image_id. The default AMI "
                         "carried the Neuron driver and no Python stack at all; recorded when used")
    ap.add_argument("--c5-tolerance", type=float, default=None,
                    help="passed to the C5 arm. No default: the arm exits 2 MEASURED_NOT_SCORED "
                         "without one, and 2 is not a pass")
    ap.add_argument("--c5-require-full-geometry", action="store_true")
    ap.add_argument("--run-id", help="defaults to a UTC stamp plus the verified revision")
    ap.add_argument("--out-dir", default=None, help="where rendered artifacts are written")
    ap.add_argument("--launch", action="store_true",
                    help="actually send the job to the SkyPilot controller over SSM. Spends "
                         "money; trn2 is spot-only at $8.596/hr in the validated region")
    ap.add_argument("--collect", metavar="RUN_ID",
                    help="pull an earlier run's artifacts from S3 and classify them")
    ap.add_argument("--teardown", metavar="CLUSTER",
                    help="emit the SSM teardown command for a cluster THIS runner named")
    ap.add_argument("--skip-revision-check", action="store_true",
                    help="render without the constraint-4 check. Refused together with --launch")
    return ap


def main(argv=None):
    opts = build_parser().parse_args(argv)
    site = load_site(opts.site_config)

    if opts.collect:
        print(json.dumps(collect(site, opts.collect,
                                 opts.out_dir or os.path.join("/tmp", "t2-collect-" + opts.collect)),
                         indent=2, sort_keys=True))
        return 0

    if opts.teardown:
        teardown_guard(opts.teardown)
        print(json.dumps({
            "cluster": opts.teardown,
            "ssm_command": ssm_commands(opts.teardown, "", opts)["down"],
            "req064": "names only this cluster. After it returns, list volumes tagged with the "
                      "cluster and REPORT any that survive -- do not delete them: REQ-064 "
                      "forbids terminating capacity-allocation resources this initiative did "
                      "not create, and an orphan of unknown provenance is exactly that case",
        }, indent=2, sort_keys=True))
        return 0

    if opts.skip_revision_check and opts.launch:
        refuse("--skip-revision-check and --launch together would put an unidentifiable revision "
               "on a paid instance. Constraint 4 is the check; skipping it is for rendering only")

    category = check_category(opts.instance_type, opts.category, opts.profile,
                              opts.allow_expensive_venue)
    image_pin = check_image_pin(site, opts.profile, opts.allow_unpinned_image)

    revision = None
    if opts.skip_revision_check:
        revision = {"skipped": True,
                    "why": "--skip-revision-check: this render is not launchable evidence of "
                           "anything and its repo_head is empty"}
    else:
        if not site.get("repo_url"):
            refuse("constraint 4 needs site config 'repo_url' to run `git ls-remote` against. "
                   "Without it there is no way to check that the instance would clone this "
                   "desk's HEAD")
        revision = check_revision(opts.repo_root, site["repo_url"], opts.ref)

    run_id = opts.run_id or make_run_id(opts.profile, (revision or {}).get("local_head"))
    cluster = f"lingbot-t2-{opts.profile}-{run_id}".lower()[:60]
    rendered = render(site, opts.profile, run_id, opts.category,
                      (revision or {}).get("local_head"), opts)

    out_dir = opts.out_dir or os.path.join("/tmp", f"t2-render-{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    yaml_path = os.path.join(out_dir, f"{cluster}.yaml")
    with open(yaml_path, "w") as fh:
        fh.write(rendered)
    yaml_b64 = base64.b64encode(rendered.encode()).decode()
    cmds = ssm_commands(cluster, yaml_b64, opts)

    manifest = {
        "run_id": run_id,
        "cluster": cluster,
        "profile": opts.profile,
        "profile_purpose": PROFILES[opts.profile]["purpose"],
        "stages": list(PROFILES[opts.profile]["stages"]),
        "instance_type": opts.instance_type,
        "spot": bool(opts.spot),
        "region": site["region"],
        "zones": site["zones"],
        "category": category,
        "image_pin": image_pin,
        "revision": revision,
        "rendered_yaml": yaml_path,
        "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "s3_base": f"s3://{site['s3_bucket']}/{site['s3_prefix']}/{run_id}",
        "ssm_target": site.get("controller_instance_id"),
        # REQ-087, as a field the launcher writes and cannot make true. stages/identity.py is the
        # only writer of the capacity block, and it runs on the instance.
        "capacity_evidence": None,
        "capacity_evidence_note": "REQ-087 accepts only an actual launch that reached a usable "
                                  "state. Rendering, a dry-run, a quota figure, an offerings "
                                  "entry and a spot price all establish nothing; two of those "
                                  "were disproved on this project. Read `capacity` out of the "
                                  "run's own artifact with --collect",
        "launched": False,
    }

    if not opts.launch:
        manifest["next"] = {
            "how_to_launch": "re-run with --launch, or hand the ssm_command below to the "
                             "controller yourself",
            "ssm_document": "AWS-RunShellScript",
            "ssm_command_launch": cmds["launch"],
            "ssm_command_poll": cmds["poll"],
            "ssm_command_teardown": cmds["down"],
        }
        with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    if not site.get("controller_instance_id"):
        refuse("--launch needs site config 'controller_instance_id': SkyPilot runs on the "
               "controller, not on this desk, and the job travels to it over SSM")
    controller_region = site.get("controller_region") or site["region"]
    manifest["controller_region"] = controller_region
    manifest["controller_check"] = check_controller(site["controller_instance_id"],
                                                   controller_region)
    proc = subprocess.run(
        ["aws", "ssm", "send-command",
         "--instance-ids", site["controller_instance_id"],
         "--document-name", "AWS-RunShellScript",
         "--parameters", json.dumps({"commands": [cmds["launch"]]}),
         "--region", controller_region, "--output", "json"],
        capture_output=True, text=True, timeout=120)
    manifest["launched"] = proc.returncode == 0
    manifest["ssm_send_command_rc"] = proc.returncode
    manifest["ssm_send_command_stderr"] = proc.stderr.strip()[:800] or None
    try:
        manifest["ssm_command_id"] = json.loads(proc.stdout)["Command"]["CommandId"]
    except Exception:                                   # noqa: BLE001
        manifest["ssm_command_id"] = None
    manifest["next"] = {
        "poll": f"--collect {run_id}",
        "ssm_command_poll": cmds["poll"],
        "teardown": f"--teardown {cluster}",
        "reminder": "an instance was requested, which is still not evidence one was obtained. "
                    "Only the run's own `capacity` block establishes that (REQ-087)",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["launched"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
