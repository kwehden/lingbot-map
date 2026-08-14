"""Pin and record the SDK, compiler, kernel and image versions (REQ-088).

REQ-088 requires, for every Neuron-vs-GPU numerical comparison, the exact Neuron SDK version,
``neuronx-cc`` version, ``nkilib``/NKI kernel version or commit, and DLAMI image ID -- and forbids
carrying a tolerance derived under one such set over to another without re-deriving it. The
grounding is ``Oncall-Report-2026-03-23``: a Beta3 compiler regression produced ``NaN`` inside
``attention_cte`` across gemma3 / qwen2_vl / llama4 / llama3 / pixtral (``NKI-1559``), fixed in
KCA commit ``f4cdba82``. An empirically-derived tolerance means nothing against a moving kernel.

Two things this deliberately does NOT do:

* It does not fail the run when a package is absent. An absence is a recorded fact -- TASK-N19's
  REQ-095 gate turns on exactly one such absence and needs it typed, not fatal.
* It does not treat "importable" as "installed at a known version". Where a module has no
  ``__version__`` it records that, rather than leaving the reader to assume one was found.

Usage::

    python3 sdk_versions.py <result.json>
"""
import json
import os
import subprocess
import sys
import urllib.request

RESULT = sys.argv[1]

MODULES = (
    "torch",
    "torch_xla",
    "torch_neuronx",
    "neuronx_distributed",
    "neuronx_distributed_inference",
    "nkilib",
    "neuronxcc",
)
# The module REQ-095's gate is actually about. `neuronx_distributed` importing OK says nothing
# about this path: they are different distributions, which is why two prior imports on a live
# instance left the question open instead of settling it.
NXD_KVCACHE = "neuronx_distributed_inference.modules.kvcache.kv_cache_manager"


# rec.py is base64'd next to this script on the instance at /tmp/t2. Resolving it rather than
# hard-coding that path is what makes these stages runnable at the desk under test_tier2_job.py:
# a stage whose only exercise is on trn2 spot time is a stage nobody exercises.
REC = os.environ.get("LINGBOT_T2_REC") or "/tmp/t2/rec.py"
if not os.path.exists(REC):
    REC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rec.py")


def rec(key, value):
    subprocess.run([sys.executable, REC, RESULT, key,
                    value if isinstance(value, str) else json.dumps(value)], check=False)


def run(cmd, timeout=60):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"rc": proc.returncode,
                "stdout": proc.stdout.strip()[:2000],
                "stderr": proc.stderr.strip()[:1000]}
    except Exception as exc:                            # noqa: BLE001
        return {"rc": None, "unavailable_because": f"{type(exc).__name__}: {exc}"[:200]}


def module_version(name):
    try:
        mod = __import__(name, fromlist=["_"])
    except Exception as exc:                            # noqa: BLE001
        return {"importable": False, "why": f"{type(exc).__name__}: {str(exc)[:160]}"}
    ver = getattr(mod, "__version__", None)
    row = {"importable": True,
           "version": ver,
           "version_absent": ver is None,
           "file": getattr(mod, "__file__", None)}
    if ver is None:
        row["note"] = ("imported but exposes no __version__; recorded as unversioned rather "
                       "than as an unknown-but-present pin, because REQ-088 needs the exact "
                       "version and this is not one")
    return row


versions = {name: module_version(name) for name in MODULES}

# The NKI kernel identity REQ-088 names. A version string is not always exposed, so the module
# file path is captured too: it is what distinguishes a DLAMI-shipped nkilib from a side install.
try:
    from nkilib.core import attention as _att            # noqa: F401
    versions["nkilib.core.attention"] = {
        "importable": True,
        "file": getattr(_att, "__file__", None),
        "attention_cte_present": hasattr(_att, "attention_cte"),
        "attention_cte_type": type(getattr(_att, "attention_cte", None)).__name__,
    }
except Exception as exc:                                # noqa: BLE001
    versions["nkilib.core.attention"] = {
        "importable": False, "why": f"{type(exc).__name__}: {str(exc)[:160]}"}

nxd_gate = {"module": NXD_KVCACHE}
try:
    __import__(NXD_KVCACHE, fromlist=["_"])
    nxd_gate["importable"] = True
except Exception as exc:                                # noqa: BLE001
    nxd_gate["importable"] = False
    nxd_gate["why"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    nxd_gate["note"] = ("REQ-095 requires an install ATTEMPT as well before the negative branch "
                        "counts, and that attempt is TASK-N19's, not this stage's. Recorded here "
                        "so TASK-N19 starts from a fact rather than a guess")

ami = None
try:
    req = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"})
    with urllib.request.urlopen(req, timeout=5) as fh:
        tok = fh.read().decode()
    req = urllib.request.Request("http://169.254.169.254/latest/meta-data/ami-id",
                                headers={"X-aws-ec2-metadata-token": tok})
    with urllib.request.urlopen(req, timeout=5) as fh:
        ami = fh.read().decode()
except Exception:                                       # noqa: BLE001
    ami = None

block = {
    "neuronx_cc": run(["neuronx-cc", "--version"]),
    "neuron_driver_dpkg": run(["bash", "-lc", "dpkg -l 'aws-neuron*' 2>/dev/null | tail -n 20"]),
    "neuron_ls": run(["bash", "-lc", "neuron-ls --json-output 2>/dev/null | head -c 4000"]),
    "dlami_image_id": ami,
    "python": sys.version.split()[0],
    "python_executable": sys.executable,
    "virtual_env": os.environ.get("VIRTUAL_ENV"),
    "pjrt_device": os.environ.get("PJRT_DEVICE"),
    "platform_target_override": os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE"),
    "modules": versions,
    "nxd_kvcache_manager_gate": nxd_gate,
}

# The pin is complete only if the four things REQ-088 names are all present. Recorded as a
# boolean so a downstream comparison can refuse to score against an incomplete pin instead of
# quietly inheriting one, which is the failure REQ-088's second clause forbids.
complete = bool(
    ami
    and block["neuronx_cc"].get("rc") == 0
    and versions.get("torch_neuronx", {}).get("importable")
    and versions.get("nkilib.core.attention", {}).get("importable"))
block["pin_complete_for_req088"] = complete
block["pin_incomplete_because"] = None if complete else (
    "REQ-088 needs all four of: DLAMI image id, neuronx-cc version, Neuron SDK "
    "(torch_neuronx) version, and the nkilib/NKI kernel identity. At least one is missing "
    "above, so no tolerance may be derived or reused from this run")

rec("sdk", block)
rec("stages.sdk_versions.status", "OK" if complete else "RECORDED_INCOMPLETE_PIN")
print(json.dumps({"pin_complete": complete, "ami": ami}))
raise SystemExit(0)
