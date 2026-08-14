"""Desk-verify TASK-N08 owed step 1's Neuron arm against the SHIPPED source.

The arm's whole purpose is to be un-fakeable, so the thing that has to be established here is
that each of its guards actually fires -- every guard is exercised in its NEGATIVE direction, not
just its passing one. Two properties are the point:

* **Raise, never fall back.** On a host with no Neuron stack the arm must block, not quietly
  produce numbers. The desk GPU is exactly such a host, so this is desk-verifiable rather than
  a promise deferred to trn2. Case 2 is the one that shows it, and case 1 exists because the
  *other* way to block -- unset env vars -- looks identical in the exit code and demonstrates
  nothing: the run never reached the import. The arm records ``null`` there rather than claiming
  a property it did not exercise, and case 1 asserts that ``null``.
* **A stub run is a failure, not a pass.** ``install_cte_stub()``'s numbers agree with the GPU
  reference to ~1e-6, so a silent fallback presents as success. Cases 3 and 4 install the stub
  (class-level, then instance-level) and require exit 1 with no ``rel_err`` computed at all.

**Cases 5-9 do NOT constitute a Neuron run and this file must never be cited as one.** They
register a fake ``nkilib.core.attention`` in ``sys.modules`` so that the real, unpatched
``_load_kernel`` import succeeds, which is the only way to reach the staged-reference comparison
path (sha256 verification, replay, rel_err, tolerance scoring, coverage) from the desk. What they
establish is that the path is correct and its guards discriminate; what runs the arm for REQ-028
is trn2 hardware under ``TASK-N14a``, where the kernel is real and this fixture is absent.

Run in the Tier 1 container (torch cannot be installed on the host)::

    docker exec lingbot-tier1 python /home/kwehden/lingbot-tier1/lingbot-map/verify/neuron/\
test_c5_neuron_arm.py --staged /home/kwehden/lingbot-tier1/c5_floor_tensors_2026-08-13
"""
import argparse
import ast
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "check_c5_attention_parity.py")
MANIFEST = os.path.join(HERE, "c5_reference_floor_results.json")

spec = importlib.util.spec_from_file_location("c5parity", HARNESS)
mod = importlib.util.module_from_spec(spec)
sys.modules["c5parity"] = mod
spec.loader.exec_module(mod)

# Sampled before anything below exports PJRT_DEVICE. Case 2 needs to know that exporting the two
# Neuron variables, which it must do to reach the real import, does not change what device the
# desk run sees -- an env setting that quietly disabled CUDA would make the comparison cases
# measure something else.
CUDA_AT_IMPORT = torch.cuda.is_available()

FAILS = []


def check(name, cond, extra=""):
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        FAILS.append(name)


def arm(reference_tensors=None, tolerance=None, require_full=False, device=None,
        manifest=MANIFEST):
    """Run the arm in-process and return (exit_code, artifact)."""
    mod.RESULTS.clear()
    out = os.path.join(tempfile.mkdtemp(), "arm.json")
    code = mod.run_neuron_arm(argparse.Namespace(
        arm_out=out, reference_tensors=reference_tensors, reference_manifest=manifest,
        tolerance=tolerance, require_full_geometry=require_full, device=device))
    with open(out) as fh:
        return code, json.load(fh)


def failing(art):
    return [r["check"] for r in art["results"] if not r["pass"]]


def set_env(on):
    if on:
        os.environ["PJRT_DEVICE"] = "NEURON"
        os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
    else:
        os.environ.pop("PJRT_DEVICE", None)
        os.environ.pop("NEURON_PLATFORM_TARGET_OVERRIDE", None)


ap = argparse.ArgumentParser()
ap.add_argument("--staged", required=True,
                help="owed step 2's retained tensor directory (outside both repos)")
opts = ap.parse_args()

print("\n=== case 1: env unset -> BLOCKED, and it must NOT claim the raise property ===")
set_env(False)
code, art = arm()
check("exit 3", code == 3, f"code={code} verdict={art['verdict']}")
check("blocked_because names the env, not the kernel",
      art["blocked_because"] == "neuron_env_not_configured", art["blocked_because"])
check("raises_rather_than_falling_back is null, not true",
      art["raises_rather_than_falling_back"] is None,
      repr(art["raises_rather_than_falling_back"]))
check("and it says why null: the import was never reached", "raises_note" in art)
check("both provenance guards still ran and passed",
      failing(art) == [] and art["n_checks"] == 2, f"{art['n_checks']} checks, {failing(art)}")
check("no rel_err anywhere in the artifact", "cells" not in art and "worst_rel_err" not in art)
check("the missing variables are named individually",
      art["env"]["PJRT_DEVICE"] is None
      and "NEURON_PLATFORM_TARGET_OVERRIDE" in art["env"]["assert_env_raised"])

print("\n=== case 2: env set, no Neuron stack -> raises rather than falling back ===")
set_env(True)
check("exporting the two Neuron variables leaves this host's device unchanged",
      torch.cuda.is_available() == CUDA_AT_IMPORT,
      f"cuda before={CUDA_AT_IMPORT} after={torch.cuda.is_available()}")
code, art = arm()
check("exit 3", code == 3, f"code={code} verdict={art['verdict']}")
check("blocked_because names the kernel",
      art["blocked_because"] == "neuron_kernel_unavailable", art["blocked_because"])
check("raises_rather_than_falling_back is TRUE here",
      art["raises_rather_than_falling_back"] is True)
raised = next(r for r in art["results"]
              if r["check"] == "absent_kernel_raises_rather_than_falling_back")
check("the guard passed and recorded the exception", raised["pass"] and raised["raised"])
# ModuleNotFoundError, not ImportError: nkilib is absent rather than broken, and the subclass is
# what `except ImportError` in the shipped _load_kernel catches. Either name is the same evidence.
check("it was a RuntimeError chained from an import failure, from the shipped _load_kernel",
      raised["raised"]["type"] == "RuntimeError"
      and raised["raised"]["cause"] in ("ImportError", "ModuleNotFoundError"),
      f"{raised['raised']['type']} <- {raised['raised']['cause']}")
check("the message names nkilib", "nkilib" in raised["raised"]["message"],
      raised["raised"]["message"][:60])
check("no rel_err was computed", "cells" not in art)
check("no kernel recorded", art["kernel"] is None)

print("\n=== case 3: the stub installed class-wide -> FAIL, and no rel_err at all ===")
mod.install_cte_stub(honour_prior_used_len=True)
code, art = arm(reference_tensors=opts.staged, tolerance=1e-3)
check("exit 1, not 0", code == 1, f"code={code} verdict={art['verdict']}")
check("both provenance guards failed",
      set(failing(art)) == {"stub_was_never_installed_in_this_process",
                            "load_kernel_is_still_the_shipped_method"}, str(failing(art)))
witness = next(r for r in art["results"]
               if r["check"] == "stub_was_never_installed_in_this_process")
check("the witness names where the stub was installed",
      witness["stub_installs"] and "test_c5_neuron_arm.py" in
      witness["stub_installs"][0]["called_from"], str(witness["stub_installs"]))
check("the patched method is reported as the stub lambda",
      "install_cte_stub" in str(next(r for r in art["results"]
                                     if r["check"] == "load_kernel_is_still_the_shipped_method"
                                     )["observed"]["qualname"]))
check("provenance was scored BEFORE rel_err: none exists despite a tolerance being supplied",
      "cells" not in art and "worst_rel_err" not in art, str(sorted(art)))
check("and it says why nothing was scored", "not_scored_because" in art)
# Restore the shipped method: every later case needs a pristine class.
mod.NeuronAttentionAdapter._load_kernel = mod._PRISTINE_LOAD_KERNEL
mod.STUB_INSTALLS.clear()
check("the restore really restored it",
      mod.NeuronAttentionAdapter._load_kernel is mod._PRISTINE_LOAD_KERNEL)

print("\n=== case 4: an INSTANCE-level shadow the class check cannot see -> FAIL ===")
real_build = mod.build_with_adapter


def shadowing_build(head_dim, num_heads, mtf, device, check_env):
    cache, ad = real_build(head_dim, num_heads, mtf, device, check_env)
    ad._load_kernel = lambda: mod.make_cte_stub(True)       # per-instance, class untouched
    return cache, ad


mod.build_with_adapter = shadowing_build
code, art = arm(reference_tensors=opts.staged, tolerance=1e-3)
mod.build_with_adapter = real_build
check("exit 1", code == 1, f"code={code} verdict={art['verdict']}")
check("the instance guard is among the failures",
      "adapter_instance_does_not_shadow_load_kernel" in failing(art), str(failing(art)))
check("and the kernel it returned was rejected for not being under nkilib",
      "kernel_is_the_real_nkilib_attention_cte" in failing(art), str(failing(art)))
check("no rel_err", "cells" not in art)

# ---------------------------------------------------------------------------------------
# Cases 5-9 install the fake nkilib. NOT A NEURON RUN -- see the module docstring.
# ---------------------------------------------------------------------------------------
print("\n=== installing a fake nkilib.core.attention (path fixture, NOT hardware) ===")
fake_kernel = mod.make_cte_stub(honour_prior_used_len=True)
fake_kernel.__module__ = "nkilib.core.attention"
pkg = types.ModuleType("nkilib")
core = types.ModuleType("nkilib.core")
att = types.ModuleType("nkilib.core.attention")
att.attention_cte = fake_kernel
core.attention = att
pkg.core = core
sys.modules.update({"nkilib": pkg, "nkilib.core": core, "nkilib.core.attention": att})
check("the shipped _load_kernel now resolves it, with no patch of any kind",
      mod.NeuronAttentionAdapter._load_kernel is mod._PRISTINE_LOAD_KERNEL
      and not mod.STUB_INSTALLS)

# Two shallow cells only: the comparison path is what is under test, and 1124-frame cells are
# ~2,250 sequential appends each. Coverage being partial is itself exercised, in case 9.
shallow = tempfile.mkdtemp()
for name in ("c5_floor_8f_sq1.pt", "c5_floor_8f_sq5.pt"):
    shutil.copy(os.path.join(opts.staged, name), os.path.join(shallow, name))

print("\n=== case 5: full path, no tolerance -> MEASURED_NOT_SCORED, exit 2 (never 0) ===")
code, art = arm(reference_tensors=shallow)
check("exit 2", code == 2, f"code={code} verdict={art['verdict']}")
check("verdict says it was not scored", art["verdict"] == "MEASURED_NOT_SCORED", art["verdict"])
check("no guard failed", failing(art) == [], str(failing(art)))
check("the kernel is recorded as under nkilib",
      "nkilib.core.attention" in art["kernel"]["modules_observed"],
      str(art["kernel"]["modules_observed"]))
check("both staged files verified against the committed sha256 manifest",
      len(next(r for r in art["results"]
               if r["check"] == "staged_reference_outputs_present_and_verified"
               )["files_verified"]) == 2)
check("two cells scored", art["cells_scored"] == 2, str(art["cells_scored"]))
check("the replay was bit-exact in both",
      all(c["replay_is_bit_exact"] for c in art["cells"]))
check("a real discrepancy was measured, and it is small",
      0.0 < art["worst_rel_err"] < 1e-4, f"{art['worst_rel_err']:.3e}")
check("not bit-equal -- the fused GPU reference and the tiled path must not coincide exactly",
      not any(c["bit_equal"] for c in art["cells"]))
check("cross_device is TRUE here, unlike the floor artifact", art["cross_device"] is True)
check("the dtype is the staged one and nothing cast it",
      art["input_dtypes"] == ["torch.float32"], str(art["input_dtypes"]))
check("it says why it was not scored", "not_scored_because" in art)
measured = art["worst_rel_err"]

print("\n=== case 6: a tolerance the run meets -> MEASURED 0; one it misses -> FAIL 1 ===")
code, art = arm(reference_tensors=shallow, tolerance=measured * 10)
check("exit 0", code == 0, f"code={code} verdict={art['verdict']}")
check("the tolerance guard passed and reports both numbers",
      next(r for r in art["results"]
           if r["check"] == "neuron_vs_staged_gpu_reference_within_tolerance")["pass"])
code, art = arm(reference_tensors=shallow, tolerance=measured / 10)
check("exit 1 on the tighter bar", code == 1, f"code={code} verdict={art['verdict']}")
check("the tolerance guard is the only failure",
      failing(art) == ["neuron_vs_staged_gpu_reference_within_tolerance"], str(failing(art)))

print("\n=== case 7: a tampered staged file must be rejected, not compared ===")
tampered = tempfile.mkdtemp()
victim = os.path.join(tampered, "c5_floor_8f_sq1.pt")
shutil.copy(os.path.join(shallow, "c5_floor_8f_sq1.pt"), victim)
mid = os.path.getsize(victim) // 2
with open(victim, "r+b") as fh:
    fh.seek(mid)
    flipped = bytes([fh.read(1)[0] ^ 0xFF])
    fh.seek(mid)
    fh.write(flipped)
code, art = arm(reference_tensors=tampered, tolerance=1e-3)
check("exit 1", code == 1, f"code={code} verdict={art['verdict']}")
check("the verification guard failed",
      "staged_reference_outputs_present_and_verified" in failing(art), str(failing(art)))
rej = next(r for r in art["results"]
           if r["check"] == "staged_reference_outputs_present_and_verified")["files_rejected"]
check("the tampered file is named with its actual and expected digests",
      len(rej) == 1 and rej[0]["why"] == "sha256 mismatch"
      and rej[0]["sha256"] != rej[0]["expected"], str(rej)[:160])
check("no rel_err was computed from it", "cells" not in art)

print("\n=== case 8: a kernel but nothing to compare against is the vacuous run -> FAIL ===")
code, art = arm(reference_tensors=None, tolerance=1e-3)
check("exit 1, not 0 and not 2", code == 1, f"code={code} verdict={art['verdict']}")
check("the verification guard failed with nothing verified",
      "staged_reference_outputs_present_and_verified" in failing(art), str(failing(art)))

print("\n=== case 9: partial coverage is recorded always, gated only on request ===")
code, art = arm(reference_tensors=shallow, tolerance=measured * 10)
check("exit 0 without the flag", code == 0, f"code={code}")
check("but full coverage is recorded as false, against the manifest's cell count",
      art["full_geometry_covered"] is False and art["cells_expected"] == 6,
      f"{art['full_geometry_covered']} {art['cells_scored']}/{art['cells_expected']}")
code, art = arm(reference_tensors=shallow, tolerance=measured * 10, require_full=True)
check("with --require-full-geometry the same run fails", code == 1, f"code={code}")
check("and the coverage guard is the failure",
      failing(art) == ["arm_geometry_fully_covered"], str(failing(art)))

print("\n=== case 10: an unreadable manifest is a failure, not a skipped check ===")
code, art = arm(reference_tensors=shallow, tolerance=1e-3,
                manifest=os.path.join(tempfile.mkdtemp(), "absent.json"))
check("exit 1", code == 1, f"code={code}")
v = next(r for r in art["results"]
         if r["check"] == "staged_reference_outputs_present_and_verified")
check("it records why the manifest could not be read",
      "FileNotFoundError" in (v["manifest_unreadable_because"] or ""),
      str(v["manifest_unreadable_because"]))
check("and rejects the files rather than trusting them",
      len(v["files_rejected"]) == 2 and all(f["why"] == "not in the manifest"
                                           for f in v["files_rejected"]))

print("\n=== case 11: the shipped source, read back ===")
src = open(HARNESS).read()
armsrc = src.split("def run_neuron_arm")[1].split("def main()")[0]
check("the source slice under test is non-empty", len(armsrc) > 2000, f"{len(armsrc)} chars")
check("--arm defaults to stub, so every existing invocation is untouched",
      'choices=("stub", "neuron"), default="stub"' in src)
check("--tolerance has no default value",
      '"--tolerance", type=float, default=None' in src)
check("the arm dispatches before the parity suite installs the stub",
      src.index('if args.arm == "neuron":')
      < src.index("install_cte_stub(honour_prior_used_len=bool(args.honour_prior_used_len))"))
check("the pristine method is captured at import, above install_cte_stub",
      src.index("_PRISTINE_LOAD_KERNEL = NeuronAttentionAdapter._load_kernel")
      < src.index("def make_cte_stub"))
# Structurally, over the parsed function body -- not by substring. The arm's guard messages
# necessarily talk ABOUT install_cte_stub and about assigning _load_kernel, so a substring test
# here is satisfied and broken by prose: the first draft of these three checks failed for exactly
# that reason, on text explaining the rule they were meant to enforce.
armfn = next(n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.FunctionDef) and n.name == "run_neuron_arm")
called = {n.func.id for n in ast.walk(armfn)
          if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
assigned = {t.attr for n in ast.walk(armfn) if isinstance(n, (ast.Assign, ast.AugAssign))
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
            if isinstance(t, ast.Attribute)}
check("the arm calls neither install_cte_stub nor make_cte_stub",
      not called & {"install_cte_stub", "make_cte_stub"}, str(sorted(called)))
check("the arm assigns no attribute named _load_kernel on anything",
      "_load_kernel" not in assigned, str(sorted(assigned)))
check("the tolerance comparison is against the caller's value, not a literal",
      "worst <= tol" in armsrc and "worst <= 1e" not in armsrc and "worst < 1e" not in armsrc)
check("the arm's only mention of 1e-4 is prose refusing it as a floor",
      armsrc.count("1e-4") == 1
      and "TOL=1e-4 above is a generous parity tolerance" in armsrc,
      str(armsrc.count("1e-4")))
check("the arm builds with check_env=True", "check_env=True" in armsrc)
check("and never with check_env=False", "check_env=False" not in armsrc)

set_env(False)
print(f"\n===== {'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)} "
      f"({len(FAILS)} failed)")
print("      Cases 5-9 used a fake nkilib and are NOT evidence of a Neuron run.\n")
raise SystemExit(1 if FAILS else 0)
