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
* **A green result has to be a discriminating one.** The two guards that make it so are the ones
  a reviewer cannot check by reading a number: bit-equality is a FAILURE (case 12 -- and the
  tolerance emit cannot catch it, since 0.0 meets every bar), and the input-side negative control
  must move the answer at least 100x further than the measurement did (case 13, where a kernel
  that ignores ``prior_used_len`` leaves the control sitting exactly on top of it).
* **Two guards, because there are two ways to be undiscriminating.** Whether the harness can SEE a
  broken mask is a property of the control; whether the caller's bar would REJECT one is a property
  of the bar. Folding them together made a generous tolerance fail the control guard on a correct
  run -- case 15 is that regression, and it requires the control guard to pass and the bar guard to
  be the single named failure.
* **Which kernel loaded is not which device ran.** Every guard above scores the kernel OBJECT, and
  a cpu-resolved trn2 run passed all of them while publishing a rel_err from a comparison that
  never touched Trainium. Case 16 is that defect reproduced at the desk, and it is the only case
  whose failing run also measures ~1e-7 and meets its bar: a number can be beautiful and void.
  Case 17 shows an explicit ``--device`` buys no provenance, 18 that a partial move is caught and
  named, 19 that with nothing scored the guard does not read PASS, 20 that both BLOCKED paths keep
  their exact guard counts, and 21 exercises the derivation's four branches without torch_xla.

**Cases 5-10 and 12-15 do NOT constitute a Neuron run and this file must never be cited as one.**
They register a fake ``nkilib.core.attention`` in ``sys.modules`` so that the real, unpatched
``_load_kernel`` import succeeds, which is the only way to reach the staged-reference comparison
path (sha256 verification, replay, rel_err, control, tolerance scoring, coverage) from the desk.
Since 2026-08-17 they also patch ``_dev`` to report ``xla:0``, because this desk is an A10G and
cuda is not Trainium -- the patch is on the OBSERVATION and never on ``NON_NEURON_DEVICE_TYPES``,
the threshold, which case 21 re-asserts as shipped. What they establish is that the path is correct
and its guards discriminate; what runs the arm for REQ-028 is trn2 hardware under ``TASK-N14a``,
where the kernel is real and both fixtures are absent.

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

# The device witnesses have to be faked for the same reason the kernel is: this desk is an A10G, so
# every tensor below genuinely lives on cuda, and cuda is not Trainium. The patch is on `_dev`, the
# OBSERVATION, and never on NON_NEURON_DEVICE_TYPES, the THRESHOLD -- widening the tuple would
# delete the guard these cases are meant to keep exercising. Case 16 runs the same fixture with the
# real `_dev` and is the case that refuses it; case 21 re-asserts the shipped tuple at the end.
REAL_DEV = mod._dev
mod._dev = lambda t: "xla:0"
check("the observation is patched, not the threshold: the shipped exclusion tuple is untouched",
      mod.NON_NEURON_DEVICE_TYPES == ("cpu", "cuda", "meta"),
      str(mod.NON_NEURON_DEVICE_TYPES))

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

print("\n=== case 12: a bit-exact result is a FAILURE, with or without a tolerance ===")
# The shape a comparison of two identical paths makes: hand back the staged reference itself.
# Historically that path is install_cte_stub(), and the tolerance emit cannot catch it -- 0.0
# satisfies every bar there is -- so the guard has to be inverted and has to be scored
# unconditionally. Only compute_attention is replaced; the control still goes through
# attend_groups, so the control guard stays green and the failure is unambiguous.
refs = {}
for name in sorted(os.listdir(shallow)):
    blob = torch.load(os.path.join(shallow, name), weights_only=False)
    refs[int(blob["seqlen_q"])] = blob["outputs"][mod.ARM_REFERENCE_MEMBER]


def echoing_build(head_dim, num_heads, mtf, device, check_env):
    cache, ad = real_build(head_dim, num_heads, mtf, device, check_env)
    cache.compute_attention = lambda i, j, q: refs[int(q.shape[2])].to(q.device)
    return cache, ad


mod.build_with_adapter = echoing_build
code, art = arm(reference_tensors=shallow, tolerance=measured * 10)
check("exit 1, not 0", code == 1, f"code={code} verdict={art['verdict']}")
check("the inverted guard is the only failure -- the tolerance guard passed on 0.0",
      failing(art) == ["neuron_and_gpu_do_not_agree_bit_exactly"], str(failing(art)))
check("worst_rel_err really was 0.0", art["worst_rel_err"] == 0.0, repr(art["worst_rel_err"]))
inverted = next(r for r in art["results"]
                if r["check"] == "neuron_and_gpu_do_not_agree_bit_exactly")
check("and it names every cell that agreed bit-exactly",
      len(inverted["cells_bit_equal"]) == 2, str(inverted["cells_bit_equal"]))
code, art = arm(reference_tensors=shallow)
check("scored with no --tolerance too: exit 1, not the unscored 2", code == 1,
      f"code={code} verdict={art['verdict']}")
check("and it is still the named failure there",
      failing(art) == ["neuron_and_gpu_do_not_agree_bit_exactly"], str(failing(art)))
check("with no --tolerance the bar guard is not emitted at all, rather than passing on a bar "
      "that does not exist",
      not [r for r in art["results"]
           if r["check"] == "the_tolerance_is_tight_enough_to_reject_a_broken_mask"],
      str([r["check"] for r in art["results"]]))
mod.build_with_adapter = real_build

print("\n=== case 13: a kernel that ignores prior_used_len -> the control cannot discriminate ===")
# The real kernel's version of the desk's honour_prior_used_len=False stub. The control moves
# only the INPUT valid_len, so a kernel that ignores it returns the identical answer and the
# control lands exactly on the measurement -- which is what "this gate discriminates nothing"
# looks like from the outside. The tolerance is deliberately loose enough that the parity gate
# itself passes, so the control guard has to be the thing that catches it -- and a bar that loose
# trips the SECOND guard as well, on its own separate cause: a 0.99 control inside a 10.0 bar means
# this bar would have forgiven the broken mask. Two defects, two names, one run.
att.attention_cte = mod.make_cte_stub(honour_prior_used_len=False)
att.attention_cte.__module__ = "nkilib.core.attention"
code, art = arm(reference_tensors=shallow, tolerance=10.0)
check("exit 1", code == 1, f"code={code} verdict={art['verdict']}")
check("both halves of the control property failed, each under its own name",
      set(failing(art)) == {"tail_masking_is_load_bearing_on_the_real_kernel",
                            "the_tolerance_is_tight_enough_to_reject_a_broken_mask"},
      str(failing(art)))
bar = next(r for r in art["results"]
           if r["check"] == "the_tolerance_is_tight_enough_to_reject_a_broken_mask")
check("and the bar guard names the cells 10.0 would have forgiven, with the bar it was given",
      len(bar["cells_the_bar_would_forgive"]) == 2 and bar["tolerance"] == 10.0,
      str(bar["cells_the_bar_would_forgive"])[:160])
control = next(r for r in art["results"]
               if r["check"] == "tail_masking_is_load_bearing_on_the_real_kernel")
check("it names every cell where the control did not move the answer",
      len(control["cells_that_did_not_discriminate"]) == 2,
      str(control["cells_that_did_not_discriminate"])[:200])
check("the control landed exactly on the measurement, which is the tell",
      all(c["control_rel_err"] == c["rel_err"] for c in art["cells"]),
      str([(f"{c['rel_err']:.3e}", f"{c['control_rel_err']:.3e}") for c in art["cells"]]))
check("and the parity gate it is protecting passed on the loose bar it was given",
      next(r for r in art["results"]
           if r["check"] == "neuron_vs_staged_gpu_reference_within_tolerance")["pass"])
att.attention_cte = fake_kernel
code, art = arm(reference_tensors=shallow, tolerance=measured * 10)
check("with the honouring kernel back, the control discriminates and the run passes again",
      code == 0 and failing(art) == [], f"code={code} {failing(art)}")
check("the control is recorded at every scored cell, ~0.99 at this depth against a ~1e-7 "
      "measurement",
      all(c["control_rel_err"] > 0.1 > c["rel_err"] for c in art["cells"]),
      str([(f"{c['rel_err']:.3e}", f"{c['control_rel_err']:.3e}") for c in art["cells"]]))
check("and its valid_len was the whole tile, not the live prefix",
      all(c["control_valid_len"] == 1152 for c in art["cells"]),
      str([c["control_valid_len"] for c in art["cells"]]))

print("\n=== case 14: the control at BOTH staged depths, and the shallow one first ===")
# The full staged set once -- ~3 s on the A10G. Every other case uses the shallow pair because
# a 1124-frame cell is ~2,250 sequential appends, but the deep control cannot be inferred from
# the shallow one: 28 padding rows of 1152 is a far weaker perturbation than 1,144, and if it
# came out anywhere near the measurement the guard shipped above would fail a CORRECT trn2 run
# at $8.60/hr. That is a fact to establish at the desk, not on the instance.
code, art = arm(reference_tensors=opts.staged, tolerance=measured * 100, require_full=True)
check("exit 0 over the full staged geometry", code == 0 and failing(art) == [],
      f"code={code} {failing(art)}")
check("cells ran shallow first, which sorted filenames are not",
      [c["frames"] for c in art["cells"]] == sorted(c["frames"] for c in art["cells"]),
      str([c["file"] for c in art["cells"]]))
deep = [c for c in art["cells"] if c["frames"] == mod.MTF]
shallowest = [c for c in art["cells"] if c["frames"] == 8]
check("the deep cells were controlled too, not just the shallow ones",
      len(deep) == 2 and all(c["control_rel_err"] is not None for c in deep), str(len(deep)))
floor = json.load(open(MANIFEST))["reference_floor_c5"]
check("and at 1124 frames the control still clears the guard's own 100x margin over the "
      "measurement, which is what stops it failing a correct run at depth",
      all(c["control_rel_err"] > 100 * c["rel_err"] for c in deep),
      str([f"{c['rel_err']:.3e} -> {c['control_rel_err']:.3e}" for c in deep]))
check("with orders of headroom over the measured reference floor besides",
      all(c["control_rel_err"] > 1000 * floor for c in deep),
      f"floor={floor:.3e} controls=" + str([f"{c['control_rel_err']:.3e}" for c in deep]))
check("the same breakage is ~50x weaker at depth than at 8 frames, which is why shallow runs "
      "first",
      min(c["control_rel_err"] for c in shallowest)
      > 20 * max(c["control_rel_err"] for c in deep),
      "shallow=" + str([f"{c['control_rel_err']:.3e}" for c in shallowest])
      + " deep=" + str([f"{c['control_rel_err']:.3e}" for c in deep]))

print("\n=== case 15: a GENEROUS bar must not fail the CONTROL guard -- why they are split ===")
# The regression the split exists for. While the tolerance clause lived inside the control guard,
# any bar at or above the deep control (~1.3e-2) failed
# `tail_masking_is_load_bearing_on_the_real_kernel` on a run whose own worst error was ~1.6e-06:
# a correct measurement rejected for having a GENEROUS bar, in the counter-intuitive direction,
# with nothing in the output naming the bar as the reason. The ceiling itself is real -- a bar above
# the control cannot tell a working mask from a broken one -- but it is a fact about TASK-N21's bar,
# so it fails under its own name and the port is not blamed for it.
code, art = arm(reference_tensors=opts.staged, tolerance=0.02)
check("the control guard PASSES: whether the mask is load-bearing does not depend on the bar",
      next(r for r in art["results"]
           if r["check"] == "tail_masking_is_load_bearing_on_the_real_kernel")["pass"],
      str(failing(art)))
check("the bar guard is the only failure, so the cause is named and it is not the port",
      code == 1 and failing(art) == ["the_tolerance_is_tight_enough_to_reject_a_broken_mask"],
      f"code={code} {failing(art)}")
check("the run itself measured correctly and inside the bar it was given",
      art["worst_rel_err"] < 1e-4
      and next(r for r in art["results"]
               if r["check"] == "neuron_vs_staged_gpu_reference_within_tolerance")["pass"],
      f"worst={art['worst_rel_err']:.3e} tol=0.02")
bar = next(r for r in art["results"]
           if r["check"] == "the_tolerance_is_tight_enough_to_reject_a_broken_mask")
check("and it reports the tightest control, which IS the ceiling on any bar N21 derives",
      float(bar["tightest_control"]) < 0.02
      and all(c["frames"] == mod.MTF for c in bar["cells_the_bar_would_forgive"]),
      f"tightest={bar['tightest_control']} forgiven="
      + str([c["frames"] for c in bar["cells_the_bar_would_forgive"]]))

# ---------------------------------------------------------------------------------------
# Cases 16-20: the device axis. The patch above comes OFF here, so these run against the
# real `_dev` on the real desk device -- which is the defect's own venue, one letter apart.
# ---------------------------------------------------------------------------------------
mod._dev = REAL_DEV

print("\n=== case 16: the reported defect, reproduced -- a real kernel on the wrong device ===")
# The whole finding in one artifact. Nothing is broken about this run except where it ran: the
# fake kernel agrees with the staged GPU reference to ~1e-7, the bar is met, every kernel-provenance
# guard passes -- and on a trn2 host the resolution that produced this device would have been
# `cpu`, silently, because there is no CUDA there. Before this guard the run exited 0 and the
# rel_err was quotable. The pairing below IS the defect: a number that agrees beautifully and is
# void anyway.
code, art = arm(reference_tensors=shallow, tolerance=measured * 10)
check("exit 1, not 0", code == 1, f"code={code} verdict={art['verdict']}")
check("the device guard is the SINGLE named failure -- nothing else about the run is wrong",
      failing(art) == ["compared_tensors_ran_on_a_neuron_device"], str(failing(art)))
check("and the parity gate it voids PASSED, on a bar the run genuinely met",
      art["worst_rel_err"] < 1e-4
      and next(r for r in art["results"]
               if r["check"] == "neuron_vs_staged_gpu_reference_within_tolerance")["pass"],
      f"worst={art['worst_rel_err']:.3e}")
dg = next(r for r in art["results"] if r["check"] == "compared_tensors_ran_on_a_neuron_device")
check("it names every cell and which witnesses disagreed",
      len(dg["cells_on_a_non_neuron_device"]) == 2
      and all(set(c["disagreeing"]) == {"input_device", "kv_device", "compute_device",
                                        "control_device"}
              for c in dg["cells_on_a_non_neuron_device"]),
      str(dg["cells_on_a_non_neuron_device"][0]["disagreeing"]))
check("the artifact stops claiming device provenance it does not have",
      art["neuron_device_observed"] is False and art["device_observed"],
      f"observed={art['device_observed']} resolution={art['device_resolution']}")
check("the run's diagnostics all survive the failure -- the cells, controls and rel_errs are "
      "still on disk, because the guard adds no early return",
      len(art["cells"]) == 2 and all(c["control_rel_err"] is not None for c in art["cells"]))
check("and the desk's honest resolution is recorded, not invented",
      art["device_resolution"] == ("cuda_visible" if CUDA_AT_IMPORT
                                   else "nothing_identified_the_venue"),
      art["device_resolution"])

print("\n=== case 17: an explicit --device is a caller assertion, and buys no provenance ===")
# Why the derivation alone would not have been enough. `--device cpu` on a trn2 host is the
# operator asserting a venue; the tensors' own .device is the venue observing itself. This arm
# already refuses a caller-supplied SHA as self_observed_git for the same reason.
code, art = arm(reference_tensors=shallow, tolerance=measured * 10, device="cpu")
check("exit 1 even though the caller named the device",
      code == 1, f"code={code} verdict={art['verdict']}")
check("the device guard is still the only failure",
      failing(art) == ["compared_tensors_ran_on_a_neuron_device"], str(failing(art)))
check("and the artifact separates what was asked for from what was observed",
      art["device_requested"] == "cpu" and art["device_resolution"] == "explicit"
      and art["device_observed"] == ["cpu"], str(art["device_observed"]))

print("\n=== case 18: a PARTIAL move is caught, and the disagreeing witness is named ===")
# The case a single witness misses: inputs staged to the device, output materialised on the host.
# It takes an input-side witness AND an output-side one to express it, and the artifact has to say
# WHICH one disagreed or an operator at $8.5964/hr cannot act on the failure. (Read "Only the
# three-witness design can express it" until 2026-08-17, when there were four -- and the count was
# never the point: two of the four corroborate rather than add a venue, per DEVICE_WITNESSES.)
# The fixture splits on dtype/dim, which is NOT a mixture the arm can physically produce; what it
# exercises is the artifact's blame-naming, not a reachable device topology.
mod._dev = lambda t: "cpu" if t.dtype == torch.float32 and t.dim() == 4 else "xla:0"
code, art = arm(reference_tensors=shallow, tolerance=measured * 10)
check("exit 1", code == 1, f"code={code} verdict={art['verdict']}")
dg = next(r for r in art["results"] if r["check"] == "compared_tensors_ran_on_a_neuron_device")
check("it fires on a mixture, not just on a uniformly wrong device",
      not dg["pass"] and all("xla:0" in c["witnesses"].values()
                             for c in dg["cells_on_a_non_neuron_device"]),
      str(dg["cells_on_a_non_neuron_device"][0]["witnesses"]))
check("and only the cpu-side witnesses are blamed",
      all(set(c["disagreeing"]) < {"input_device", "kv_device", "compute_device",
                                   "control_device"}
          for c in dg["cells_on_a_non_neuron_device"]),
      str(dg["cells_on_a_non_neuron_device"][0]["disagreeing"]))
mod._dev = lambda t: "xla:0"
code, art = arm(reference_tensors=shallow, tolerance=measured * 10)
check("with every witness on the device again, the guard passes and the run is green",
      code == 0 and failing(art) == [] and art["neuron_device_observed"] is True,
      f"code={code} {failing(art)}")

print("\n=== case 19: with nothing scored the device guard must not read PASS ===")
# The anti-vacuity direction every guard in this arm needs: a run with no cells has no evidence
# about the device either way, and a guard that passes on an absent input is the vacuity this whole
# file exists to close.
code, art = arm(reference_tensors=None, tolerance=1e-3)
check("exit 1, and the device guard is absent rather than passing on nothing",
      code == 1 and not [r for r in art["results"]
                         if r["check"] == "compared_tensors_ran_on_a_neuron_device"],
      str([r["check"] for r in art["results"]]))
tampered_only = tempfile.mkdtemp()
code, art = arm(reference_tensors=tampered_only, tolerance=1e-3)
check("same for an empty staged directory: no cells, no device claim",
      code == 1 and not [r for r in art["results"]
                         if r["check"] == "compared_tensors_ran_on_a_neuron_device"],
      f"code={code}")
# Both checks above return at the "no verified staged reference" early exit, so they establish that
# the guard is ABSENT and never that it reads False -- which left `bool(scored) and not offending`
# untested: dropping the conjunct kept this suite at ALL PASS (mutation run 2026-08-17, in place,
# 130/130 on the mutant). That is the same shape as the reverted-call-site mutation case 21 exists
# for, so it gets the same treatment: reach the emit with a VERIFIED reference and zero scored
# cells. A first trn2 run dying inside the loop on a kernel assert is the arm's own stated
# expectation, so this is the reachable route, not a contrived one.
real_replay = mod.replay
mod.replay = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("every cell dies in the loop"))
try:
    code, art = arm(reference_tensors=shallow, tolerance=measured * 10)
finally:
    mod.replay = real_replay
dg19 = [r for r in art["results"] if r["check"] == "compared_tensors_ran_on_a_neuron_device"]
check("with the reference VERIFIED and every cell dead, the guard is present and reads FAIL "
      "on zero evidence rather than passing because nothing disagreed",
      code == 1 and len(dg19) == 1 and dg19[0]["pass"] is False
      and dg19[0]["witnesses_per_cell"] == []
      and art["neuron_device_observed"] is False,
      f"code={code} present={len(dg19)} "
      f"{dg19[0]['pass'] if dg19 else None} obs={art.get('neuron_device_observed')}")

print("\n=== case 20: the two BLOCKED paths are untouched, which is the placement tripwire ===")
# The guard sits after all five early returns. If it ever migrates above them, these counts move
# and the desk says so -- rather than trn2 saying it at $8.5964/hr. Case 1 already pins the
# env-unset path at 2; this pins the other one, which the reported defect's fix could have broken.
mod._dev = REAL_DEV
saved = dict(sys.modules)
for name in ("nkilib", "nkilib.core", "nkilib.core.attention"):
    sys.modules.pop(name, None)
code, art = arm(reference_tensors=shallow, tolerance=measured * 10)
check("no Neuron stack still BLOCKS at exit 3, not a device failure",
      code == 3 and art["blocked_because"] == "neuron_kernel_unavailable",
      f"code={code} {art.get('blocked_because')}")
check("exactly the four provenance guards ran: nothing new leaked above the import",
      art["n_checks"] == 4 and failing(art) == [], f"{art['n_checks']} checks {failing(art)}")
check("and the device guard is not among them",
      not [r for r in art["results"] if r["check"] == "compared_tensors_ran_on_a_neuron_device"])
check("the blocked artifact still records how the device WOULD have been resolved",
      "device_resolution" in art, str(art.get("device_resolution")))
set_env(False)
code, art = arm()
check("env unset still BLOCKS with exactly two guards, as case 1 requires",
      code == 3 and art["n_checks"] == 2
      and art["blocked_because"] == "neuron_env_not_configured",
      f"code={code} {art['n_checks']} checks")
set_env(True)
sys.modules.update(saved)

print("\n=== case 21: the derivation, all four branches, without torch_xla installed ===")
# The half of the fix this desk could otherwise never exercise. `_resolve_arm_device` is pure and
# total precisely so its xla branch is testable here: torch_xla is installed nowhere in this
# project, so a derivation that reached for it inside the function would be desk-unprovable and
# would first run on the instance that costs money.
xm_ok = types.SimpleNamespace(xla_device=lambda: torch.device("xla:0"))
xm_bad = types.SimpleNamespace(xla_device=lambda: (_ for _ in ()).throw(RuntimeError("no plugin")))
check("an explicit request wins, and is labelled as the caller's",
      mod._resolve_arm_device("trn2:0", True, xm_ok, "NEURON") == ("trn2:0", "explicit"))
check("with a live runtime and PJRT_DEVICE=NEURON it ASKS instead of guessing",
      mod._resolve_arm_device(None, False, xm_ok, "NEURON") == ("xla:0", "xla_runtime"))
check("and returns a STRING -- a torch.device would raise while writing the artifact, after "
      "the measurement",
      isinstance(mod._resolve_arm_device(None, False, xm_ok, "NEURON")[0], str))
dev, why = mod._resolve_arm_device(None, False, xm_bad, "NEURON")
check("a runtime present but unusable falls back and SAYS so, rather than raising",
      dev == "cpu" and why.startswith("xla_runtime_present_but_failed"), f"{dev} {why}")
check("no torch_xla: the desk resolves cuda exactly as before, so both BLOCKED paths are safe",
      mod._resolve_arm_device(None, True, None, "NEURON") == ("cuda", "cuda_visible"))
check("and the trn2 defect's own shape is now NAMED rather than silent",
      mod._resolve_arm_device(None, False, None, "NEURON")
      == ("cpu", "nothing_identified_the_venue"))
check("_xla_runtime never raises on a host without torch_xla",
      mod._xla_runtime() is None)
check("the exclusion tuple shipped is still cpu/cuda/meta after every patch above",
      mod.NON_NEURON_DEVICE_TYPES == ("cpu", "cuda", "meta")
      and mod._dev is REAL_DEV, str(mod.NON_NEURON_DEVICE_TYPES))
armsrc2 = open(HARNESS).read().split("def run_neuron_arm")[1].split("def main()")[0]
check("the guard scores the tensors' devices, not the argument",
      "_device_type(c[w]) in NON_NEURON_DEVICE_TYPES" in armsrc2
      and "compared_tensors_ran_on_a_neuron_device" in armsrc2)
check("and it is an exclusion, with no allowlist literal naming one acceptable Trainium string",
      '== "xla"' not in armsrc2 and 'type == "trn' not in armsrc2)
check("the device witnesses are captured before the .cpu() that erases them",
      armsrc2.index('row["compute_device"]') < armsrc2.index('got.detach().cpu().float()'))
# A unit-tested helper the production path does not call is precisely the vacuity this file exists
# to catch, and it was reachable here: reverting the resolution at the top of the arm to the old
# `args.device or ("cuda" if cuda else "cpu")` left every check above green, because the checks
# exercised the helper directly and nothing pinned the call site. Found by mutation, 2026-08-17.
armfn2 = next(n for n in ast.walk(ast.parse(open(HARNESS).read()))
              if isinstance(n, ast.FunctionDef) and n.name == "run_neuron_arm")
armcalls = {n.func.id for n in ast.walk(armfn2)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
check("the arm RESOLVES THROUGH the helper rather than inlining the rule",
      {"_resolve_arm_device", "_xla_runtime"} <= armcalls, str(sorted(armcalls)))
check("and the silent cpu default is gone from the arm's own source",
      'or ("cuda" if torch.cuda.is_available() else "cpu")' not in armsrc2)
check("the witnesses are read through the seam, so patching _dev really does reach the guard",
      armsrc2.count("_dev(") == len(mod.DEVICE_WITNESSES), str(armsrc2.count("_dev(")))

set_env(False)
print(f"\n===== {'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)} "
      f"({len(FAILS)} failed)")
print("      Cases 5-10 and 12-15 used a fake nkilib AND a patched _dev reporting xla:0, and are\n"
      "      NOT evidence of a Neuron run. Cases 16-21 run against the real _dev on this desk.\n")
raise SystemExit(1 if FAILS else 0)
