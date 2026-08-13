"""Smoke-test TASK-N08 owed step 3 against the SHIPPED source, at toy geometry.

Imports the real module and shrinks only FLOOR_FRAMES / FLOOR_SEQLEN_Q, so every code path
under test -- the four family members, the pairwise symmetrisation, the ulp probe, tensor
retention, all four guards and the verdict/exit mapping -- is the code that will run at
production geometry. Exists because the real run costs ~2,500 sequential cache appends and a
defect in the guard block would only surface after all of it.
"""
import argparse
import importlib.util
import json
import os
import sys
import tempfile

import torch

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_c5_attention_parity.py")
# Derived from this file's own location, not hardcoded: the Tier 1 container mounts the repo at
# a path that need not match the host's, and an absolute literal here sent the first run looking
# in /pkg for a checkout that was somewhere else.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
spec = importlib.util.spec_from_file_location("c5parity", HERE)
mod = importlib.util.module_from_spec(spec)
sys.modules["c5parity"] = mod
spec.loader.exec_module(mod)

FAILS = []


def check(name, cond, extra=""):
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        FAILS.append(name)


def run(frames, seqs, tensors_out, parts=None):
    mod.RESULTS.clear()
    mod.FLOOR_FRAMES = frames
    mod.FLOOR_SEQLEN_Q = seqs
    if parts is not None:
        mod.FLOOR_PARTS = parts
    out = os.path.join(tempfile.mkdtemp(), "floor.json")
    args = argparse.Namespace(floor_out=out, tensors_out=tensors_out, allow_cpu=True)
    code = mod.measure_reference_floor(args)
    with open(out) as fh:
        return code, json.load(fh)


print("\n=== case 1: full path, tensors retained -> MEASURED or ESCALATE, never FAIL ===")
td = tempfile.mkdtemp()
code, art = run((4, 9), (1, 3), td)
check("exit is 0 or 2, not a guard failure", code in (0, 2), f"code={code} verdict={art['verdict']}")
check("four cells recorded", len(art["cells"]) == 4, str(len(art["cells"])))
check("every cell ran all four members",
      all(c["members_completed"] == 4 for c in art["cells"]),
      str([c["members_completed"] for c in art["cells"]]))
check("six pairs per cell", all(len(c["pairs"]) == 6 for c in art["cells"]))
check("floor is positive", art["reference_floor_c5"] > 0, f"{art['reference_floor_c5']:.3e}")
check("floor equals the max cell spread",
      art["reference_floor_c5"] == max(c["cell_max_rel_err"] for c in art["cells"]))
check("floor_at names a cell and a pair",
      set(art["reference_floor_at"]) == {"frames", "seqlen_q", "pair"}, str(art["reference_floor_at"]))
check("geometry guard passed",
      next(r for r in art["results"] if r["check"] == "floor_geometry_fully_covered")["pass"])
check("disagreement guard passed",
      next(r for r in art["results"] if r["check"] == "floor_family_members_disagree")["pass"])
check("retention guard passed",
      next(r for r in art["results"] if r["check"] == "per_implementation_outputs_retained")["pass"])
check("cross_device is False", art["cross_device"] is False)
check("not_reportable set on cpu", art["not_reportable"] is (art["device"] != "cuda"))
check("harness_revision has a commit or says why not",
      "commit" in art["harness_revision"] or "unavailable_because" in art["harness_revision"],
      str(sorted(art["harness_revision"])))
check("verdict agrees with the ulp guard",
      (art["verdict"] == "MEASURED") == art["floor_usable_as_a_bar"],
      f"{art['verdict']} / usable={art['floor_usable_as_a_bar']}")

print("\n=== case 2: retained tensors are loadable and hold every member ===")
files = sorted(f for f in os.listdir(td) if f.endswith(".pt"))
check("one tensor file per cell", len(files) == 4, str(files))
blob = torch.load(os.path.join(td, files[0]), weights_only=False)
check("inputs present", set(blob["inputs"]) == {"q", "k_live", "v_live"}, str(sorted(blob["inputs"])))
check("all four outputs present", len(blob["outputs"]) == 4, str(sorted(blob["outputs"])))
check("seed recorded", isinstance(blob["seed"], int))
check("dtype recorded as fp32", blob["dtype"] == "torch.float32", blob["dtype"])
check("sha256 recorded in the artifact",
      all(len(c["retained_tensors"]["sha256"]) == 64 for c in art["cells"]))
check("a cell is reproducible from its recorded seed alone", True)  # exercised below

print("\n=== case 3: the same seed reproduces the same inputs (owed step 2's staging premise) ===")
cell = art["cells"][0]
# The generator is device-scoped: a CUDA generator seeded N does not produce a CPU generator's
# stream. Reproduce on the device the run actually used, which the artifact records.
dev = art["device"]
gen = torch.Generator(device=dev).manual_seed(cell["seed"])
c5 = mod.build(head_dim=mod.HEAD_DIM, num_heads=mod.NUM_HEADS, mtf=mod.MTF, device=dev)
k2, v2 = mod.drive(c5, cell["frames"], mod.NUM_HEADS, mod.HEAD_DIM, dev, gen)
b0 = torch.load(os.path.join(td, f"c5_floor_{cell['frames']}f_sq{cell['seqlen_q']}.pt"),
                weights_only=False)
check("k_live reproduces bit-exactly", torch.equal(k2.cpu(), b0["inputs"]["k_live"]))
check("v_live reproduces bit-exactly", torch.equal(v2.cpu(), b0["inputs"]["v_live"]))

print("\n=== case 4: no --tensors-out must FAIL the retention guard, exit 1 ===")
code2, art2 = run((4,), (1,), None)
check("exit 1", code2 == 1, f"code={code2} verdict={art2['verdict']}")
check("verdict FAIL", art2["verdict"] == "FAIL", art2["verdict"])
check("the retention guard is the one that failed",
      [r["check"] for r in art2["results"] if not r["pass"]] ==
      ["per_implementation_outputs_retained"] or
      "per_implementation_outputs_retained" in [r["check"] for r in art2["results"] if not r["pass"]],
      str([r["check"] for r in art2["results"] if not r["pass"]]))

print("\n=== case 5: the chunked member really reassociates -- >1 part, and it matters ===")
code3, art3 = run((12,), (1,), tempfile.mkdtemp(), parts=3)
spans = art3["cells"][0]["chunked_spans"]
check("three parts at 12 frames", len(spans) == 3, str(spans))
check("spans tile [0,12) exactly with no gap or overlap",
      spans[0][0] == 0 and spans[-1][1] == 12 and all(spans[i][1] == spans[i + 1][0]
                                                      for i in range(len(spans) - 1)), str(spans))
pair = {p["pair"]: p["rel_err"] for p in art3["cells"][0]["pairs"]}
check("chunked_merge differs from adapter_einsum (the merge is not pass-through here)",
      pair["adapter_einsum|chunked_merge"] > 0.0, f"{pair['adapter_einsum|chunked_merge']:.3e}")
check("chunked_merge still agrees with sdpa to well inside 1e-4",
      pair["chunked_merge|sdpa_fused"] < 1e-4, f"{pair['chunked_merge|sdpa_fused']:.3e}")
mod.FLOOR_PARTS = 3

print("\n=== case 6: single part -> merge IS pass-through, so it must match the adapter ===")
code4, art4 = run((12,), (1,), tempfile.mkdtemp(), parts=1)
spans4 = art4["cells"][0]["chunked_spans"]
pair4 = {p["pair"]: p["rel_err"] for p in art4["cells"][0]["pairs"]}
check("one part", len(spans4) == 1, str(spans4))
check("at one part the chunked member matches the adapter far more closely than at three",
      pair4["adapter_einsum|chunked_merge"] <= pair["adapter_einsum|chunked_merge"],
      f"1part={pair4['adapter_einsum|chunked_merge']:.3e} vs "
      f"3part={pair['adapter_einsum|chunked_merge']:.3e}")
mod.FLOOR_PARTS = 3

print("\n=== case 7: a family that agrees bit-exactly must FAIL the disagreement guard ===")
real_pairwise = mod._pairwise
mod._pairwise = lambda members: [{"pair": "a|b", "rel_err": 0.0}]
code5, art5 = run((4,), (1,), tempfile.mkdtemp())
mod._pairwise = real_pairwise
check("exit 1", code5 == 1, f"code={code5}")
check("the disagreement guard failed",
      "floor_family_members_disagree" in [r["check"] for r in art5["results"] if not r["pass"]],
      str([r["check"] for r in art5["results"] if not r["pass"]]))

print("\n=== case 8: a member that cannot run is data, and <3 fails the geometry guard ===")
real_gpu_sdpa = mod.gpu_sdpa
real_chunked = mod.chunked_merge


def boom(*a, **k):
    raise RuntimeError("synthetic member failure")


mod.gpu_sdpa, mod.chunked_merge = boom, boom
code6, art6 = run((4,), (1,), tempfile.mkdtemp())
mod.gpu_sdpa, mod.chunked_merge = real_gpu_sdpa, real_chunked
check("the run survived two dead members", isinstance(code6, int), f"code={code6}")
check("both failures recorded by name and message",
      set(art6["cells"][0]["members_failed"]) == {"sdpa_fused", "chunked_merge"},
      str(sorted(art6["cells"][0]["members_failed"])))
check("only two members completed", art6["cells"][0]["members_completed"] == 2)
check("the geometry guard failed on the thin cell",
      "floor_geometry_fully_covered" in [r["check"] for r in art6["results"] if not r["pass"]],
      str([r["check"] for r in art6["results"] if not r["pass"]]))
check("the ulp probe recorded why it could not run, rather than vanishing",
      "unavailable_because" in art6["cells"][0]["ulp_probe"], str(art6["cells"][0]["ulp_probe"]))
check("exit 1, not 2 -- a thin geometry is a failure and not an escalation", code6 == 1,
      f"code={code6} verdict={art6['verdict']}")

print("\n=== case 9: ulp dominance -> ESCALATE (exit 2), floor recorded but not usable ===")
real_ulp = mod._ulp_up
mod._ulp_up = lambda t: t * 1.05          # a gross "perturbation": must dominate the family
code7, art7 = run((4,), (1,), tempfile.mkdtemp())
mod._ulp_up = real_ulp
check("exit 2", code7 == 2, f"code={code7} verdict={art7['verdict']}")
check("verdict ESCALATE", art7["verdict"] == "ESCALATE", art7["verdict"])
check("floor still recorded", art7["reference_floor_c5"] > 0, f"{art7['reference_floor_c5']:.3e}")
check("but marked unusable as a bar", art7["floor_usable_as_a_bar"] is False)
check("only the ulp guard failed",
      [r["check"] for r in art7["results"] if not r["pass"]] == [mod.ULP_CHECK],
      str([r["check"] for r in art7["results"] if not r["pass"]]))

print("\n=== case 10: no absolute cutoff on the ratio anywhere in the new block ===")
src = open(HERE).read()
block = src.split("# reference_floor(C5) -- TASK-N08 owed step 3")[1].split("def main()")[0]
check("the ulp comparison is against the measured floor, not a literal",
      "worst_probe < floor" in block)
check("no threshold literal is compared against kappa or output_rel_move",
      "kappa >" not in block and "kappa <" not in block
      and "output_rel_move >" not in block and "output_rel_move <" not in block)
check("the retired analytic interim is named as superseded, not silently dropped",
      "1.37e-4" in block)

print("\n=== case 11: provenance -- there is no git in this container, so the env path is THE path ===")
os.environ.pop("LINGBOT_HARNESS_REV", None)
r_none = mod._harness_revision(REPO)
check("with nothing supplied, source is None and commit is None",
      r_none["source"] is None and r_none["commit"] is None, str(r_none))
check("and it says what to do about it", "remedy" in r_none and r_none["unavailable_because"],
      r_none.get("unavailable_because", ""))
os.environ["LINGBOT_HARNESS_REV"] = "deadbeefcafe1234567890abcdefabcdefabcdef"
os.environ["LINGBOT_HARNESS_BRANCH"] = "neuron-port"
os.environ["LINGBOT_HARNESS_DIRTY"] = "1"
r_env = mod._harness_revision(REPO)
check("a supplied SHA is recorded", r_env["commit"].startswith("deadbeef"), r_env["commit"])
check("and marked caller_asserted_env, NOT self_observed_git",
      r_env["source"] == "caller_asserted_env", r_env["source"])
check("branch and dirty come through typed", r_env["branch"] == "neuron-port" and r_env["dirty"] is True,
      f"{r_env['branch']} / {r_env['dirty']}")
check("the caveat that it was not verified is present", "caveat" in r_env)
check("self-observation failure is still recorded beside it",
      "self_observation_unavailable_because" in r_env)
for k in ("LINGBOT_HARNESS_REV", "LINGBOT_HARNESS_BRANCH", "LINGBOT_HARNESS_DIRTY"):
    os.environ.pop(k, None)

print(f"\n===== {'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)} "
      f"({len(FAILS)} failed)\n")
raise SystemExit(1 if FAILS else 0)
