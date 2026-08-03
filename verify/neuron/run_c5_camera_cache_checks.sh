#!/usr/bin/env bash
# Full C5 camera-cache verification: baseline + four injected bugs.
#
# spec/neuron-port/design.md V5(a) and V6. See check_c5_camera_cache.py's docstring for what is
# being tested and why the reference is the real CameraCausalHead rather than a model of it.
#
# Unlike run_c4_wiring_checks.sh this script needs NO backup/restore trap: C5 is a standalone
# class and this pass edits nothing on the GPU path. attention.py:239-240 and camera_head.py
# stay byte-identical -- the reference is only a valid oracle while they do, which the suite's
# own `gpu_path_untouched_cat_still_present` check asserts.
#
# Usage:  bash verify/neuron/run_c5_camera_cache_checks.sh
# Runtime: ~1 minute on the desk A10G (no checkpoint load; the head is randomly initialised,
# which is sufficient because every gate here is about cache mechanics, not weights).

set -uo pipefail

PKG=/home/kwehden/lingbot-tier1/lingbot-map
DOCKER="docker exec -w $PKG lingbot-tier1 python verify/neuron/check_c5_camera_cache.py"

FAILED=0
expect() {  # expect <wanted-exit> <label> <args...>
    local want=$1 label=$2; shift 2
    echo "----- $label (expect exit $want)"
    # shellcheck disable=SC2086
    $DOCKER "$@" >"/tmp/c5_${label}.log" 2>&1
    local got=$?
    grep -E "^\[(PASS|FAIL|SKIP)\]|^===|INJECTED" "/tmp/c5_${label}.log" | cut -c1-160 || true
    if [ "$got" -ne "$want" ]; then
        echo "!!! $label: exit $got, wanted $want"
        FAILED=1
    fi
    # An injected pass must fail via a GATE, not via a traceback. Sizing MAX_TOTAL too small
    # once made two of these four die inside the replay loop before any comparison emitted,
    # which exits 1 for the wrong reason and proves nothing.
    if [ "$want" -eq 1 ] && grep -q "^Traceback" "/tmp/c5_${label}.log"; then
        echo "!!! $label: exited 1 via a TRACEBACK, not a failed check -- no discriminating power"
        FAILED=1
    fi
}

echo "===== PASS 1: baseline -- C5 vs the real CameraCausalHead ====="
expect 0 baseline

echo
echo "===== PASS 2: discriminating power -- every injected bug MUST fail ====="
# store_on_non_keyframe is the primary one for the comparison oracles: quiet at the frame of
# injection, diverging one frame later. The last three target gate classes the first four cannot
# reach -- the F6 shape[3] invariant, the graph-facing compute_attention path, and the capacity
# asymmetry -- each of which was measured to be UNGATED before its gate existed (the suite
# reported PASS with both shape[3] guards deleted, and had zero occurrences of
# 'compute_attention'). See inject_bug's docstring for the measured per-bug outcomes.
expect 1 inject_store_on_non_keyframe  --inject_bug store_on_non_keyframe
expect 1 inject_cursor_double_advance  --inject_bug cursor_double_advance
expect 1 inject_bf16_storage           --inject_bug bf16_storage
expect 1 inject_valid_len_is_capacity  --inject_bug valid_len_is_capacity
expect 1 inject_accept_shape3_gt_1     --inject_bug accept_shape3_gt_1
expect 1 inject_alloc_raw_frames       --inject_bug alloc_raw_max_total_frames
expect 1 inject_charge_cap_non_kf      --inject_bug charge_capacity_on_non_keyframe

echo
if [ "$FAILED" -eq 0 ]; then
    echo "===== C5 CAMERA CACHE: PASS (baseline bit-exact; all 7 injected bugs caught by gates)"
else
    echo "===== C5 CAMERA CACHE: FAIL -- see the !!! lines above"
fi
exit "$FAILED"
