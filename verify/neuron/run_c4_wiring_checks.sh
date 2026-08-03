#!/usr/bin/env bash
# Full C4 wiring verification: two regimes x (baseline, compare) + three injected bugs.
#
# spec/neuron-port/design.md C4. See check_c4_wiring.py's docstring for what is being tested
# and why two regimes are required. This script exists because the protocol is easy to get
# subtly wrong by hand: the baseline must be recorded against PRE-C4 code, and the C4 edits
# must come back afterwards even if a run dies partway.
#
# Restores via a file copy rather than `git stash`, so an interrupted run cannot leave the C4
# edits buried on the stash stack. The restore is in a trap, so ^C restores too.
#
# Usage:  bash verify/neuron/run_c4_wiring_checks.sh
# Runtime: ~8 minutes on the desk A10G (7 model loads of a 4.6 GB checkpoint dominate).

set -uo pipefail

PKG=/home/kwehden/lingbot-tier1/lingbot-map
TARGET="$PKG/lingbot_map/models/gct_stream_window.py"
BACKUP=/tmp/gct_stream_window.C4.py
DOCKER="docker exec -w $PKG lingbot-tier1 python verify/neuron/check_c4_wiring.py"

# Regime 1: default. Flow term decides; is_first_streaming_frame is MASKED here.
R1="--output_dir /tmp/c4_r1 --baseline_dir /tmp/c4_r1"
# Regime 2: flow suppressed (threshold 1e9). Only the first-frame and gap terms can decide,
# which is the only way the is_first_streaming_frame wiring gets covered at all.
R2="--flow_threshold 1e9 --n_frames 12 --output_dir /tmp/c4_r2 --baseline_dir /tmp/c4_r2"

cd "$PKG" || exit 1
if ! git diff --quiet -- "$TARGET"; then
    echo "== backing up C4 edits to $BACKUP"
    cp "$TARGET" "$BACKUP"
else
    echo "!! $TARGET has no uncommitted changes."
    echo "!! If C4 is already committed, check out the pre-C4 revision by hand instead;"
    echo "!! this script's baseline pass relies on 'git checkout --' reaching pre-C4 code."
    exit 2
fi

restore() { echo "== restoring C4 edits"; cp "$BACKUP" "$TARGET"; }
trap restore EXIT

FAILED=0
expect() {  # expect <wanted-exit> <label> <args...>
    local want=$1 label=$2; shift 2
    echo "----- $label (expect exit $want)"
    # shellcheck disable=SC2086
    $DOCKER "$@" >"/tmp/c4_${label}.log" 2>&1
    local got=$?
    grep -vE "it/s\]|^$" "/tmp/c4_${label}.log" | grep -E "^\[(PASS|FAIL|SKIP)\]|^===|INJECTED|mix:" || true
    if [ "$got" -ne "$want" ]; then
        echo "!!! $label: exit $got, wanted $want"
        FAILED=1
    fi
}

echo "===== PASS 1: baseline against PRE-C4 code ====="
git checkout -- "$TARGET"
if grep -q "keyframe_decision(" "$TARGET"; then
    echo "!!! pre-C4 checkout still references keyframe_decision -- C4 may be committed."
    exit 2
fi
# shellcheck disable=SC2086
expect 0 r1_baseline --mode baseline $R1
# shellcheck disable=SC2086
expect 0 r2_baseline --mode baseline $R2

echo
echo "===== PASS 2: compare against POST-C4 code ====="
restore
grep -c "keyframe_decision(" "$TARGET" | xargs echo "keyframe_decision call sites (expect 2):"
# shellcheck disable=SC2086
expect 0 r1_compare --mode compare $R1
# shellcheck disable=SC2086
expect 0 r2_compare --mode compare $R2

echo
echo "===== PASS 3: discriminating power -- every injected bug MUST fail ====="
# gap_off_by_one and threshold are detectable in regime 1; first_frame only in regime 2.
# shellcheck disable=SC2086
expect 1 r1_inject_gap    --mode compare --inject_bug gap_off_by_one $R1 --output_dir /tmp/c4_r1_gap
# shellcheck disable=SC2086
expect 1 r1_inject_thresh --mode compare --inject_bug threshold      $R1 --output_dir /tmp/c4_r1_thr
# shellcheck disable=SC2086
expect 1 r2_inject_first  --mode compare --inject_bug first_frame    $R2 --output_dir /tmp/c4_r2_ff

echo
if [ "$FAILED" -eq 0 ]; then
    echo "===== C4 WIRING: PASS (both regimes bit-exact; all 3 injected bugs detected)"
else
    echo "===== C4 WIRING: FAIL -- see the !!! lines above"
fi
exit "$FAILED"
