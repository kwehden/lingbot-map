#!/usr/bin/env bash
# Detached preemption watcher. Started with `setsid nohup ... < /dev/null &` (TASK-N18
# constraint 3) because a watcher started any other way dies with the session that spawned it,
# and a watcher that dies records no preemption -- which is precisely the case REQ-092 exists for.
#
# Two jobs, and it does them in a SIDECAR file rather than in the run's own artifact:
#
#   * detect an interruption notice and write its cause, immediately, before the instance goes;
#   * heartbeat, so that a hard preemption with no notice at all still bounds the elapsed time
#     REQ-092 requires captured.
#
# Sidecar and not the main artifact because rec.py is read-modify-write: two writers racing on
# tier2_result.json can lose a stage's keys or leave a half-written document, and the one moment
# this script matters is the one moment the main script is also writing. Separate S3 objects
# cannot collide, and --collect reads both.
#
# The long wait lives INSIDE this detached script, never in a foreground command: TASK-N18
# constraint 2 caps a single command's sleep at 115 s because an SSM document that blocks past its
# own timeout is killed with its stage half-done. Each iteration here sleeps 20 s.
#
# Usage: spot_watch.sh <result.json> <s3-prefix> <log> <start-ts>
set -u

RESULT="$1"; S3="$2"; LOG="$3"; START_TS="$4"
SIDE="${LINGBOT_T2_SIDECAR:-/tmp/t2/interrupt.json}"
IMDS="http://169.254.169.254/latest"
POLL_S=20
HEARTBEAT_EVERY=6          # ~2 min, so an unnoticed hard preemption is bounded to ~2 min

log() { echo "[$(date -u +%H:%M:%S)] [watch] $*" >> "$LOG" 2>/dev/null || true; }

token() {
  curl -s -X PUT "$IMDS/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 300" --max-time 5 2>/dev/null
}

# Returns the notice body on stdout and 0 when the given metadata path answers 200; 1 otherwise.
# A 404 is the normal, healthy answer for both of these paths -- it means "no notice" and must not
# be mistaken for an error, or every poll would look like an interruption.
notice() {
  local tok="$1" path="$2" code
  code=$(curl -s -o /tmp/t2/notice.body -w '%{http_code}' \
    -H "X-aws-ec2-metadata-token: $tok" --max-time 5 "$IMDS/meta-data/$path" 2>/dev/null)
  if [ "$code" = "200" ]; then
    cat /tmp/t2/notice.body
    return 0
  fi
  return 1
}

write_side() {
  # $1 = cause, $2 = detail json-ish string
  local elapsed
  elapsed=$(( $(date +%s) - START_TS ))
  python3 /tmp/t2/rec.py "$SIDE" "cause" "$1"
  python3 /tmp/t2/rec.py "$SIDE" "detail" "$2"
  python3 /tmp/t2/rec.py "$SIDE" "elapsed_s" "$elapsed"
  python3 /tmp/t2/rec.py "$SIDE" "observed_utc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python3 /tmp/t2/rec.py "$SIDE" "note" \
    "REQ-092: an infrastructure-level interruption is inconclusive-infrastructure-interrupted, not a phase failure. Recorded by the detached watcher; the run itself may never have reached its complete stage."
  aws s3 cp "$SIDE" "$S3/tier2_interrupt.json" --only-show-errors 2>/dev/null || true
  aws s3 cp "$LOG" "$S3/tier2_result.log" --only-show-errors 2>/dev/null || true
}

log "watcher up (poll ${POLL_S}s, heartbeat every $((POLL_S * HEARTBEAT_EVERY))s)"
i=0
while true; do
  i=$((i + 1))
  TOK=$(token)
  if [ -n "$TOK" ]; then
    if BODY=$(notice "$TOK" "spot/instance-action"); then
      log "SPOT INTERRUPTION NOTICE: $BODY"
      write_side "spot_preemption" "$BODY"
      # Flush and stop. There are ~2 minutes; spending them re-polling would be the one way to
      # lose the record this script exists to write.
      exit 0
    fi
    if BODY=$(notice "$TOK" "events/recommendations/rebalance"); then
      # A rebalance recommendation is NOT a termination. Recorded as a warning so that a run which
      # later dies has this in its trail, and a run which finishes is not mislabelled interrupted.
      log "spot REBALANCE recommendation: $BODY"
      python3 /tmp/t2/rec.py "$SIDE" "rebalance_recommended_utc" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      aws s3 cp "$SIDE" "$S3/tier2_interrupt.json" --only-show-errors 2>/dev/null || true
    fi
  else
    log "WARN: no IMDS token this poll; cannot see an interruption notice"
  fi

  if [ $((i % HEARTBEAT_EVERY)) -eq 0 ]; then
    python3 /tmp/t2/rec.py "$SIDE" "heartbeat_elapsed_s" "$(( $(date +%s) - START_TS ))"
    python3 /tmp/t2/rec.py "$SIDE" "heartbeat_utc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    aws s3 cp "$SIDE" "$S3/tier2_interrupt.json" --only-show-errors 2>/dev/null || true
  fi
  sleep "$POLL_S"
done
