#!/usr/bin/env bash
set -euo pipefail

: "${QE_INPUT_GCS:?QE_INPUT_GCS is required}"
: "${QE_OUTPUT_GCS:?QE_OUTPUT_GCS is required}"

WORKDIR=${QE_WORKDIR:-/workspace}
JOB_DIR_NAME=${QE_JOB_DIR_NAME:-dft_qe_jarvis_50_mix}
JOBS=${JOBS:-8}
MPI_PROCS=${MPI_PROCS:-4}
PW_CMD=${PW_CMD:-pw.x}
STRUCT_TIMEOUT_SEC=${STRUCT_TIMEOUT_SEC:-21600}
PROGRESS_SYNC_SEC=${PROGRESS_SYNC_SEC:-300}

mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "Downloading inputs from ${QE_INPUT_GCS}..."
gsutil -m cp -r "${QE_INPUT_GCS}" .

if [ -d "${WORKDIR}/${JOB_DIR_NAME}" ]; then
  JOB_DIR="${WORKDIR}/${JOB_DIR_NAME}"
else
  JOB_DIR=$(find "$WORKDIR" -maxdepth 3 -name "candidates.txt" -print -quit | xargs -I{} dirname {})
fi

if [ -z "${JOB_DIR:-}" ] || [ ! -d "$JOB_DIR" ]; then
  echo "ERROR: Could not locate QE job directory." >&2
  exit 1
fi

echo "Using job directory: $JOB_DIR"

sed -i 's/\r$//' "$JOB_DIR"/candidates.txt "$JOB_DIR"/run_all_qe.sh "$JOB_DIR"/run_all_qe_parallel.sh || true
chmod +x "$JOB_DIR"/run_all_qe.sh "$JOB_DIR"/run_all_qe_parallel.sh

python3 "$JOB_DIR"/check_pseudos.py

cd "$JOB_DIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

PROGRESS_DIR="$JOB_DIR/_progress"
STATUS_DIR="$PROGRESS_DIR/status"
mkdir -p "$STATUS_DIR"

NORMALIZED_CANDIDATES="$PROGRESS_DIR/candidates.normalized.txt"
: > "$NORMALIZED_CANDIDATES"

while IFS= read -r raw_line || [ -n "$raw_line" ]; do
  line="${raw_line//$'\r'/}"
  line="${line//\\//}"
  line="${line#./}"
  if [[ "$line" == reports/* ]]; then
    line="${line#reports/}"
    line="${line#*/}"
  fi
  line="${line#${JOB_DIR_NAME}/}"
  line="${line#/}"

  if [ -n "$line" ] && [ -d "$JOB_DIR/$line" ]; then
    echo "$line" >> "$NORMALIZED_CANDIDATES"
  fi
done < "$JOB_DIR/candidates.txt"

TOTAL=$(wc -l < "$NORMALIZED_CANDIDATES" | tr -d ' ')
if [ "${TOTAL:-0}" -eq 0 ]; then
  echo "ERROR: No valid candidate directories were resolved from candidates.txt" >&2
  exit 1
fi

count_state() {
  local target="$1"
  local count=0
  local state_file
  while IFS= read -r -d '' state_file; do
    if grep -q "^state=${target}$" "$state_file"; then
      count=$((count + 1))
    fi
  done < <(find "$STATUS_DIR" -type f -name '*.status' -print0)
  echo "$count"
}

write_progress_snapshot() {
  local done_count failed_count timeout_count running_count completed_count pending_count percent active_pw now_utc
  done_count=$(count_state "done")
  failed_count=$(count_state "failed")
  timeout_count=$(count_state "timeout")
  running_count=$(count_state "running")
  completed_count=$((done_count + failed_count + timeout_count))
  pending_count=$((TOTAL - completed_count - running_count))
  if [ "$pending_count" -lt 0 ]; then
    pending_count=0
  fi
  percent=$((100 * completed_count / TOTAL))
  if command -v pgrep >/dev/null 2>&1; then
    active_pw=$(pgrep -fc 'pw.x -in pw.in' || true)
  elif command -v ps >/dev/null 2>&1; then
    active_pw=$(ps -ef | grep -c '[p]w.x -in pw.in' || true)
  else
    active_pw=0
  fi
  now_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

  cat > "$PROGRESS_DIR/progress.txt" <<EOF
timestamp_utc=${now_utc}
total=${TOTAL}
completed=${completed_count}
pending=${pending_count}
running=${running_count}
done=${done_count}
failed=${failed_count}
timeout=${timeout_count}
percent=${percent}
active_pw_processes=${active_pw}
jobs=${JOBS}
mpi_procs=${MPI_PROCS}
timeout_sec=${STRUCT_TIMEOUT_SEC}
EOF

  echo "[progress] ${completed_count}/${TOTAL} (${percent}%) done=${done_count} failed=${failed_count} timeout=${timeout_count} running=${running_count} pending=${pending_count} active_pw=${active_pw}"
}

sync_progress() {
  write_progress_snapshot
  gsutil -m rsync -r "$PROGRESS_DIR" "${QE_OUTPUT_GCS}/progress" >/dev/null 2>&1 || true
}

RUN_ONE_SCRIPT="$WORKDIR/run_one_qe_candidate.sh"
cat > "$RUN_ONE_SCRIPT" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

candidate_rel="$1"
candidate_dir="${JOB_DIR}/${candidate_rel}"
candidate_key="${candidate_rel//\//__}"
candidate_key="${candidate_key//[^A-Za-z0-9._-]/_}"
status_file="${STATUS_DIR}/${candidate_key}.status"

start_epoch=$(date -u +%s)
start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

cat > "$status_file" <<EOS
candidate_rel=${candidate_rel}
state=running
start_time_utc=${start_utc}
start_epoch=${start_epoch}
timeout_sec=${STRUCT_TIMEOUT_SEC}
EOS

if [ ! -d "$candidate_dir" ]; then
  end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  cat > "$status_file" <<EOS
candidate_rel=${candidate_rel}
state=failed
return_code=2
job_done=0
start_time_utc=${start_utc}
end_time_utc=${end_utc}
duration_sec=0
reason=missing_candidate_directory
EOS
  echo "[candidate] ${candidate_rel} state=failed rc=2 duration_sec=0 reason=missing_candidate_directory"
  exit 0
fi

cd "$candidate_dir"
set +e
if command -v timeout >/dev/null 2>&1; then
  timeout --signal=TERM --kill-after=120 "${STRUCT_TIMEOUT_SEC}s" \
    mpirun -np "$MPI_PROCS" "$PW_CMD" -in pw.in > pw.out 2> pw.err
  rc=$?
else
  mpirun -np "$MPI_PROCS" "$PW_CMD" -in pw.in > pw.out 2> pw.err
  rc=$?
fi
set -e

end_epoch=$(date -u +%s)
end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
duration_sec=$((end_epoch - start_epoch))

state="failed"
if [ "$rc" -eq 0 ]; then
  state="done"
elif [ "$rc" -eq 124 ]; then
  state="timeout"
fi

job_done=0
if grep -q "JOB DONE" pw.out 2>/dev/null; then
  job_done=1
fi

# Some MPI ranks can exit non-zero after QE finishes; treat explicit JOB DONE as success.
if [ "$job_done" -eq 1 ]; then
  state="done"
fi

cat > "$status_file" <<EOS
candidate_rel=${candidate_rel}
state=${state}
return_code=${rc}
job_done=${job_done}
start_time_utc=${start_utc}
end_time_utc=${end_utc}
duration_sec=${duration_sec}
EOS

printf "%s\t%s\t%s\t%s\t%s\n" \
  "$end_utc" "$candidate_rel" "$state" "$rc" "$duration_sec" >> "${PROGRESS_DIR}/events.tsv"
echo "[candidate] ${candidate_rel} state=${state} rc=${rc} duration_sec=${duration_sec} job_done=${job_done}"
exit 0
EOF
chmod +x "$RUN_ONE_SCRIPT"

echo "Starting QE run: JOBS=${JOBS} MPI_PROCS=${MPI_PROCS} PW_CMD=${PW_CMD} TIMEOUT=${STRUCT_TIMEOUT_SEC}s SYNC=${PROGRESS_SYNC_SEC}s"
echo "Tracking path: ${QE_OUTPUT_GCS}/progress"
echo "candidate_rel	state	return_code	job_done	start_time_utc	end_time_utc	duration_sec" > "$PROGRESS_DIR/status_schema.tsv"
echo -e "timestamp_utc\tcandidate_rel\tstate\treturn_code\tduration_sec" > "$PROGRESS_DIR/events.tsv"
sync_progress

(
  while true; do
    sleep "$PROGRESS_SYNC_SEC"
    sync_progress
  done
) &
SYNC_PID=$!

cleanup() {
  if [ -n "${SYNC_PID:-}" ] && kill -0 "$SYNC_PID" 2>/dev/null; then
    kill "$SYNC_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

export JOB_DIR STATUS_DIR PROGRESS_DIR MPI_PROCS PW_CMD STRUCT_TIMEOUT_SEC
export OMP_NUM_THREADS MKL_NUM_THREADS OMPI_ALLOW_RUN_AS_ROOT OMPI_ALLOW_RUN_AS_ROOT_CONFIRM
export RUN_ONE_SCRIPT
xargs -a "$NORMALIZED_CANDIDATES" -I{} -P "$JOBS" bash -lc '"$RUN_ONE_SCRIPT" "$1"' _ {}

sync_progress

echo "Uploading results to ${QE_OUTPUT_GCS}..."
gsutil -m rsync -r \
  -x '(^|/)(pseudos/|out/|.*\\.save/)|SSSP_.*\\.tar\\.gz$' \
  "$JOB_DIR" "${QE_OUTPUT_GCS}"

echo "Done."
