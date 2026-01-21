#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
JOBS=${JOBS:-4}
MPI_PROCS=${MPI_PROCS:-2}
PW_CMD=${PW_CMD:-pw.x}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MPI_PROCS PW_CMD

cat "$ROOT_DIR/candidates.txt" \
  | tr -d '\r' \
  | sed 's#\\#/#g' \
  | sed 's#^reports/dft_qe_jarvis_50_mix/##' \
  | awk 'NF' \
  | xargs -I{} -P "$JOBS" bash -lc 'cd "$1" && mpirun -np "$MPI_PROCS" "$PW_CMD" -in pw.in > pw.out 2> pw.err' _ "$ROOT_DIR/{}"
