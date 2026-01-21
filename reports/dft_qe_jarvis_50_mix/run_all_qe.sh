#!/usr/bin/env bash
set -euo pipefail
PW_CMD=${PW_CMD:-pw.x}
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
clean_jobdir() {
  local raw="$1"
  raw=${raw//$'\r'/}
  raw=${raw//\\//}
  raw=${raw#reports/dft_qe_jarvis_50_mix/}
  echo "$raw"
}
while IFS= read -r jobdir; do
  [ -z "$jobdir" ] && continue
  jobdir=$(clean_jobdir "$jobdir")
  [ -z "$jobdir" ] && continue
  echo "Running $jobdir"
  cd "$ROOT_DIR/$jobdir"
  $PW_CMD -in pw.in > pw.out 2> pw.err
done < "$ROOT_DIR/candidates.txt"
