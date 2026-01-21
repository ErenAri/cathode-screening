#!/usr/bin/env bash
#SBATCH --job-name=qe_jarvis
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-49

set -euo pipefail
PW_CMD=${PW_CMD:-pw.x}
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
jobdir=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$ROOT_DIR/candidates.txt")
if [ -z "$jobdir" ]; then
  echo "No jobdir found for index $SLURM_ARRAY_TASK_ID"
  exit 1
fi
cd "$ROOT_DIR/$jobdir"
$PW_CMD -in pw.in > pw.out 2> pw.err
