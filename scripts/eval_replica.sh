#!/usr/bin/env bash
# Run IBD-SLAM + reconstruction evaluation on the 8 Replica scenes and print a summary table.
# Scenes are distributed over all visible GPUs; finished scenes are skipped when re-run.
#
#   bash scripts/eval_replica.sh [CKPT] [OUT]
#   CKPT defaults to pretrained/ibdslam_scannet.pth (use `ibrnet` for the untrained baseline)
set -uo pipefail
cd "$(dirname "$0")/.."
CKPT="${1:-pretrained/ibdslam_scannet.pth}"
OUT="${2:-outputs/replica}"
PY="${PY:-python}"
PER_GPU="${PER_GPU:-1}"
SCENES="${SCENES:-room0 room1 room2 office0 office1 office2 office3 office4}"

NGPU=$(nvidia-smi -L | wc -l)
NSLOT=$((NGPU * PER_GPU))
mkdir -p "$OUT"

run_one() {
  local scene=$1 gpu=$2 dir="$OUT/replica_$1"
  mkdir -p "$dir"
  if [[ ! -f "$dir/slam_result.json" ]]; then
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/run_slam.py --seq "replica/$scene" --ckpt "$CKPT" --out "$dir" \
      > "$dir/run.log" 2>&1 || { echo "$scene: run_slam failed, see $dir/run.log"; return; }
  fi
  if [[ ! -f "$dir/recon_result_d8.json" ]]; then
    "$PY" scripts/eval_recon.py --run "$dir" --scene "$scene" > "$dir/recon.log" 2>&1 \
      || { echo "$scene: eval_recon failed, see $dir/recon.log"; return; }
  fi
  echo "$scene: done"
}

i=0
for scene in $SCENES; do
  run_one "$scene" $((i % NGPU)) &
  i=$((i + 1))
  while (( $(jobs -rp | wc -l) >= NSLOT )); do wait -n; done
done
wait
"$PY" scripts/summarize.py "$OUT" | tee "$OUT/summary.md"
