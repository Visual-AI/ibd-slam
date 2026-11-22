#!/usr/bin/env bash
# Replica (NICE-SLAM release) and the culled GT meshes used for reconstruction metrics.
set -euo pipefail
DATA="${IBDSLAM_DATA:-$(dirname "$0")/../data}"
mkdir -p "$DATA" && cd "$DATA"
for f in data/Replica.zip cull_replica_mesh.zip; do
  wget -c "https://cvg-data.inf.ethz.ch/nice-slam/$f"
  unzip -q -n "$(basename "$f")"
  rm "$(basename "$f")"
done
