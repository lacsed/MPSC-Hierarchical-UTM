#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-computational_matrix_commands.txt}"
: > "$OUT"

BASELINES=(proposed greedy_distance sct_only no_utm)
UAVS=(2 4 6)
GRAPHS=(small current)
DENSITIES=(low high)
HORIZONS=(8 10)
SEEDS=(1 2 3)

for baseline in "${BASELINES[@]}"; do
  for n in "${UAVS[@]}"; do
    for graph in "${GRAPHS[@]}"; do
      for density in "${DENSITIES[@]}"; do
        for H in "${HORIZONS[@]}"; do
          for seed in "${SEEDS[@]}"; do
            echo "source ~/MPSC-Hierarchical-UTM/scripts/experiments/setup_computational_scenario.sh $baseline $n $graph $density $seed $H computational" >> "$OUT"
          done
        done
      done
    done
  done
done

echo "[OK] Wrote $(wc -l < "$OUT") scenario-generation commands to $OUT"
