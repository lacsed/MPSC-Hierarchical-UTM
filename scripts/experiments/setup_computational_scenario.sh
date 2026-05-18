#!/usr/bin/env bash
set -euo pipefail

BASELINE="${1:-proposed}"
UAVS="${2:-6}"
GRAPH_SIZE="${3:-current}"
DENSITY="${4:-medium}"
SEED="${5:-77}"
HORIZON="${6:-8}"
SCENARIO="${7:-computational}"

PROJECT="$HOME/MPSC-Hierarchical-UTM"
WORLD_DIR="$PROJECT/scripts/gz_world_out"

case "$GRAPH_SIZE" in
  small|minimo|minimum)
    VERTIPORTS=1
    STATIONS=1
    SUPPLIERS=1
    CLIENTS=2
    GRAPH_SIZE="small"
    ;;
  current|medio|medium)
    VERTIPORTS=1
    STATIONS=1
    SUPPLIERS=2
    CLIENTS=4
    GRAPH_SIZE="current"
    ;;
  large|grande)
    VERTIPORTS=2
    STATIONS=2
    SUPPLIERS=4
    CLIENTS=8
    GRAPH_SIZE="large"
    ;;
  *)
    echo "[ERROR] GRAPH_SIZE must be small/current/large"
    exit 1
    ;;
esac

case "$DENSITY" in
  low)
    TASKS=4
    TASK_SLEEP=2.0
    ;;
  medium)
    TASKS=8
    TASK_SLEEP=1.0
    ;;
  high)
    TASKS=20
    TASK_SLEEP=0.25
    ;;
  *)
    echo "[ERROR] DENSITY must be low/medium/high"
    exit 1
    ;;
esac

RUN_ID="${BASELINE}_N${UAVS}_${GRAPH_SIZE}_${DENSITY}_H${HORIZON}_seed${SEED}_${SCENARIO}"
RUN_DIR="$HOME/utm_runs/$RUN_ID"
mkdir -p "$RUN_DIR"

export UTM_BASELINE="$BASELINE"
export UTM_RUN_ID="$RUN_ID"
export UTM_RUN_DIR="$RUN_DIR"
export UTM_SCENARIO="$SCENARIO"
export UTM_GRAPH_SIZE="$GRAPH_SIZE"
export UTM_DENSITY="$DENSITY"
export UTM_SEED="$SEED"
export UTM_UAVS="$UAVS"
export UTM_NUM_TASKS="$TASKS"
export UTM_PLANNING_HORIZON="$HORIZON"
export UTM_WORK_TIME_S="2.0"
export UTM_CHARGE_TIME_S="5.0"
export UTM_CONTROL_RATE_HZ="8"
export UTM_TASK_SLEEP="$TASK_SLEEP"

cd "$PROJECT/scripts"
python3 gen_world_from_image_gz.py \
  "$UAVS" "$VERTIPORTS" "$STATIONS" "$SUPPLIERS" "$CLIENTS" \
  --map ./assets/finalmap.png \
  --out gz_world_out \
  --res 0.2 \
  --seed "$SEED"

cat > "$RUN_DIR/scenario.env" <<EOF
export UTM_BASELINE="$UTM_BASELINE"
export UTM_RUN_ID="$UTM_RUN_ID"
export UTM_RUN_DIR="$UTM_RUN_DIR"
export UTM_SCENARIO="$UTM_SCENARIO"
export UTM_GRAPH_SIZE="$UTM_GRAPH_SIZE"
export UTM_DENSITY="$UTM_DENSITY"
export UTM_SEED="$UTM_SEED"
export UTM_UAVS="$UTM_UAVS"
export UTM_NUM_TASKS="$UTM_NUM_TASKS"
export UTM_PLANNING_HORIZON="$UTM_PLANNING_HORIZON"
export UTM_WORK_TIME_S="$UTM_WORK_TIME_S"
export UTM_CHARGE_TIME_S="$UTM_CHARGE_TIME_S"
export UTM_CONTROL_RATE_HZ="$UTM_CONTROL_RATE_HZ"
export UTM_TASK_SLEEP="$UTM_TASK_SLEEP"
EOF

cat > "$RUN_DIR/task_list.txt" <<EOF
SUPPLIER_000,CLIENT_000
SUPPLIER_001,CLIENT_001
SUPPLIER_000,CLIENT_002
SUPPLIER_001,CLIENT_003
SUPPLIER_000,CLIENT_001
SUPPLIER_001,CLIENT_000
SUPPLIER_000,CLIENT_003
SUPPLIER_001,CLIENT_002
SUPPLIER_000,CLIENT_000
SUPPLIER_001,CLIENT_001
SUPPLIER_000,CLIENT_002
SUPPLIER_001,CLIENT_003
SUPPLIER_000,CLIENT_001
SUPPLIER_001,CLIENT_000
SUPPLIER_000,CLIENT_003
SUPPLIER_001,CLIENT_002
SUPPLIER_000,CLIENT_000
SUPPLIER_001,CLIENT_001
SUPPLIER_000,CLIENT_002
SUPPLIER_001,CLIENT_003
EOF

if [[ "$GRAPH_SIZE" == "small" ]]; then
  sed -i 's/SUPPLIER_001/SUPPLIER_000/g; s/CLIENT_002/CLIENT_000/g; s/CLIENT_003/CLIENT_001/g' "$RUN_DIR/task_list.txt"
fi

head -n "$TASKS" "$RUN_DIR/task_list.txt" > "$RUN_DIR/task_list.active.txt"

echo "[OK] Scenario generated"
echo "RUN_ID=$RUN_ID"
echo "RUN_DIR=$RUN_DIR"
echo "Use in every terminal: source $RUN_DIR/scenario.env"
