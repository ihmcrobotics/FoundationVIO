#!/usr/bin/env bash
# Project Aria BigRoom3 (download, extract, run, evaluate): pixi run example-aria [dataset_dir] [--max-iter N]
# Omit --max-iter for the whole sequence.
set -euo pipefail
DATA=data/aria
MAX_ITER=
args=("$@")
i=0
while [ "$i" -lt "${#args[@]}" ]; do
  case "${args[$i]}" in
    --max-iter)
      i=$((i + 1))
      MAX_ITER="${args[$i]}"
      ;;
    *) DATA="${args[$i]}" ;;
  esac
  i=$((i + 1))
done
SEQ=BigRoom3
SEQ_DIR="$DATA/${SEQ}_sequence"
nonempty() { [ -n "$(ls -A "$1" 2>/dev/null)" ]; }   # a failed step can leave an empty folder

if ! nonempty "$SEQ_DIR/images/camera_rgb"; then
  mkdir -p "$DATA"
  ZIP="$DATA/${SEQ}_sequence.zip"
  [ -e "$ZIP" ] || gdown 1-usS7bHyYeHCrmCNKW0Qufa9WwKUESJo -O "$ZIP"
  unzip -o "$ZIP" -d "$DATA" && rm -f "$ZIP"   # -o: overwrite a partial extract without prompting
fi

run=(python -m foundation_vio.run --config configs/aria.yaml --sequence "$SEQ" --set files.dataset_dir="$DATA")
[ -n "$MAX_ITER" ] && run+=(--max-iter "$MAX_ITER")
"${run[@]}" | tee "$DATA/$SEQ.log"
RUN=$(grep -m1 '^RESULT PATH: ' "$DATA/$SEQ.log" | cut -d' ' -f3-)
python scripts/evaluate.py "$RUN" "$SEQ_DIR/gt_poses_in_right_imu_frame_tum.txt" --gt-frame imu

echo
echo "View the result:"
echo "  pixi run view $RUN --ceiling 0.7"
echo "  pixi run replay $RUN --speed 4"
