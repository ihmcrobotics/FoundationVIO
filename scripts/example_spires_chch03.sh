#!/usr/bin/env bash
# Oxford Spires christ-church-03 (download, undistort, run, evaluate): pixi run example-spires [dataset_dir] [--max-iter N]
# Omit --max-iter for the whole sequence.
set -euo pipefail
DATA=data/oxford-spires
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
SEQ=2024-03-18-christ-church-03
SEQ_DIR="$DATA/$SEQ"
nonempty() { [ -n "$(ls -A "$1" 2>/dev/null)" ]; }   # a failed step can leave an empty folder

if [ ! -e "$SEQ_DIR/raw/imu.csv" ] || [ ! -e "$SEQ_DIR/processed/trajectory/gt-tum.txt" ] \
    || { ! nonempty "$SEQ_DIR/raw/cam0" && ! nonempty "$SEQ_DIR/raw/cam0_pinhole_f700" && [ ! -e "$SEQ_DIR/raw/images.zip" ]; }; then
  HF="$DATA/.hf-download"
  hf download ori-drs/oxford_spires_dataset --repo-type dataset --local-dir "$HF" \
    "sequences/$SEQ/raw/images.zip" "sequences/$SEQ/raw/imu.csv" \
    "sequences/$SEQ/processed/trajectory/gt-tum.txt"
  mkdir -p "$SEQ_DIR"
  cp -a "$HF/sequences/$SEQ/." "$SEQ_DIR/"
  rm -rf "$HF"
fi
if [ -e "$SEQ_DIR/raw/images.zip" ] && ! nonempty "$SEQ_DIR/raw/cam0" && ! nonempty "$SEQ_DIR/raw/cam0_pinhole_f700"; then
  unzip "$SEQ_DIR/raw/images.zip" -d "$SEQ_DIR/raw" && rm -f "$SEQ_DIR/raw/images.zip"
fi
[ -e "$SEQ_DIR/sensor.yaml" ] || \
  curl -L -o "$SEQ_DIR/sensor.yaml" https://raw.githubusercontent.com/ori-drs/oxford_spires_dataset/main/configs/sensor.yaml
nonempty "$SEQ_DIR/raw/cam0_pinhole_f700" && [ -e "$SEQ_DIR/sensor_pinhole_f700.yaml" ] || \
  python scripts/spires_undistort.py "$SEQ_DIR"

run=(python -m foundation_vio.run --config configs/spires.yaml --sequence christ-church-03 \
  --set files.dataset_dir="$DATA")
[ -n "$MAX_ITER" ] && run+=(--max-iter "$MAX_ITER")
"${run[@]}" | tee "$SEQ_DIR.log"
RUN=$(grep -m1 '^RESULT PATH: ' "$SEQ_DIR.log" | cut -d' ' -f3-)
python scripts/evaluate.py "$RUN" "$SEQ_DIR/processed/trajectory/gt-tum.txt" \
  --calib "$SEQ_DIR/sensor_pinhole_f700.yaml" --gt-frame base

echo
echo "View the result:"
echo "  pixi run view $RUN --ceiling 0.7"
echo "  pixi run replay $RUN --speed 4"
