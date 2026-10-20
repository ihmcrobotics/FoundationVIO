#!/usr/bin/env bash
# ScaRF-SLAM r05 (download, extract, undistort, run, evaluate): pixi run example-scarf [dataset_dir] [--max-iter N]
# Omit --max-iter for the whole sequence.
set -euo pipefail
DATA=data/scarf
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
SEQ=r05
SEQ_DIR="$DATA/$SEQ/${SEQ}_bag_sequence"
nonempty() { [ -n "$(ls -A "$1" 2>/dev/null)" ]; }   # a failed step can leave an empty folder

mkdir -p "$DATA/config"
[ -e "$DATA/config/rectified_intrinsic.yaml" ] || \
  gdown 1jEE2wDQ_972w6FEAgqukpfVgONgV5vOw -O "$DATA/config/rectified_intrinsic.yaml"
[ -e "$DATA/config/kalibr_imucam_chain.yaml" ] || \
  gdown 1rn8ROYIH7R8Ssqw6FE4DLH03ueJTS0xX -O "$DATA/config/kalibr_imucam_chain.yaml"
[ -e "$DATA/config/kalibr_imu_chain.yaml" ] || \
  gdown 1aIFSpVWAEmAHFQLsh5qSkofPfXclwLbW -O "$DATA/config/kalibr_imu_chain.yaml"
if [ ! -e "$DATA/$SEQ/${SEQ}_bag" ]; then
  ZIP="$DATA/${SEQ}.zip"
  [ -e "$ZIP" ] || gdown 1nOc6ns0wl-Uu_6m0allAPUXac8XYIrhk -O "$ZIP"
  unzip "$ZIP" -d "$DATA" && rm -f "$ZIP"
fi

nonempty "$SEQ_DIR/images/camera_rgb" || python scripts/bag_to_folder.py "$DATA/$SEQ/${SEQ}_bag" \
  --image-topic /insta/cam0/image_raw/compressed --imu-topic /insta/imu/data_raw
nonempty "$SEQ_DIR/images/camera_rgb_undistorted" || python scripts/scarf_undistort.py "$DATA" "$SEQ"

run=(python -m foundation_vio.run --config configs/scarf.yaml --sequence "$SEQ" --set files.dataset_dir="$DATA")
[ -n "$MAX_ITER" ] && run+=(--max-iter "$MAX_ITER")
"${run[@]}" | tee "$DATA/$SEQ.log"
RUN=$(grep -m1 '^RESULT PATH: ' "$DATA/$SEQ.log" | cut -d' ' -f3-)
python scripts/evaluate.py "$RUN" "$DATA/$SEQ/${SEQ}_gt/poses_gt.txt" --calib "$DATA/config/kalibr_imucam_chain.yaml"

echo
echo "View the result:"
echo "  pixi run view $RUN --ceiling 0.7"
echo "  pixi run replay $RUN --speed 4"
