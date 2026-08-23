#!/usr/bin/env bash
# =====================================================================
# setup_project.sh — Tạo cây thư mục dự án PKLot
#
# Chạy MỘT LẦN, bởi P1 (Tech Lead), rồi commit lên Git.
#
#   bash setup_project.sh
#
# An toàn khi chạy lại: chỉ tạo thứ còn thiếu, không ghi đè file đã có.
# =====================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

blue()  { printf '\033[34m%s\033[0m\n' "$1"; }
green() { printf '\033[32m  ✓ %s\033[0m\n' "$1"; }
dim()   { printf '\033[2m  · %s\033[0m\n' "$1"; }

# Tạo file nếu chưa có; không bao giờ ghi đè
new() {
  if [ -e "$1" ]; then dim "đã có: $1"; else
    mkdir -p "$(dirname "$1")"; cat > "$1"; green "tạo: $1"
  fi
}

blue "Tạo cây thư mục..."
mkdir -p src notebooks scripts report/figures processed
green "src/ notebooks/ scripts/ report/figures/ processed/"

# ---------------------------------------------------------------------
# src/ — khung file, mỗi file ghi rõ chủ sở hữu
# ---------------------------------------------------------------------
blue $'\nTạo khung file src/...'

new src/config.py <<'PY'
"""config.py — Hằng số dùng chung. Chủ sở hữu: P1.
QUY TẮC: KHÓA sau Ngày 4. Đổi hằng số = phải trích lại toàn bộ feature.
"""
from pathlib import Path

DRIVE = Path("/content/drive/MyDrive/pklot_project")
RAW, PROC = DRIVE / "raw", DRIVE / "processed"
FEAT, MODELS, PREDS = DRIVE / "features", DRIVE / "models", DRIVE / "predictions"

# --- Ảnh & cửa sổ ---
RESIZE_TO   = None      # ⚠️ P2 chốt Ngày 1
WINDOW_SIZE = None      # ⚠️ P2 chốt Ngày 1 (theo p50 kích thước ô)
SCALES      = [1.0]     # ⚠️ P2 chốt Ngày 1 (perspective_report)
STRIDE      = 16
N_IMAGES    = 150       # 90 train / 30 val / 30 test

# --- Gán nhãn ---
IOU_POSITIVE, IOU_IGNORE = 0.5, 0.3
NEG_SAMPLE_RATE = 0.10

# --- Đánh giá & hậu xử lý ---
IOU_EVAL, SCORE_THR, NMS_IOU = 0.5, 0.50, 0.45

# --- Chống leakage ---
TEMPORAL_MINUTES = 120
CROP_MARGIN = 0.06

# --- Split chính: theo bãi đỗ ---
TRAIN_LOTS, VAL_LOTS, TEST_LOTS = ["UFPR04", "UFPR05"], ["UFPR04"], ["PUCPR"]

RANDOM_SEED, N_JOBS = 42, -1
PY

for f in "pklot_data.py|P2|Parse XML, cắt vùng có nhãn, lấy mẫu thời gian" \
         "build_dataset.py|P2|Sinh feature theo shard, lưu parquet" \
         "windows.py|P3|Sinh cửa sổ trượt, gán nhãn IoU 3 lớp" \
         "features.py|P3|Trích 395 feature: HOG, LBP, color, texture" \
         "evaluate.py|P1|Hàm average precision dùng chung" \
         "evaluate_pklot.py|P1|Harness 3 tầng: định vị, đếm, tỉ lệ lấp đầy" \
         "baselines.py|P1|Baseline A — đoán theo vị trí" \
         "detect.py|P1+P4|NMS, quét ngưỡng, sinh file dự đoán" \
         "train_model.py|P4|DT -> RF -> mining -> LightGBM"; do
  IFS='|' read -r name owner desc <<< "$f"
  new "src/$name" <<PY
"""$name — $desc
Chủ sở hữu: $owner
"""
PY
done

new src/__init__.py <<'PY'
PY

# ---------------------------------------------------------------------
# notebooks/ — mỗi người một file để tránh conflict
# ---------------------------------------------------------------------
blue $'\nTạo notebook khung...'

nb() {
  new "notebooks/$1" <<JSON
{
 "cells": [
  {"cell_type": "markdown", "metadata": {},
   "source": ["# $2\n", "\n", "Chủ sở hữu: **$3**\n"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [],
   "source": [
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "\n",
    "!pip -q install pyarrow mahotas lightgbm optuna shap imagehash\n",
    "!git clone -q https://github.com/NHOM/pklot-project.git /content/code || (cd /content/code && git pull -q)\n",
    "\n",
    "import sys; sys.path.insert(0, '/content/code/src')\n",
    "import config, pklot_data, windows, features, evaluate_pklot\n",
    "print('sẵn sàng')"
   ]}
 ],
 "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
              "language_info": {"name": "python"}},
 "nbformat": 4, "nbformat_minor": 4
}
JSON
}

nb 00_setup.ipynb            "Ngày 1-4 — Dựng nền"                "cả nhóm"
nb p1_analysis.ipynb         "P1 — Sweep ngưỡng, SHAP, tổng hợp"  "P1 Tech Lead"
nb p2_data_check.ipynb       "P2 — Kiểm định dữ liệu, split"      "P2 Data Engineer"
nb p3_features.ipynb         "P3 — Feature, tốc độ, ablation"     "P3 Pipeline Engineer"
nb p4_models.ipynb           "P4 — DT, RF, mining, LightGBM"      "P4 Model Engineer"

# ---------------------------------------------------------------------
# scripts/ & report/
# ---------------------------------------------------------------------
blue $'\nTạo script và khung báo cáo...'

new scripts/download_data.sh <<'SH'
#!/usr/bin/env bash
# Tải PKLot (~2GB) về Google Drive. Chỉ P2 chạy, một lần.
set -euo pipefail
DEST="${1:-/content/drive/MyDrive/pklot_project/raw}"
mkdir -p "$DEST" && cd "$DEST"

if [ -d PKLot ]; then echo "Đã có PKLot/ — bỏ qua."; exit 0; fi

echo "Đang tải PKLot (~2GB, 15-40 phút)..."
wget -c http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz
tar -xzf PKLot.tar.gz && rm -f PKLot.tar.gz

echo "Kiểm tra:"
find PKLot -name '*.jpg' | wc -l | xargs echo "  ảnh:"
find PKLot -name '*.xml' | wc -l | xargs echo "  nhãn:"
SH
chmod +x scripts/download_data.sh 2>/dev/null || true

new results.csv <<'CSV'
date,person,experiment,split,mAP_macro,AP_occupied,AP_empty,free_slots_MAE,occupancy_MAE_pp,train_time,notes
CSV

new report/BAO_CAO.md <<'MD'
# Phát hiện và Đếm Chỗ Đỗ Xe — Báo cáo

1. Bài toán & output — *P1*
2. PKLot & 4 đặc thù — *P2*
3. Chống leakage + Baseline A — *P2*
4. Pipeline & feature — *P3*
5. Ablation feature — *P3*
6. Tiến hóa model: DT → RF → mining — *P4*
7. SHAP & phân tích lỗi — *P4*
8. Test set, domain shift, hướng Module 4 — *P1*
MD

blue $'\nXong.'
cat <<'MSG'

Bước tiếp theo:
  1. git add -A && git commit -m "khởi tạo dự án" && git push
  2. P2 chạy: bash scripts/download_data.sh
  3. P1 điền WINDOW_SIZE / SCALES vào config.py sau khi P2 báo cáo Ngày 1
MSG
