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
