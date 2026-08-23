"""config.py — Hằng số dùng chung. Chủ sở hữu: P1.
QUY TẮC: KHÓA sau Ngày 4. Đổi hằng số = phải trích lại toàn bộ feature.

🔒 KHÓA (2026-08-23) — P1 đã review, đủ điều kiện chốt sớm:
  - harness self-test PASS (report/data_processing_report.md § Kiểm chứng harness)
  - Baseline A: test(PUCPR)=0.00 đúng kỳ vọng -> split không leak vị trí
  - 8/8 shard feature trích xong không lỗi (report/data_processing_report.md § 7)
  Đổi bất kỳ hằng số nào dưới đây sau mốc này phải trích lại toàn bộ feature.
"""
from pathlib import Path

try:
    import google.colab  # noqa: F401
    ON_COLAB = True
except ImportError:
    ON_COLAB = False

# Colab: dữ liệu trên Drive. Local: dữ liệu ở raw/, processed/... ngay trong repo.
DRIVE = Path("/content/drive/MyDrive/pklot_project") if ON_COLAB else Path(__file__).resolve().parent.parent
RAW, PROC = DRIVE / "raw", DRIVE / "processed"
FEAT, MODELS, PREDS = DRIVE / "features", DRIVE / "models", DRIVE / "predictions"

# --- Ảnh & cửa sổ ---
# Đo trên pklot_data.slot_size_report() / perspective_report() — CHỈ trên
# TRAIN_LOTS+VAL_LOTS (UFPR04, UFPR05), không đụng PUCPR (test) để tránh leakage.
RESIZE_TO   = None      # giữ nguyên độ phân giải gốc — WINDOW_SIZE đo bằng pixel gốc,
                        # resize sẽ làm sai lệch; PUCPR có ô nhỏ tới ~32px, resize dễ mất chi tiết
WINDOW_SIZE = 96        # p50 = (97, 64.5)px (UFPR04+UFPR05) → làm tròn lên bội số của 16/8
                        # để chia hết cho STRIDE và ô HOG, đủ chứa cạnh rộng nhất trung vị
SCALES      = [0.5, 0.75, 1.0, 1.5, 2.0]  # perspective_report(): p90/p10 area = 3.3–3.45x
                        # (> ngưỡng 2.0, KE_HOACH.md §7) → BẮT BUỘC multi-scale
STRIDE      = 16
N_IMAGES    = None      # đã bỏ cap N_IMAGES=150 — dùng TOÀN BỘ ảnh sau temporal_subsample()
                        # (340 train / 29 val / 210 test = 579, xem report/data_processing_report.md § 4)

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
