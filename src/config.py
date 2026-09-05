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
# ⚠️ Ba hằng số này là HẬU XỬ LÝ (KE_HOACH.md §2: "code tự viết, không phải model"), KHÔNG phải
# hằng số trích feature. Đổi chúng KHÔNG làm mất giá trị 1,9 GB parquet — khác hẳn WINDOW_SIZE /
# SCALES / STRIDE ở trên. Khoá ở đầu file nhắm vào nhóm trên, không nhắm vào dòng này.
#
# NMS_IOU: 0.45 -> 0.20 (2026-09-04, P1). Quét trên val theo đúng KE_HOACH §7 Ngày 6-9
# ("quét ngưỡng + NMS" — phần NMS trước đó chưa từng làm).
#
#   Cửa sổ đa tỉ lệ ĐỒNG TÂM ở hai tỉ lệ liền nhau có IoU đúng bằng (48/72)^2 = (96/144)^2
#   = 0.4444 — nằm NGAY DƯỚI 0.45, nên NMS không gộp chúng và mỗi ô đỗ phát ra nhiều box trùng
#   ở các tỉ lệ khác nhau. 0.45 rơi trúng khe hở đó; lệch 0.01 là hành vi đảo ngược.
#
#   Đo trên val với Random Forest:
#       nms_iou   box   precision  recall     mAP
#         0.45   1534     0.362     0.686   0.2743   <- cũ, DƯỚI sàn Baseline A 0.5176
#         0.20    554     0.973     0.665   0.6782   <- mới, VƯỢT sàn +0.161
#   recall gần như không đổi -> 978 box bị loại toàn là BẢN SAO cùng một ô, không phải
#   phát hiện sai. baselines.py không dùng NMS nên sàn 0.5176 không đổi, phép so vẫn công bằng.
#
#   Quét 2 chiều: score_thr 0.34-0.40 x nms_iou 0.20-0.25 đều cho ~0.683; score_thr ảnh hưởng
#   rất ít, giữ 0.50 cho nhất quán với các số cũ. Chi tiết: report/model_pipeline_audit.md §5b.
IOU_EVAL, SCORE_THR, NMS_IOU = 0.5, 0.50, 0.20

# --- Chống leakage ---
TEMPORAL_MINUTES = 120
CROP_MARGIN = 0.06

# --- Split chính: theo bãi đỗ ---
TRAIN_LOTS, VAL_LOTS, TEST_LOTS = ["UFPR04", "UFPR05"], ["UFPR04"], ["PUCPR"]

RANDOM_SEED, N_JOBS = 42, -1
