"""baselines.py — Baseline A — đoán theo vị trí
Chủ sở hữu: P1
"""
import pandas as pd

import config
import evaluate_pklot
from pklot_data import UNKNOWN_OCCUPIED


def _with_epoch(gt):
    """Gắn 'chữ ký kỳ camera' cho mỗi ảnh — dùng vị trí ô slot_id NHỎ NHẤT còn rõ nhãn trong CÙNG
    ảnh đó làm mốc. Cần thiết vì UFPR04 có camera bị xê dịch vị trí 2 lần trong quá trình ghi hình
    (lệch tới ~94px giữa các kỳ, đo bằng cách so khớp tâm ô giữa các ngày cách xa nhau — UFPR05/
    PUCPR xê dịch đúng 0px suốt toàn bộ thời gian, tức camera THẬT SỰ cố định ở 2 bãi đó — xem
    report/data_processing_report.md mục 2/4). Nếu memorize theo (lot, slot_id) mà không phân biệt
    kỳ, vị trí ô nhớ được từ kỳ này sẽ bị áp nhầm sang ảnh chụp ở kỳ khác, cho dự đoán sai vị trí
    một cách có hệ thống — không phải do leakage/không leakage mà do lỗi này."""
    ref_id = gt.groupby("lot")["slot_id"].transform("min")
    ref = gt[(gt.slot_id == ref_id) & (gt.label != UNKNOWN_OCCUPIED)]
    ref = ref.drop_duplicates(subset="image_id")[["image_id", "x_min", "y_min"]]
    ref = ref.rename(columns={"x_min": "epoch_x", "y_min": "epoch_y"})
    return gt.merge(ref, on="image_id", how="left")


def _memorize(train_gt):
    """Với mỗi (lot, kỳ camera, slot_id), nhớ box + nhãn đa số từ TRAIN. Không nhìn ảnh.
    Loại nhãn UNKNOWN_OCCUPIED trước khi lấy đa số — nếu không, giá trị -1 sẽ kéo lệch trung bình."""
    train_gt = _with_epoch(train_gt)
    known = train_gt[train_gt.label != UNKNOWN_OCCUPIED]
    memo = known.groupby(["lot", "epoch_x", "epoch_y", "slot_id"]).agg(
        x_min=("x_min", "first"), y_min=("y_min", "first"),
        x_max=("x_max", "first"), y_max=("y_max", "first"),
        label=("label", lambda s: int(s.mean() >= 0.5)),
    )
    return memo


def predict(split_gt, memo):
    """Lặp lại y nguyên box+nhãn đã nhớ cho mỗi ảnh trong split_gt — không đụng ảnh thật.
    Bỏ qua vị trí mà chính ảnh đó đánh dấu UNKNOWN_OCCUPIED — harness coi đó là vùng ignore,
    đoán bừa vào đó chỉ làm lệch free_slots_MAE/occupancy_MAE_pp một cách giả tạo. Chỉ đoán khi
    ảnh thuộc ĐÚNG kỳ camera đã thấy trong TRAIN (xem _with_epoch) — kỳ camera lạ thì không có gì
    để nhớ, thà không đoán còn hơn đoán bừa sang vị trí của kỳ khác."""
    split_gt = _with_epoch(split_gt)
    rows = []
    for r in split_gt.itertuples():
        if r.label == UNKNOWN_OCCUPIED:
            continue
        key = (r.lot, r.epoch_x, r.epoch_y, r.slot_id)
        if key not in memo.index:
            continue  # ô không xuất hiện trong TRAIN, hoặc ảnh thuộc kỳ camera TRAIN chưa từng thấy
        m = memo.loc[key]
        rows.append([r.image_id, m.x_min, m.y_min, m.x_max, m.y_max, m.label, 1.0])
    return pd.DataFrame(rows, columns=evaluate_pklot.CSV_COLS)


def run_all(gt_path=None):
    """Chạy Baseline A trên VAL và TEST. TEST phải ~0 nếu split theo bãi đúng."""
    gt_path = gt_path or (config.PROC / "gt.csv")
    gt = pd.read_csv(gt_path)
    memo = _memorize(gt[gt.split == "train"])

    results = {}
    for split_name in ("val", "test"):
        split_gt = gt[gt.split == split_name]
        pred = predict(split_gt, memo)
        results[split_name] = evaluate_pklot.evaluate(split_gt, pred)
    return results


def self_test():
    """Chống tái phát bug: camera 1 bãi xê dịch vị trí giữa 2 'kỳ' -> nếu memorize không phân biệt
    kỳ, vị trí nhớ từ kỳ A bị áp nhầm sang ảnh kỳ B, cho dự đoán sai vị trí một cách hệ thống (đã
    phát hiện thật ở UFPR04 — xem report/data_processing_report.md)."""
    cols = ["image_id", "lot", "slot_id", "split", "x_min", "y_min", "x_max", "y_max", "label"]
    rows = [
        # kỳ A: 2 ảnh train, ô 1 tại (0,0,50,50), ô 2 tại (100,0,150,50) — cả 2 đều "có xe" (1)
        ["a1", "L", 1, "train", 0, 0, 50, 50, 1], ["a1", "L", 2, "train", 100, 0, 150, 50, 1],
        ["a2", "L", 1, "train", 0, 0, 50, 50, 1], ["a2", "L", 2, "train", 100, 0, 150, 50, 1],
        # kỳ B: 1 ảnh val, CÙNG lot nhưng camera xê dịch — ô 1 tại (500,500,550,550), ô 2 tại (600,500,650,550)
        ["b1", "L", 1, "val", 500, 500, 550, 550, 1], ["b1", "L", 2, "val", 600, 500, 650, 550, 1],
    ]
    gt = pd.DataFrame(rows, columns=cols)
    gt["score"] = 1.0

    memo = _memorize(gt[gt.split == "train"])
    pred = predict(gt[gt.split == "val"], memo)
    assert len(pred) == 0, (
        f"BUG tái phát: ảnh kỳ B (val) không có trong TRAIN nhưng vẫn bị đoán {len(pred)} box — "
        "chắc chắn đang dùng vị trí kỳ A áp nhầm sang kỳ B. pred:\n" + str(pred)
    )
    print("baselines.self_test PASS: đã chặn đúng dự đoán sang kỳ camera lạ")
    return True


if __name__ == "__main__":
    import json
    self_test()
    print(json.dumps(run_all(), indent=2))
