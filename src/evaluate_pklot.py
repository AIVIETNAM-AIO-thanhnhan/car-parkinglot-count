"""evaluate_pklot.py — Harness 3 tầng: định vị, đếm, tỉ lệ lấp đầy
Chủ sở hữu: P1
"""
import numpy as np
import pandas as pd

import config
from evaluate import average_precision
from pklot_data import UNKNOWN_OCCUPIED

CSV_COLS = ["image_id", "x_min", "y_min", "x_max", "y_max", "label", "score"]


def _load(df_or_path):
    df = pd.read_csv(df_or_path) if not isinstance(df_or_path, pd.DataFrame) else df_or_path.copy()
    if "score" not in df.columns:
        df["score"] = 1.0  # ground truth không có score -> coi như chắc chắn 1.0
    return df


def _by_image(df, label):
    sub = df[df["label"] == label]
    out = {}
    for img, g in sub.groupby("image_id"):
        out[img] = list(zip(g.x_min, g.y_min, g.x_max, g.y_max, g.score))
    return out


def _by_image_boxes(df, label):
    sub = df[df["label"] == label]
    out = {}
    for img, g in sub.groupby("image_id"):
        out[img] = list(zip(g.x_min, g.y_min, g.x_max, g.y_max))
    return out


def evaluate(gt, pred, iou_threshold=None):
    """gt, pred: đường dẫn CSV hoặc DataFrame theo Quy ước README (image_id,x_min,y_min,x_max,y_max,label,[score]).
    Trả về dict: AP_occupied, AP_empty, mAP_macro, free_slots_MAE, occupancy_MAE_pp.

    gt có thể chứa nhãn UNKNOWN_OCCUPIED (-1) — ô có vị trí thật nhưng không rõ trạng thái (XML gốc
    thiếu occupied). Các ô này KHÔNG tính vào AP (dùng làm vùng ignore — prediction trùng vị trí
    không bị phạt oan là false positive) và KHÔNG tính vào free_slots/occupancy (không rõ free hay
    occupied thì không đưa vào mẫu số).
    """
    iou_threshold = iou_threshold or config.IOU_EVAL
    gt_df, pred_df = _load(gt), _load(pred)
    ignore_by_image = _by_image_boxes(gt_df, UNKNOWN_OCCUPIED)

    # --- Tầng 1: định vị + phân loại (AP mỗi lớp, mAP = trung bình) ---
    ap_occupied = average_precision(_by_image_boxes(gt_df, 1), _by_image(pred_df, 1), iou_threshold, ignore_by_image)
    ap_empty = average_precision(_by_image_boxes(gt_df, 0), _by_image(pred_df, 0), iou_threshold, ignore_by_image)
    map_macro = (ap_occupied + ap_empty) / 2

    # --- Tầng 2 + 3: đếm số ô trống & tỉ lệ lấp đầy, theo từng ảnh (loại ô UNKNOWN_OCCUPIED) ---
    gt_known = gt_df[gt_df.label != UNKNOWN_OCCUPIED]
    pred_known = pred_df[pred_df.label != UNKNOWN_OCCUPIED]
    images = sorted(set(gt_known.image_id) | set(pred_known.image_id))
    free_errs, occ_errs = [], []
    for img in images:
        g = gt_known[gt_known.image_id == img]
        p = pred_known[pred_known.image_id == img]
        g_free, g_total = (g.label == 0).sum(), len(g)
        p_free, p_total = (p.label == 0).sum(), len(p)
        free_errs.append(abs(int(p_free) - int(g_free)))
        g_occ = 1 - g_free / g_total if g_total else 0.0
        p_occ = 1 - p_free / p_total if p_total else 0.0
        occ_errs.append(abs(p_occ - g_occ) * 100)

    return {
        "AP_occupied": ap_occupied,
        "AP_empty": ap_empty,
        "mAP_macro": map_macro,
        "free_slots_MAE": float(np.mean(free_errs)) if free_errs else 0.0,
        "occupancy_MAE_pp": float(np.mean(occ_errs)) if occ_errs else 0.0,
    }


def self_test():
    """Cổng Ngày 2: nộp ground truth làm 'dự đoán' -> phải ra mAP = 1.0, MAE = 0."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    rows = []
    for img_i in range(5):
        for box_i in range(6):
            x1, y1 = int(rng.integers(0, 500)), int(rng.integers(0, 500))
            w, h = int(rng.integers(20, 80)), int(rng.integers(20, 80))
            label = int(box_i % 2 == 0)
            rows.append([f"img{img_i}", x1, y1, x1 + w, y1 + h, label, 1.0])
    gt = pd.DataFrame(rows, columns=CSV_COLS)

    result = evaluate(gt, gt)
    assert result["mAP_macro"] == 1.0, result
    assert result["AP_occupied"] == 1.0 and result["AP_empty"] == 1.0, result
    assert result["free_slots_MAE"] == 0.0, result
    assert result["occupancy_MAE_pp"] == 0.0, result

    # Dự đoán rỗng -> AP phải rơi về 0, không phải lỗi hay 1.0 giả.
    empty_pred = pd.DataFrame(columns=CSV_COLS)
    zero_result = evaluate(gt, empty_pred)
    assert zero_result["mAP_macro"] == 0.0, zero_result

    # Ô UNKNOWN_OCCUPIED (-1): prediction đúng vị trí đó KHÔNG được tính là false positive,
    # và KHÔNG được lẫn vào free_slots/occupancy. Nếu bug tái xuất hiện, mAP sẽ tụt khỏi 1.0.
    from pklot_data import UNKNOWN_OCCUPIED
    gt_with_unknown = pd.concat([gt, pd.DataFrame(
        [["imgU", 900, 900, 950, 950, UNKNOWN_OCCUPIED, 1.0]], columns=CSV_COLS)], ignore_index=True)
    pred_guess_at_unknown = pd.concat([gt, pd.DataFrame(
        [["imgU", 900, 900, 950, 950, 1, 0.99]], columns=CSV_COLS)], ignore_index=True)
    unknown_result = evaluate(gt_with_unknown, pred_guess_at_unknown)
    assert unknown_result["mAP_macro"] == 1.0, unknown_result

    # Vùng ignore LỚN hơn hẳn 1 prediction (VD margin region PUCPR): 1 prediction nhỏ nằm TRỌN
    # trong đó vẫn phải bị loại — nếu code lỡ dùng lại IoU đối xứng thay vì containment, IoU sẽ nhỏ
    # giả tạo (~0.1-0.2, dưới IOU_EVAL) và prediction sẽ bị tính nhầm thành false positive.
    big_ignore_region = [1000, 1000, 1300, 1100]  # 300x100, lớn hơn hẳn 1 box 50x50 bên trong
    gt_with_big_ignore = pd.concat([gt, pd.DataFrame(
        [["imgB"] + big_ignore_region + [UNKNOWN_OCCUPIED, 1.0]], columns=CSV_COLS)], ignore_index=True)
    pred_inside_big_ignore = pd.concat([gt, pd.DataFrame(
        [["imgB", 1100, 1030, 1150, 1080, 1, 0.99]], columns=CSV_COLS)], ignore_index=True)
    big_ignore_result = evaluate(gt_with_big_ignore, pred_inside_big_ignore)
    assert big_ignore_result["mAP_macro"] == 1.0, big_ignore_result
    assert unknown_result["free_slots_MAE"] == 0.0, unknown_result
    assert zero_result["free_slots_MAE"] > 0, zero_result

    print("self_test PASS:", result)
    return True


if __name__ == "__main__":
    self_test()
