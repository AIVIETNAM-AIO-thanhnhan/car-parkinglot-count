"""detect.py — cửa sổ + xác suất  ->  box cuối cùng.  Chủ sở hữu: P1+P4

    xác suất mỗi cửa sổ  ->  lọc ngưỡng  ->  NMS theo (ảnh, lớp)  ->  CSV dự đoán

Đây là 3 bước "code tự viết" trong KE_HOACH.md §2 — KHÔNG phải model. Model chỉ làm bước
"cửa sổ -> xác suất"; mọi thứ trong file này là hậu xử lý tất định.

Nhãn dùng chung toàn dự án (khớp windows.py): 0 = ô trống, 1 = có xe, 2 = nền.
Chỉ lớp 0 và 1 được sinh ra box — lớp 2 (nền) bị loại, không bao giờ vào file dự đoán.
"""
import numpy as np
import pandas as pd

import config
import evaluate_pklot

EMPTY, OCCUPIED, BACKGROUND = 0, 1, 2
OBJECT_LABELS = (EMPTY, OCCUPIED)


def nms(boxes, scores, iou_thr=None):
    """Greedy non-maximum suppression. boxes:(N,4) x_min,y_min,x_max,y_max — trả về chỉ số giữ lại.

    Cần thiết vì STRIDE=16 < WINDOW_SIZE=96: mỗi ô đỗ thật bị nhiều cửa sổ chồng lên cùng bắt
    được. Không có NMS thì 1 ô sinh ra hàng chục box -> precision sụp, free_slots_MAE vô nghĩa.
    """
    iou_thr = config.NMS_IOU if iou_thr is None else iou_thr
    if len(boxes) == 0:
        return np.empty(0, dtype=int)
    boxes = np.asarray(boxes, dtype=np.float64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.asarray(scores, dtype=np.float64).argsort()[::-1]

    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest]); iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest]); iy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=int)


def build_predictions(meta, proba, classes, score_thr=None, nms_iou=None, bg_veto=True):
    """(cửa sổ + xác suất) -> DataFrame đúng Quy ước README (evaluate_pklot.CSV_COLS).

    meta:    DataFrame có image_id, x_min, y_min, x_max, y_max — cùng thứ tự dòng với proba
    proba:   (N, C) xác suất từ clf.predict_proba
    classes: clf.classes_ — ánh xạ cột của proba sang nhãn 0/1/2
    bg_veto: cửa sổ mà lớp NỀN thắng argmax thì không sinh box cho bất kỳ lớp nào

    Mỗi lớp đối tượng được lọc ngưỡng và NMS ĐỘC LẬP (chuẩn detection): một cửa sổ về lý thuyết
    có thể sinh box cho cả 2 lớp, NMS trong từng lớp sẽ tự dọn. Không dùng argmax vì argmax bỏ
    mất score — mà AP cần score để xếp hạng.

    🔴 bg_veto: trước đây cột xác suất lớp NỀN bị bỏ qua HOÀN TOÀN — lớp nền bị loại khỏi ĐẦU RA
    nhưng cũng bị loại khỏi QUYẾT ĐỊNH.

    ⚠️ PHẠM VI TÁC DỤNG THẬT, đã đo: ở score_thr >= 0.5 thì bg_veto KHÔNG đổi gì cả. Ba lớp có
    tổng xác suất = 1 nên nhiều nhất một lớp đạt được >= 0.5, và lớp đó đương nhiên là argmax;
    ngưỡng 0.5 đã bao hàm phủ quyết. bg_veto chỉ thật sự cắt được box khi score_thr < 0.5
    (VD thr=0.3: P(nền)=0.45, P(ô trống)=0.35 -> có box nếu không phủ quyết).
    Vì vậy đây KHÔNG phải cách chữa lệch prior nền — việc đó là của
    infer.correct_negative_sampling(). Đừng nhầm hai thứ.

    Đặt bg_veto=False để tái lập đúng những con số đã ghi trong results.csv trước ngày 04/09.
    """
    score_thr = config.SCORE_THR if score_thr is None else score_thr
    nms_iou = config.NMS_IOU if nms_iou is None else nms_iou

    classes = np.asarray(classes)
    proba = np.asarray(proba)
    boxes_all = meta[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=np.float64)
    image_ids = meta["image_id"].to_numpy()

    vetoed = np.zeros(len(proba), dtype=bool)
    if bg_veto:
        bg_col = np.where(classes == BACKGROUND)[0]
        if bg_col.size:
            vetoed = proba.argmax(axis=1) == bg_col[0]

    rows = []
    for label in OBJECT_LABELS:
        col = np.where(classes == label)[0]
        if col.size == 0:
            continue  # model không hề biết lớp này (VD train thiếu lớp) -> không sinh box
        scores_all = proba[:, col[0]]
        cand = np.where((scores_all >= score_thr) & ~vetoed)[0]
        if cand.size == 0:
            continue
        # NMS phải chạy TRONG TỪNG ẢNH — trộn ảnh với nhau thì box ảnh này sẽ dập box ảnh khác.
        order_by_img = np.argsort(image_ids[cand], kind="stable")
        cand = cand[order_by_img]
        bounds = np.flatnonzero(image_ids[cand][1:] != image_ids[cand][:-1]) + 1
        for chunk in np.split(cand, bounds):
            kept = chunk[nms(boxes_all[chunk], scores_all[chunk], nms_iou)]
            for i in kept:
                x1, y1, x2, y2 = boxes_all[i]
                rows.append([image_ids[i], x1, y1, x2, y2, label, float(scores_all[i])])

    return pd.DataFrame(rows, columns=evaluate_pklot.CSV_COLS)


def sweep_threshold(meta, proba, classes, gt, thresholds=None, nms_ious=None, bg_veto=True):
    """Quét SCORE_THR **x** NMS_IOU, trả bảng chỉ số mỗi tổ hợp (KE_HOACH.md §7 Ngày 6-9, P1).

    🔴 Trước 04/09 hàm này CHỈ quét score_thr. Đó là lý do không ai phát hiện NMS_IOU=0.45 rơi
    trúng khe hở (48/72)^2 = 0.444 và làm mAP tụt từ 0.68 xuống 0.27 — §7 giao P1 "quét ngưỡng
    + NMS" nhưng công cụ chỉ làm được một nửa. Giờ quét cả hai chiều.

    Chỉ được quét trên VAL. Quét trên test = tinh chỉnh trên test = vi phạm quy tắc §8 #3.
    """
    thresholds = thresholds if thresholds is not None else [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if nms_ious is None:
        nms_ious = [0.10, 0.15, 0.20, 0.25, 0.30, 0.45]
    elif np.isscalar(nms_ious):
        nms_ious = [nms_ious]
    rows = []
    for n in nms_ious:
        for thr in thresholds:
            pred = build_predictions(meta, proba, classes, score_thr=thr, nms_iou=n, bg_veto=bg_veto)
            m = evaluate_pklot.evaluate(gt, pred)
            rows.append({"nms_iou": n, "score_thr": thr, "n_pred": len(pred), **m})
    return pd.DataFrame(rows)


def self_test():
    """NMS phải gộp box chồng nhau trong CÙNG ảnh, và KHÔNG được gộp qua ảnh khác nhau."""
    # 3 cửa sổ chồng nhau trên ô A của img1 + 1 cửa sổ ở ô B xa hẳn -> phải còn đúng 2 box.
    meta = pd.DataFrame({
        "image_id": ["img1", "img1", "img1", "img1", "img2"],
        "x_min": [0, 8, 16, 500, 0], "y_min": [0, 0, 0, 500, 0],
        "x_max": [96, 104, 112, 596, 96], "y_max": [96, 96, 96, 596, 96],
    })
    classes = np.array([EMPTY, OCCUPIED, BACKGROUND])
    proba = np.array([
        [0.05, 0.90, 0.05],   # img1 ô A — score cao nhất, phải được giữ
        [0.05, 0.80, 0.15],   # img1 ô A — chồng lên trên, phải bị NMS dập
        [0.05, 0.70, 0.25],   # img1 ô A — chồng lên trên, phải bị NMS dập
        [0.05, 0.85, 0.10],   # img1 ô B — xa, phải được giữ
        [0.05, 0.88, 0.07],   # img2 — ẢNH KHÁC, trùng toạ độ ô A: KHÔNG được bị img1 dập
    ])
    pred = build_predictions(meta, proba, classes, score_thr=0.5, nms_iou=0.45)
    got = sorted(zip(pred.image_id, pred.x_min))
    assert got == [("img1", 0.0), ("img1", 500.0), ("img2", 0.0)], (
        f"NMS sai: kỳ vọng 3 box (img1 x2, img2 x1), nhận được {got}")

    # Lớp nền (2) không bao giờ được sinh box, dù xác suất có cao đến đâu.
    bg = np.tile(np.array([[0.0, 0.0, 1.0]]), (5, 1))
    assert len(build_predictions(meta, bg, classes, score_thr=0.5)) == 0, "lớp nền bị lọt vào dự đoán"

    # Phủ quyết nền: P(nền)=0.90 áp đảo, nhưng P(ô trống)=0.55 vẫn vượt ngưỡng 0.5.
    # Không có phủ quyết -> sinh box "ô trống" sai. Có phủ quyết -> im lặng, đúng.
    # (proba ở đây cố ý không chuẩn hoá tổng 1 — chỉ cần argmax là lớp nền.)
    ambiguous = np.tile(np.array([[0.55, 0.05, 0.90]]), (5, 1))
    assert len(build_predictions(meta, ambiguous, classes, score_thr=0.5, bg_veto=True)) == 0, (
        "bg_veto không chặn được cửa sổ mà lớp nền thắng argmax")
    assert len(build_predictions(meta, ambiguous, classes, score_thr=0.5, bg_veto=False)) > 0, (
        "bg_veto=False phải giữ nguyên hành vi cũ để tái lập số cũ trong results.csv")

    # Phủ quyết KHÔNG được đụng tới cửa sổ mà lớp đối tượng thắng argmax.
    assert len(build_predictions(meta, proba, classes, score_thr=0.5, bg_veto=True)) == 3, (
        "bg_veto chặn nhầm cửa sổ có xe/ô trống hợp lệ")

    print("detect.self_test PASS: NMS gộp đúng trong ảnh, không rò qua ảnh khác, "
          "nền bị loại khỏi đầu ra VÀ phủ quyết được cửa sổ mơ hồ")
    return True


if __name__ == "__main__":
    self_test()
