"""evaluate.py — Hàm average precision dùng chung
Chủ sở hữu: P1
"""
import numpy as np


def _intersection_area(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def iou(box_a, box_b):
    """box: (x_min, y_min, x_max, y_max)."""
    inter = _intersection_area(box_a, box_b)
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def contains_ratio(small_box, big_box):
    """Tỉ lệ diện tích small_box nằm trong big_box (IoA — intersection over area của small_box).
    Dùng cho vùng ignore LỚN hơn hẳn 1 cửa sổ/prediction (VD vùng lề chưa gán nhãn) — IoU thường sẽ
    NHỎ một cách giả tạo khi 1 box nhỏ nằm trọn trong 1 vùng lớn hơn nhiều (IoU ưu tiên kích thước
    tương đồng), trong khi containment mới là điều thực sự cần kiểm tra ở đây."""
    inter = _intersection_area(small_box, big_box)
    x1, y1, x2, y2 = small_box
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / area if area > 0 else 0.0


def average_precision(gt_by_image, pred_by_image, iou_threshold=0.5, ignore_by_image=None):
    """AP 1 lớp, nội suy toàn điểm (all-point, kiểu VOC2010+/COCO).

    gt_by_image:     {image_id: [(x1,y1,x2,y2), ...]}
    pred_by_image:   {image_id: [(x1,y1,x2,y2,score), ...]}
    ignore_by_image: {image_id: [(x1,y1,x2,y2), ...]} — vùng có thể có ô thật nhưng KHÔNG rõ nhãn
        (VD: 1 ô thiếu occupied trong XML gốc, HOẶC cả 1 dải chưa gán nhãn ở rìa bãi — xem
        pklot_data.unlabeled_margin_regions()). Prediction NẰM PHẦN LỚN trong 1 vùng ignore
        (contains_ratio >= iou_threshold — dùng containment chứ không phải IoU đối xứng, vì vùng
        ignore có thể lớn hơn hẳn 1 prediction) bị LOẠI HOÀN TOÀN khỏi tính AP — không tính TP cũng
        không tính FP (chuẩn COCO/VOC "ignore"/"crowd"), để không phạt oan một dự đoán đúng vào
        đúng vị trí mà ta chỉ đơn giản là không biết nhãn.
    """
    n_gt = sum(len(v) for v in gt_by_image.values())
    if n_gt == 0:
        return 0.0

    ignore_by_image = ignore_by_image or {}
    matched = {img: np.zeros(len(boxes), dtype=bool) for img, boxes in gt_by_image.items()}

    preds = []
    for img, boxes in pred_by_image.items():
        ignores = ignore_by_image.get(img, [])
        for b in boxes:
            if any(contains_ratio(b[:4], ig) >= iou_threshold for ig in ignores):
                continue  # nằm trong vùng "không rõ nhãn" -> bỏ qua, không tính TP/FP
            preds.append((img, b))
    preds.sort(key=lambda p: p[1][4], reverse=True)

    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for i, (img, pbox) in enumerate(preds):
        gts = gt_by_image.get(img, [])
        best_iou, best_j = 0.0, -1
        for j, gbox in enumerate(gts):
            if matched[img][j]:
                continue
            v = iou(pbox[:4], gbox)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_threshold and best_j >= 0:
            tp[i] = 1
            matched[img][best_j] = True
        else:
            fp[i] = 1

    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # Nội suy toàn điểm: precision envelope giảm dần theo recall.
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([precision[0] if len(precision) else 1.0], precision))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))
