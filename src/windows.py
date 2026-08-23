"""windows.py — Sinh cửa sổ trượt, gán nhãn IoU 3 lớp
Chủ sở hữu: P3
"""
import config
from evaluate import contains_ratio, iou
from pklot_data import UNKNOWN_OCCUPIED


def slide_windows(img_w, img_h, window_size=None, stride=None, scales=None):
    """Sinh box (x_min,y_min,x_max,y_max) trượt qua ảnh, nhiều tỉ lệ (multi-scale).
    Ở mỗi scale, cửa sổ và bước trượt cùng nhân theo scale (giữ tỉ lệ overlap không đổi).
    """
    window_size = window_size or config.WINDOW_SIZE
    stride = stride or config.STRIDE
    scales = scales or config.SCALES
    for scale in scales:
        w = round(window_size * scale)
        s = max(1, round(stride * scale))
        if w > img_w or w > img_h:
            continue
        for y in range(0, img_h - w + 1, s):
            for x in range(0, img_w - w + 1, s):
                yield (x, y, x + w, y + w)


def count_windows(img_w, img_h, window_size=None, stride=None, scales=None):
    """Đếm nhanh không cần sinh hết list — dùng để ước lượng tốc độ/dung lượng."""
    window_size = window_size or config.WINDOW_SIZE
    stride = stride or config.STRIDE
    scales = scales or config.SCALES
    total = 0
    per_scale = {}
    for scale in scales:
        w = round(window_size * scale)
        s = max(1, round(stride * scale))
        if w > img_w or w > img_h:
            per_scale[scale] = 0
            continue
        nx = (img_w - w) // s + 1
        ny = (img_h - w) // s + 1
        per_scale[scale] = nx * ny
        total += nx * ny
    return {"total": total, "per_scale": per_scale}


def label_windows(win_boxes, gt_boxes, gt_labels, iou_positive=None, iou_ignore=None):
    """Gán nhãn 3 lớp mỗi cửa sổ theo IoU lớn nhất với các ô thật (KE_HOACH.md §2):
    'car' (khớp ô có xe), 'empty' (khớp ô trống), 'background' (không phải ô nào),
    'ignore' (mơ hồ, IOU_IGNORE <= iou < IOU_POSITIVE — loại khỏi train, không tính là background).
    gt_boxes/gt_labels: box + nhãn occupied của TẤT CẢ ô thật trong ảnh (đã cùng hệ toạ độ với win_boxes),
    gt_labels có thể chứa UNKNOWN_OCCUPIED (-1) — vùng có thể có ô thật nhưng không rõ nhãn (1 ô lẻ
    thiếu occupied, HOẶC cả 1 dải lề chưa gán nhãn — pklot_data.unlabeled_margin_regions()). Cửa sổ
    NẰM PHẦN LỚN trong 1 vùng đó vẫn LUÔN là 'ignore', không được đoán bừa thành car/empty/background.

    Kiểm tra vùng UNKNOWN_OCCUPIED bằng containment (cửa sổ nằm trong vùng bao nhiêu %), KHÔNG dùng
    IoU đối xứng — vùng lề có thể lớn hơn hẳn 1 cửa sổ, IoU sẽ nhỏ giả tạo dù cửa sổ nằm trọn trong đó.

    Trả về list (class, matched_index) — matched_index là vị trí trong gt_boxes/gt_labels của ô đã
    khớp (dùng để tra rotatedRect thật, cắt ảnh "dựng thẳng" thay vì box axis-aligned dư nền — xem
    build_dataset.py), None nếu class là 'ignore'/'background' (không có ô nào để tra).
    """
    iou_positive = iou_positive if iou_positive is not None else config.IOU_POSITIVE
    iou_ignore = iou_ignore if iou_ignore is not None else config.IOU_IGNORE
    out = []
    for wb in win_boxes:
        if any(contains_ratio(wb, gb) >= iou_ignore
               for gb, gl in zip(gt_boxes, gt_labels) if gl == UNKNOWN_OCCUPIED):
            out.append(("ignore", None))
            continue

        best_iou, best_label, best_j = 0.0, None, None
        for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
            if gl == UNKNOWN_OCCUPIED:
                continue  # đã xét riêng ở trên bằng containment
            v = iou(wb, gb)
            if v > best_iou:
                best_iou, best_label, best_j = v, gl, j
        if best_iou >= iou_positive:
            out.append(("car" if best_label == 1 else "empty", best_j))
        elif best_iou >= iou_ignore:
            out.append(("ignore", None))
        else:
            out.append(("background", None))
    return out
