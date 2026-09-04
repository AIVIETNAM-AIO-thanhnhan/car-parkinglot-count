"""windows.py — Sinh cửa sổ trượt + gán nhãn theo IoU.  Chủ sở hữu: P3 (Pipeline Engineer).

Biến ảnh bãi đỗ + box ground-truth (parse từ XML PKLot) thành cửa sổ ứng viên có nhãn.

Khung nhãn dùng chung toàn dự án:
    0 = ô trống    (empty)      -> cửa sổ trùng một ô ground-truth đang trống
    1 = có xe      (occupied)   -> cửa sổ trùng một ô ground-truth có xe
    2 = nền        (background) -> không trùng ô nào
   -1 = bỏ qua     (ignore)     -> trùng ở "vùng xám" -> loại khỏi train

🔴 FILE NÀY CÓ HAI API SONG SONG, CỐ Ý. Đừng gộp lại.

  (A) slide_windows / label_windows      — bản ĐÃ SINH RA features/shard_*.parquet.
      Nhận (img_w, img_h) — RỘNG TRƯỚC. Mặc định dùng config.SCALES (5 tỉ lệ) và stride
      nhân theo scale. label_windows trả [(chuỗi_lớp, chỉ_số_ô_khớp)] để build_dataset.py
      tra ngược rotatedRect. Mọi suy luận trên model train từ parquet PHẢI dùng nhánh này,
      nếu không cửa sổ lúc chạy sẽ khác cửa sổ lúc train.

  (B) generate_windows / label_window_ids — bản viết sau (commit 9fc4384), dùng bởi
      build_features.py. Nhận (img_h, img_w) — CAO TRƯỚC. Mặc định một tỉ lệ, stride cố định.
      Trả mảng int8. Chưa từng sinh ra shard nào.

⚠️ Thứ tự tham số của hai nhánh NGƯỢC NHAU. Gọi nhầm sẽ không báo lỗi trên ảnh vuông, và
   lệch âm thầm trên ảnh chữ nhật. Luôn truyền bằng keyword khi không chắc.
"""
import numpy as np

import config


# ======================================================================
# (A) API GỐC — đã sinh ra features/shard_*.parquet.  ĐỪNG ĐỔI HÀNH VI.
# ======================================================================
def slide_windows(img_w, img_h, window_size=None, stride=None, scales=None):
    """Sinh box (x_min,y_min,x_max,y_max) trượt qua ảnh, nhiều tỉ lệ (multi-scale).

    ⚠️ RỘNG TRƯỚC, CAO SAU — ngược với generate_windows().

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

    gt_boxes/gt_labels: box + nhãn occupied của TẤT CẢ ô thật trong ảnh (đã cùng hệ toạ độ với
    win_boxes), gt_labels có thể chứa UNKNOWN_OCCUPIED (-1) — vùng có thể có ô thật nhưng không
    rõ nhãn (1 ô lẻ thiếu occupied, HOẶC cả 1 dải lề chưa gán nhãn —
    pklot_data.unlabeled_margin_regions()). Cửa sổ NẰM PHẦN LỚN trong 1 vùng đó vẫn LUÔN là
    'ignore', không được đoán bừa thành car/empty/background.

    Kiểm tra vùng UNKNOWN_OCCUPIED bằng containment (cửa sổ nằm trong vùng bao nhiêu %), KHÔNG
    dùng IoU đối xứng — vùng lề có thể lớn hơn hẳn 1 cửa sổ, IoU sẽ nhỏ giả tạo dù cửa sổ nằm
    trọn trong đó.

    Trả về list (class, matched_index) — matched_index là vị trí trong gt_boxes/gt_labels của ô
    đã khớp (dùng để tra rotatedRect thật, cắt ảnh "dựng thẳng" thay vì box axis-aligned dư nền
    — xem build_dataset.py), None nếu class là 'ignore'/'background'.
    """
    # import trong hàm: pklot_data kéo theo cả chuỗi phụ thuộc, mà infer.py chỉ cần slide_windows
    from evaluate import contains_ratio, iou
    from pklot_data import UNKNOWN_OCCUPIED

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


# ======================================================================
# (B) API vector hoá — dùng bởi build_features.py.  Chưa sinh shard nào.
# ======================================================================
def generate_windows(img_h, img_w, window_size, stride, scales=(1.0,)):
    """Trả mảng (N,4) các cửa sổ (x1,y1,x2,y2) phủ đều ảnh.

    ⚠️ CAO TRƯỚC, RỘNG SAU — ngược với slide_windows(). Và stride KHÔNG nhân theo scale.
    """
    wins = []
    for s in scales:
        w = int(round(window_size * s))
        for y in range(0, img_h - w + 1, stride):
            for x in range(0, img_w - w + 1, stride):
                wins.append((x, y, x + w, y + w))
    return np.asarray(wins, dtype=np.int32)


def iou_matrix(windows, gt_boxes):
    """windows:(N,4), gt_boxes:(G,4) -> ma trận IoU (N,G)."""
    if len(gt_boxes) == 0:
        return np.zeros((len(windows), 0), dtype=np.float32)
    w = windows[:, None, :].astype(np.float32)   # (N,1,4)
    g = gt_boxes[None, :, :].astype(np.float32)   # (1,G,4)
    xa = np.maximum(w[..., 0], g[..., 0]); ya = np.maximum(w[..., 1], g[..., 1])
    xb = np.minimum(w[..., 2], g[..., 2]); yb = np.minimum(w[..., 3], g[..., 3])
    inter = np.clip(xb - xa, 0, None) * np.clip(yb - ya, 0, None)
    area_w = (w[..., 2] - w[..., 0]) * (w[..., 3] - w[..., 1])
    area_g = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    return inter / (area_w + area_g - inter + 1e-9)


def label_window_ids(windows, gt_boxes, gt_labels, iou_pos=None, iou_ignore=None):
    """Bản mảng-số của label_windows(): trả (N,) int8 theo khung nhãn ở docstring module.

    Đổi tên từ 'label_windows' (commit 9fc4384) để không đụng hàm cùng tên ở nhánh (A) —
    hai hàm trả kiểu khác nhau hoàn toàn (mảng int vs list chuỗi), trùng tên là bẫy thật sự.
    """
    iou_pos = iou_pos if iou_pos is not None else config.IOU_POSITIVE
    iou_ignore = iou_ignore if iou_ignore is not None else config.IOU_IGNORE
    ious = iou_matrix(windows, gt_boxes)                 # (N,G)
    labels = np.full(len(windows), 2, dtype=np.int8)     # mặc định = nền
    if ious.shape[1] == 0:
        return labels
    best = ious.max(axis=1)                              # trùng nhiều nhất là bao nhiêu
    arg = ious.argmax(axis=1)                            # trùng với ô nào
    pos = best >= iou_pos
    labels[pos] = np.asarray(gt_labels)[arg[pos]].astype(np.int8)   # 0 hoặc 1 từ ô đã khớp
    ign = (best >= iou_ignore) & (best < iou_pos)
    labels[ign] = -1                                     # vùng xám -> bỏ qua
    return labels


def sample_windows(windows, labels, neg_rate=None, seed=None):
    """Giữ toàn bộ cửa sổ có xe/trống, bỏ hết -1, chỉ giữ neg_rate phần cửa sổ nền.

    ⚠️ Đây là nguồn gốc của lệch prior giữa train và inference: model học với ~80% nền trong khi
    ảnh thật có ~98% nền. infer.correct_negative_sampling() bù lại đúng bằng neg_rate này.
    """
    neg_rate = config.NEG_SAMPLE_RATE if neg_rate is None else neg_rate
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    keep = np.zeros(len(labels), dtype=bool)
    keep[(labels == 0) | (labels == 1)] = True           # giữ hết lớp thật
    bg = np.where(labels == 2)[0]
    chosen = rng.choice(bg, size=int(len(bg) * neg_rate), replace=False) if len(bg) else []
    keep[chosen] = True
    return windows[keep], labels[keep]


def self_test():
    """Chốt chặn: hai API không được lẫn lộn thứ tự (w,h) / (h,w)."""
    # Ảnh chữ nhật rõ rệt để bắt lỗi hoán vị: 400 rộng x 200 cao, cửa sổ 100, stride 100.
    # slide_windows(w=400, h=200) -> x có 4 vị trí (0,100,200,300), y có 2 (0,100) = 8 cửa sổ.
    wins_a = list(slide_windows(400, 200, window_size=100, stride=100, scales=[1.0]))
    assert len(wins_a) == 8, f"slide_windows: kỳ vọng 8 cửa sổ, nhận {len(wins_a)}"
    assert max(w[2] for w in wins_a) == 400 and max(w[3] for w in wins_a) == 200, (
        "slide_windows đang hoán vị rộng/cao")

    # generate_windows(h=200, w=400) phải cho ĐÚNG tập cửa sổ đó — chỉ khác thứ tự tham số.
    wins_b = generate_windows(200, 400, 100, 100)
    assert sorted(map(tuple, wins_b.tolist())) == sorted(wins_a), (
        "generate_windows và slide_windows lệch nhau ở cùng một ảnh")

    assert count_windows(400, 200, window_size=100, stride=100, scales=[1.0])["total"] == 8

    # stride nhân theo scale (đặc trưng riêng của nhánh A): scale 2.0 -> cửa sổ 200, bước 200
    wins_s2 = list(slide_windows(400, 200, window_size=100, stride=100, scales=[2.0]))
    assert wins_s2 == [(0, 0, 200, 200), (200, 0, 400, 200)], wins_s2

    # nhãn: 1 ô có xe phủ trọn cửa sổ đầu tiên
    gt_boxes = np.array([[0, 0, 100, 100]])
    labs = label_window_ids(np.asarray(wins_a, dtype=np.int32), gt_boxes, np.array([1]))
    assert labs[0] == 1 and (labs[1:] == 2).all(), labs

    print("windows.self_test PASS: slide_windows(w,h) và generate_windows(h,w) khớp nhau, "
          "stride nhân theo scale đúng")
    return True


if __name__ == "__main__":
    self_test()

    # 3 ô giả: có xe, trống, có xe
    gt_boxes = np.array([[40, 30, 92, 86], [100, 30, 152, 86], [160, 32, 214, 88]])
    gt_labels = np.array([1, 0, 1])                      # từ thuộc tính 'occupied' trong XML

    wins = generate_windows(img_h=580, img_w=1020, window_size=52, stride=16)
    labs = label_window_ids(wins, gt_boxes, gt_labels)
    wins_s, labs_s = sample_windows(wins, labs)

    print(f"cửa sổ sinh ra    : {len(wins)}")
    print(f"  có xe    (1)    : {(labs == 1).sum()}")
    print(f"  ô trống  (0)    : {(labs == 0).sum()}")
    print(f"  bỏ qua   (-1)   : {(labs == -1).sum()}")
    print(f"  nền      (2)    : {(labs == 2).sum()}")
    print(f"sau khi lấy mẫu nền: còn {len(wins_s)} cửa sổ")
