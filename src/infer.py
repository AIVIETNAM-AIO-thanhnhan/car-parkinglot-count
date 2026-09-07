"""infer.py — MỘT ảnh bãi đỗ  ->  ba con số.  Chủ sở hữu: P1.

    ảnh  ->  cửa sổ / ô đỗ  ->  395 feature  ->  model  ->  box  ->  SỐ XE / SỐ CHỖ TRỐNG / % LẤP ĐẦY

Đây là đường mà toàn bộ phần còn lại của dự án KHÔNG có: mọi thứ khác đều chạy theo lô qua
features/shard_*.parquet đã trích sẵn. File này là thứ UI gọi.

Hai chế độ, cố ý khác nhau về bản chất:

  NHÁNH B — classify_slots(): đã biết trước vị trí từng ô, chỉ phân loại có xe / trống.
      ~300 ô mỗi ảnh, không cần ngưỡng, không cần NMS, không đụng tới lớp nền.
      Cắt ô bằng rotatedRect y hệt lúc train (build_dataset._rotated_crop) nên KHÔNG dính lệch
      train/inference. Đây là chế độ cho ra số đáng tin.

  DETECTOR — detect_image(): không cần biết gì về bãi, tự trượt cửa sổ tìm ô.
      ~16.000-20.000 cửa sổ mỗi ảnh. Đúng đề bài KE_HOACH §2 nhưng kém chính xác hơn hẳn, vì
      lúc train mọi ô car/empty đều được cắt theo rotatedRect đã xoay thẳng, còn ở đây không có
      nhãn nên mọi cửa sổ đều vuông góc (xem report §lệch train/inference).

Nhãn dùng chung: 0 = ô trống, 1 = có xe, 2 = nền, -1 = không rõ.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image, ImageDraw

import config
import detect
import evaluate_pklot
import features
import windows

EMPTY, OCCUPIED, BACKGROUND = detect.EMPTY, detect.OCCUPIED, detect.BACKGROUND
LABEL_TEXT = {EMPTY: "ô trống", OCCUPIED: "có xe"}
BOX_COLOR = {EMPTY: (60, 220, 90), OCCUPIED: (235, 60, 60)}   # xanh = trống, đỏ = có xe

# Khoá tra rotatedRect trong một dict ô đỗ
ROT_KEYS = ("rot_cx", "rot_cy", "rot_w", "rot_h", "rot_angle")


# ======================================================================
# Nạp model
# ======================================================================
def load_model(path):
    """Nạp bundle .joblib do train_model.save_model() ghi ra, có kiểm hợp đồng nhãn/feature."""
    import train_model
    return train_model.load_model(path)


def available_models(models_dir=None):
    d = Path(models_dir) if models_dir else config.MODELS
    return sorted(d.glob("*.joblib")) if d.exists() else []


# ======================================================================
# Layout ô đỗ — ba nguồn
# ======================================================================
def _slot(x_min, y_min, x_max, y_max, slot_id=None, rot=None):
    s = {"slot_id": slot_id, "x_min": int(x_min), "y_min": int(y_min),
         "x_max": int(x_max), "y_max": int(y_max)}
    if rot is not None:
        s.update(dict(zip(ROT_KEYS, (float(v) for v in rot))))
    return s


def slot_boxes_from_xml(xml_path):
    """XML PKLot -> danh sách ô. Giữ cả rotatedRect để cắt ô "dựng thẳng" như lúc train.

    Dùng lại pklot_data.parse_spaces() — đã xử lý đủ các ca lệch chuẩn của PKLot.
    Ô thiếu thuộc tính 'occupied' vẫn được giữ (ta đang dự đoán, không cần nhãn thật).
    """
    import pklot_data
    out = []
    for sp in pklot_data.parse_spaces(xml_path):
        out.append(_slot(sp["x_min"], sp["y_min"], sp["x_max"], sp["y_max"], sp.get("id"),
                         rot=[sp[k] for k in ROT_KEYS] if all(k in sp for k in ROT_KEYS) else None))
    return out


def slot_boxes_from_gt(image_id, gt_csv=None):
    """Ô đỗ của một ảnh PKLot lấy từ processed/gt.csv (đã ở hệ toạ độ ẢNH ĐÃ CROP)."""
    gt = pd.read_csv(gt_csv or (config.PROC / "gt.csv"))
    g = gt[gt.image_id == image_id]
    if g.empty:
        raise ValueError(f"Không có ô nào cho image_id={image_id!r} trong gt.csv")
    g = g[g.label != -1]   # bỏ vùng không rõ nhãn
    return [_slot(r.x_min, r.y_min, r.x_max, r.y_max, r.slot_id,
                  rot=[getattr(r, k) for k in ROT_KEYS]) for r in g.itertuples()]


def slot_boxes_from_json(text_or_path):
    """Layout tự vẽ / dán tay: [[x_min,y_min,x_max,y_max], ...] hoặc [{"x_min":…}, ...]."""
    raw = json.loads(Path(text_or_path).read_text(encoding="utf-8")
                     if Path(str(text_or_path)).exists() else text_or_path)
    out = []
    for i, item in enumerate(raw):
        if isinstance(item, dict):
            out.append(_slot(item["x_min"], item["y_min"], item["x_max"], item["y_max"],
                             item.get("slot_id", i),
                             rot=[item[k] for k in ROT_KEYS] if all(k in item for k in ROT_KEYS) else None))
        else:
            out.append(_slot(*item[:4], slot_id=i))
    if not out:
        raise ValueError("Layout rỗng — cần ít nhất 1 ô.")
    return out


def crop_for_lot(lot, crops_path=None):
    """Hộp cắt vùng có nhãn của 1 bãi PKLot, từ processed/crops.json (hệ toạ độ ảnh GỐC)."""
    crops = json.loads(Path(crops_path or (config.PROC / "crops.json")).read_text(encoding="utf-8"))
    if lot not in crops:
        raise KeyError(f"Không có hộp cắt cho bãi {lot!r}. Có: {sorted(crops)}")
    return tuple(crops[lot])


def crop_image(image_rgb, box):
    """Cắt ảnh theo hộp, ĐỆM ĐEN nếu hộp vượt ra ngoài ảnh.

    🔴 Phải dùng hàm này chứ không phải cắt lát numpy. build_dataset.py cắt bằng
    PIL.Image.crop(), và PIL CHO PHÉP hộp vượt biên rồi đệm 0 vào phần thiếu. Hộp của UFPR04
    trong crops.json là [31,44,1036,765] — cao 721px trong khi ảnh PKLot chỉ cao 720px, nên
    kích thước thật lúc train là 1005x721 với một dải đen 1px ở đáy.

    Cắt bằng `image[44:765]` sẽ ra 1005x676 (cụt 45 dòng): lưới cửa sổ trượt lệch đi và mọi
    toạ độ trong gt.csv/parquet không còn khớp. Đã bắt được lỗi này bằng phép so feature.
    """
    return np.array(Image.fromarray(np.asarray(image_rgb)).crop(tuple(int(v) for v in box)))


# ======================================================================
# Cắt ô + trích feature
# ======================================================================
def _rotated_crop(image, cx, cy, w, h, angle):
    """Cắt 'dựng thẳng' một ô nghiêng bằng rotatedRect thật.

    Sao đúng từ build_dataset._rotated_crop() — KHÔNG import để tránh kéo theo cả chuỗi phụ
    thuộc của build_dataset (pklot_data, gt.csv) vào UI. Hai bản phải giống nhau từng dòng:
    lệch một chi tiết là feature lúc chạy khác feature lúc train.
    """
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(image, m, (image.shape[1], image.shape[0]))
    return cv2.getRectSubPix(rotated, (max(1, round(w)), max(1, round(h))), (cx, cy))


def _to_window(crop):
    """Đưa crop về đúng WINDOW_SIZE — features.extract() yêu cầu (HOG 6x6 cell)."""
    n = config.WINDOW_SIZE
    if crop.size == 0:
        return np.zeros((n, n, 3), dtype=np.uint8)
    if crop.shape[:2] != (n, n):
        # build_dataset.py dùng PIL.Image.resize (bilinear) — giữ y hệt để feature khớp bit-for-bit
        crop = np.array(Image.fromarray(crop).resize((n, n)))
    return crop


def _crop_slot(image_rgb, slot):
    """Ô có rotatedRect -> cắt dựng thẳng (như lúc train). Không có -> cắt vuông góc."""
    if all(k in slot for k in ROT_KEYS):
        crop = _rotated_crop(image_rgb, slot["rot_cx"], slot["rot_cy"],
                             slot["rot_w"], slot["rot_h"], slot["rot_angle"])
    else:
        crop = image_rgb[slot["y_min"]:slot["y_max"], slot["x_min"]:slot["x_max"]]
    return _to_window(crop)


def _extract_chunk(image_rgb, boxes):
    """Trích feature cho một LÔ vùng — đơn vị công việc gửi sang worker."""
    out = np.empty((len(boxes), 395), dtype=np.float32)
    for i, b in enumerate(boxes):
        if isinstance(b, dict):
            crop = _crop_slot(image_rgb, b)
        else:
            x1, y1, x2, y2 = b
            crop = _to_window(image_rgb[y1:y2, x1:x2])
        out[i] = features.extract(crop)
    return out


def extract_batch(image_rgb, boxes, n_jobs=None, chunk=512):
    """Trích feature cho nhiều vùng. boxes: list dict (ô) hoặc list (x1,y1,x2,y2) (cửa sổ).

    ⚠️ Chia theo LÔ chứ không phải theo từng cửa sổ. joblib dùng backend tiến trình (loky), nên
    mỗi tác vụ phải pickle lại cả ảnh: gửi 20.000 tác vụ một-cửa-sổ thì chi phí truyền ảnh nuốt
    sạch phần lợi, đo được là song song KHÔNG nhanh hơn tuần tự. Gửi ~40 lô thì ảnh chỉ bị
    pickle 40 lần và tăng tốc mới thật sự xảy ra.
    """
    n_jobs = config.N_JOBS if n_jobs is None else n_jobs
    boxes = list(boxes)
    if not boxes:
        return np.empty((0, 395), dtype=np.float32)

    n_workers = os.cpu_count() or 1 if n_jobs in (-1, None) else abs(n_jobs)
    # đủ lô để cân tải (~4 lô/worker) nhưng không nhỏ tới mức chi phí lại chiếm ưu thế
    size = max(chunk, int(np.ceil(len(boxes) / max(1, n_workers * 4))))
    chunks = [boxes[i:i + size] for i in range(0, len(boxes), size)]

    if n_jobs == 1 or len(chunks) == 1:
        return np.concatenate([_extract_chunk(image_rgb, c) for c in chunks])
    parts = Parallel(n_jobs=n_jobs)(delayed(_extract_chunk)(image_rgb, c) for c in chunks)
    return np.concatenate(parts)


# ======================================================================
# Bù lệch prior lớp nền
# ======================================================================
def correct_negative_sampling(proba, classes, neg_rate=None):
    """Bù lại việc build_dataset.py chỉ giữ neg_rate phần cửa sổ NỀN khi trích feature.

    Model học trên tập có ~80% nền, nhưng ảnh thật có ~98% nền (val) / ~91% (test). Nó vì thế
    đánh giá THẤP P(nền) một cách hệ thống -> false positive tràn ngập.

    Vì tỉ lệ lấy mẫu đã biết CHÍNH XÁC (config.NEG_SAMPLE_RATE = 0.10, windows.sample_windows),
    đây là hiệu chỉnh prior chuẩn cho negative subsampling, không phải ước lượng:

        p'(nền) ∝ p(nền) / neg_rate       (các lớp khác giữ nguyên trọng số 1)

    rồi chuẩn hoá lại cho tổng bằng 1. Với neg_rate = 1.0 hàm trả về đúng proba ban đầu.

    ⚠️ CHỈ áp cho ảnh thật. KHÔNG áp khi đo trên parquet val/test — parquet cũng đã bị lấy mẫu
    y như train nên ở đó phân phối vốn đã khớp, hiệu chỉnh sẽ làm SAI đi.
    """
    neg_rate = config.NEG_SAMPLE_RATE if neg_rate is None else neg_rate
    if not 0 < neg_rate <= 1:
        raise ValueError(f"neg_rate phải trong (0,1], nhận {neg_rate}")
    proba = np.asarray(proba, dtype=np.float64)
    if neg_rate == 1.0:
        return proba

    bg_col = np.where(np.asarray(classes) == BACKGROUND)[0]
    if bg_col.size == 0:
        return proba          # model không có lớp nền -> không có gì để bù
    w = np.ones(proba.shape[1])
    w[bg_col[0]] = 1.0 / neg_rate
    out = proba * w
    return out / out.sum(axis=1, keepdims=True)


# ======================================================================
# NHÁNH B — vị trí ô đã biết
# ======================================================================
def classify_slots(image_rgb, slots, bundle, n_jobs=None):
    """Phân loại từng ô đỗ đã biết vị trí -> DataFrame theo evaluate_pklot.CSV_COLS.

    Không dùng lớp nền và không cần ngưỡng: vị trí ô đã biết chắc LÀ ô, câu hỏi duy nhất là
    có xe hay trống. Vì vậy chọn argmax GIỮA HAI lớp 0 và 1, bỏ qua cột nền.
    Cũng không cần NMS — các ô không chồng nhau và mỗi ô cho đúng một box.
    """
    clf = bundle["clf"]
    if not slots:
        return pd.DataFrame(columns=evaluate_pklot.CSV_COLS)

    X = extract_batch(image_rgb, slots, n_jobs=n_jobs)
    proba = clf.predict_proba(X)
    classes = np.asarray(clf.classes_)

    cols = {}
    for label in (EMPTY, OCCUPIED):
        c = np.where(classes == label)[0]
        if c.size == 0:
            raise ValueError(
                f"Model không có lớp {label} ({LABEL_TEXT[label]}) trong classes_={classes.tolist()} "
                "— mọi ô sẽ bị dồn về lớp còn lại. Phải train lại trên dữ liệu đủ 3 lớp.")
        cols[label] = proba[:, c[0]]

    two = np.column_stack([cols[EMPTY], cols[OCCUPIED]])
    idx = two.argmax(axis=1)
    labels = np.where(idx == 0, EMPTY, OCCUPIED)
    # Điểm = xác suất của lớp thắng, chuẩn hoá trong hai lớp (bỏ khối lượng của lớp nền ra)
    scores = two.max(axis=1) / np.clip(two.sum(axis=1), 1e-9, None)

    return pd.DataFrame({
        "image_id": "input",
        "x_min": [s["x_min"] for s in slots], "y_min": [s["y_min"] for s in slots],
        "x_max": [s["x_max"] for s in slots], "y_max": [s["y_max"] for s in slots],
        "label": labels, "score": scores,
    })[evaluate_pklot.CSV_COLS]


# ======================================================================
# DETECTOR — trượt cửa sổ
# ======================================================================
def plan_windows(img_w, img_h, window_size=None, stride=None, scales=None):
    """Báo trước số cửa sổ sẽ phải trích — để UI cảnh báo thời gian trước khi chạy."""
    return windows.count_windows(img_w, img_h, window_size, stride, scales)


def detect_image(image_rgb, bundle, scales=None, stride=None, window_size=None,
                 score_thr=None, nms_iou=None, n_jobs=None,
                 correct_prior=False, bg_veto=True, image_id="input", progress=None):
    """Trượt cửa sổ trên cả ảnh -> box cuối cùng. Ảnh nào cũng chạy được, không cần layout.

    Dùng windows.slide_windows (RỘNG trước) — đúng bản đã sinh ra parquet lúc train.
    progress: callable(done, total) để UI vẽ thanh tiến độ.

    ⚠️ correct_prior MẶC ĐỊNH TẮT dù về lý thuyết nó đúng (xem correct_negative_sampling).
    Đo thật trên 3 ảnh mỗi bãi, cùng model rf_axis:

                        bật    tắt      (số box, ô thật)
        UFPR04 (28 ô)    22     69
        PUCPR (100 ô)     2     77

    Lỗi chủ đạo của việc trượt cửa sổ là BỎ SÓT, không phải báo thừa. Bù prior đẩy thêm xác
    suất về phía "nền" nên làm nặng thêm đúng cái lỗi đang có — trên PUCPR nó bóp từ 77 box
    xuống còn 2. Chỉ bật lại nếu đo được là ảnh đang bị báo thừa.
    """
    clf = bundle["clf"]
    h, w = image_rgb.shape[:2]
    win = list(windows.slide_windows(w, h, window_size or bundle.get("window_size"),
                                     stride or bundle.get("stride"),
                                     scales if scales is not None else bundle.get("scales")))
    if not win:
        raise ValueError(f"Ảnh {w}x{h} nhỏ hơn mọi cửa sổ — không có gì để quét.")

    if progress:
        progress(0, len(win))
    X = extract_batch(image_rgb, win, n_jobs=n_jobs)
    if progress:
        progress(len(win), len(win))

    proba = clf.predict_proba(X)
    if correct_prior:
        proba = correct_negative_sampling(proba, clf.classes_, bundle.get("neg_sample_rate"))

    arr = np.asarray(win)
    meta = pd.DataFrame({"image_id": image_id, "x_min": arr[:, 0], "y_min": arr[:, 1],
                         "x_max": arr[:, 2], "y_max": arr[:, 3]})
    return detect.build_predictions(meta, proba, clf.classes_, score_thr, nms_iou, bg_veto=bg_veto)


# ======================================================================
# Gộp box chéo lớp + đếm
# ======================================================================
def dedup_across_classes(pred, iou_thr=None):
    """Gộp box của HAI lớp khác nhau chồng lên nhau, giữ box điểm cao hơn.

    detect.build_predictions chạy NMS ĐỘC LẬP từng lớp (đúng chuẩn detection, vì AP tính riêng
    mỗi lớp). Hệ quả: một ô đỗ có thể sinh ra CẢ box "có xe" LẪN box "ô trống" chồng lên nhau,
    và phép đếm sẽ tính nó hai lần — vừa thành một xe, vừa thành một chỗ trống.

    ⚠️ CHỈ dùng cho hiển thị và đếm. KHÔNG dùng trước khi tính mAP.
    """
    iou_thr = config.NMS_IOU if iou_thr is None else iou_thr
    if len(pred) <= 1:
        return pred.reset_index(drop=True)

    out = []
    for _, g in pred.groupby("image_id", sort=False):
        boxes = g[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=np.float64)
        keep = detect.nms(boxes, g["score"].to_numpy(dtype=np.float64), iou_thr)
        out.append(g.iloc[keep])
    return pd.concat(out, ignore_index=True)


def count_from_predictions(pred):
    """Ba con số của KE_HOACH §1.

    Công thức khớp CHÍNH XÁC evaluate_pklot.evaluate() để số trên UI và số trong bảng kết quả
    không bao giờ mâu thuẫn nhau:
        chỗ trống = (label == 0).sum() | số xe = (label == 1).sum() | tổng ô = len(pred)
        tỉ lệ lấp đầy = 1 - chỗ trống / tổng ô
    """
    p = pred[pred.label != -1] if len(pred) else pred
    empty = int((p.label == EMPTY).sum()) if len(p) else 0
    cars = int((p.label == OCCUPIED).sum()) if len(p) else 0
    total = len(p)
    return {"cars": cars, "empty": empty, "total": total,
            "occupancy_pct": (1 - empty / total) * 100 if total else 0.0}


# ======================================================================
# Vẽ
# ======================================================================
ANNOTATIONS = {
    "coords": lambda i, r: f"{int(r.x_min)},{int(r.y_min)}",
    "box": lambda i, r: f"{int(r.x_min)},{int(r.y_min)},{int(r.x_max)},{int(r.y_max)}",
    "size": lambda i, r: f"{int(r.x_max - r.x_min)}x{int(r.y_max - r.y_min)}",
    "score": lambda i, r: f"{r.score:.2f}",
    "label": lambda i, r: LABEL_TEXT.get(r.label, "?"),
    "index": lambda i, r: str(i),
}


def _font(size):
    """Font bitmap mặc định của PIL rất nhỏ (~11px) và không đọc được trên ảnh 1200px.
    Ưu tiên font vector có sẵn của Pillow; nếu bản Pillow quá cũ thì đành dùng bitmap."""
    from PIL import ImageFont
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        pass
    try:
        return ImageFont.load_default(size=size)     # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def draw_boxes(image_rgb, pred, width=2, annotate=None, font_size=11):
    """Vẽ box lên ảnh: xanh = ô trống, đỏ = có xe. Trả PIL.Image.

    annotate: None (chỉ vẽ khung) | tên trong ANNOTATIONS | hàm (thứ_tự, dòng) -> chuỗi
        "coords" x,y góc trên trái · "box" cả 4 số · "size" rộng x cao
        "score" độ tin cậy · "label" tên lớp · "index" số thứ tự

    Chữ được vẽ trên nền đặc cùng màu khung, chữ trắng — nếu vẽ chữ màu trực tiếp lên ảnh thì
    trên nền nhựa đường sáng gần như không đọc được.
    """
    img = Image.fromarray(np.asarray(image_rgb)).convert("RGB")
    d = ImageDraw.Draw(img)

    fmt = None
    if annotate:
        fmt = annotate if callable(annotate) else ANNOTATIONS.get(annotate)
        if fmt is None:
            raise ValueError(f"annotate={annotate!r} không hợp lệ. Chọn: {sorted(ANNOTATIONS)}")
    font = _font(font_size) if fmt else None

    # Hai lượt: vẽ HẾT khung rồi mới vẽ chữ. Một lượt thì khung của ô sau đè lên chữ của ô
    # trước — ở bãi đỗ các ô kề sát nhau nên gần như nhãn nào cũng bị cắt mất một phần.
    rows = [(i, r) for i, r in enumerate(pred.itertuples()) if r.label in BOX_COLOR]
    for _, r in rows:
        d.rectangle((r.x_min, r.y_min, r.x_max, r.y_max), outline=BOX_COLOR[r.label], width=width)
    if not fmt:
        return img

    for i, r in rows:
        text = fmt(i, r)
        x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
        tw, th = x1 - x0, y1 - y0
        # đặt nhãn NGAY TRÊN khung; nếu chạm mép trên ảnh thì lật xuống nằm trong khung
        tx = min(r.x_min, img.width - tw - 3)
        ty = r.y_min - th - 3
        if ty < 0:
            ty = r.y_min + 1
        d.rectangle((tx, ty, tx + tw + 3, ty + th + 3), fill=BOX_COLOR[r.label])
        d.text((tx + 2 - x0, ty + 1 - y0), text, fill=(255, 255, 255), font=font)
    return img


def load_image(path_or_file):
    """Đọc ảnh -> mảng RGB uint8 (PIL, không phải cv2 — tránh bẫy BGR)."""
    return np.array(Image.open(path_or_file).convert("RGB"))


# ======================================================================
# Tự sinh layout ô đỗ (không cần khai báo tay)
# ======================================================================
def auto_layout(images, bundle, min_frames=None, nms_iou=None, correct_prior=True,
                progress=None, **kw):
    """Nhiều ảnh của CÙNG một camera -> danh sách ô đỗ, không cần ai khai báo gì.

    Vì sao phải nhiều ảnh: dò trên một ảnh đơn lẻ bỏ sót nhiều ô — xe che khuất vạch, bóng đổ,
    ô bị cây phủ. Nhưng camera bãi đỗ cố định còn xe thì đổi chỗ liên tục, nên một ô bị bỏ sót ở
    ảnh này sẽ lộ ra ở ảnh khác. Gom dự đoán của N ảnh rồi chỉ giữ vị trí xuất hiện LẶP LẠI ở
    >= min_frames ảnh: vừa vá được chỗ bỏ sót, vừa loại được báo nhầm ngẫu nhiên (báo nhầm hiếm
    khi rơi đúng một chỗ ở nhiều ảnh khác nhau).

    images     : list ảnh RGB (hoặc đường dẫn) của cùng một camera, cùng kích thước
    min_frames : số ảnh tối thiểu phải cùng thấy 1 vị trí. Mặc định 40% số ảnh.
    Trả list ô dạng dict như slot_boxes_from_*(), kèm 'n_frames' để biết độ tin cậy.

    ⚠️ Chỉ chạy được với model train từ ô cắt VUÔNG GÓC
    (`build_dataset.py --axis-aligned`). Model cắt xoay thẳng gọi 100% cửa sổ là nền.

    ⚠️ correct_prior MẶC ĐỊNH BẬT ở đây, NGƯỢC với detect_image (mặc định tắt). Không mâu
    thuẫn — hai bài toán khác nhau:
      · dò MỘT ảnh: lỗi chủ đạo là bỏ sót, bù prior làm nặng thêm -> tắt.
      · dò layout qua NHIỀU ảnh: camera cố định nên báo nhầm trên nền tĩnh (bụi cây, vệt
        nhựa đường) LẶP LẠI y hệt ở mọi khung hình, bộ lọc min_frames không loại được chúng.
        Phải chặn ngay từ đầu bằng bù prior.
    Đo trên 20 ảnh UFPR04 (28 ô thật): bật -> 28 ô đúng số; tắt -> 91 ô, thừa gấp 3.
    """
    nms_iou = config.NMS_IOU if nms_iou is None else nms_iou
    imgs = [load_image(i) if isinstance(i, (str, Path)) else np.asarray(i) for i in images]
    if not imgs:
        raise ValueError("Cần ít nhất 1 ảnh.")
    shapes = {im.shape[:2] for im in imgs}
    if len(shapes) > 1:
        raise ValueError(f"Các ảnh phải cùng kích thước (cùng 1 camera), nhận được {shapes}.")
    if min_frames is None:
        min_frames = max(1, round(0.4 * len(imgs)))

    parts = []
    for k, im in enumerate(imgs):
        if progress:
            progress(k, len(imgs))
        p = detect_image(im, bundle, image_id=str(k), correct_prior=correct_prior, **kw)
        parts.append(dedup_across_classes(p, nms_iou))
    if progress:
        progress(len(imgs), len(imgs))
    allp = pd.concat(parts, ignore_index=True)
    if allp.empty:
        return []

    # Gom vị trí trùng nhau qua các ảnh: NMS trên toàn bộ box (bỏ qua image_id) chọn ra các "ổ",
    # rồi đếm xem mỗi ổ được bao nhiêu ảnh khác nhau xác nhận.
    boxes = allp[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=np.float64)
    order = detect.nms(boxes, allp["score"].to_numpy(dtype=np.float64), nms_iou)
    slots = []
    for rank, i in enumerate(order):
        iou = _iou_one_to_many(boxes[i], boxes)
        member = iou >= nms_iou
        n_frames = allp.loc[member, "image_id"].nunique()
        if n_frames < min_frames:
            continue
        b = boxes[member].mean(axis=0)          # lấy trung bình cho vị trí ổn định hơn 1 ảnh
        slots.append(_slot(*b.round().astype(int), slot_id=len(slots) + 1) |
                     {"n_frames": int(n_frames)})
    return slots


def _iou_one_to_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + b - inter + 1e-9)


def slots_to_xml(slots, lot_id="lot"):
    """Xuất layout ra XML đúng định dạng PKLot — dùng lại được với slot_boxes_from_xml().

    Ô không có rot_* thì ghi rotatedRect vuông góc (angle=0) và contour là 4 đỉnh hình chữ nhật.
    """
    import xml.etree.ElementTree as ET

    root = ET.Element("parking", id=str(lot_id))
    for i, s in enumerate(slots, 1):
        sp = ET.SubElement(root, "space", id=str(s.get("slot_id", i)))
        if "occupied" in s:
            sp.set("occupied", str(int(s["occupied"])))
        cx = s.get("rot_cx", (s["x_min"] + s["x_max"]) / 2)
        cy = s.get("rot_cy", (s["y_min"] + s["y_max"]) / 2)
        w = s.get("rot_w", s["x_max"] - s["x_min"])
        h = s.get("rot_h", s["y_max"] - s["y_min"])
        ang = s.get("rot_angle", 0.0)
        rr = ET.SubElement(sp, "rotatedRect")
        ET.SubElement(rr, "center", x=str(round(cx)), y=str(round(cy)))
        ET.SubElement(rr, "size", w=str(round(w)), h=str(round(h)))
        ET.SubElement(rr, "angle", d=str(round(ang)))
        pts = cv2.boxPoints(((cx, cy), (w, h), ang)).round().astype(int)
        ct = ET.SubElement(sp, "contour")
        for x, y in pts:
            ET.SubElement(ct, "Point", x=str(int(x)), y=str(int(y)))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


# ======================================================================
def self_test():
    classes = np.array([EMPTY, OCCUPIED, BACKGROUND])

    # (a) đếm — phải khớp công thức của evaluate_pklot
    pred = pd.DataFrame({
        "image_id": ["i"] * 4, "x_min": [0, 10, 20, 30], "y_min": [0] * 4,
        "x_max": [5, 15, 25, 35], "y_max": [5] * 4,
        "label": [EMPTY, EMPTY, OCCUPIED, -1], "score": [0.9] * 4,
    })
    c = count_from_predictions(pred)
    assert c == {"cars": 1, "empty": 2, "total": 3, "occupancy_pct": (1 - 2 / 3) * 100}, c
    assert count_from_predictions(pred.iloc[:0])["occupancy_pct"] == 0.0, "ảnh rỗng phải cho 0%"

    # (b) gộp chéo lớp — hai box khác lớp chồng nhau phải còn 1, giữ box điểm cao
    dup = pd.DataFrame({
        "image_id": ["i", "i", "i"], "x_min": [0, 1, 500], "y_min": [0, 0, 500],
        "x_max": [100, 101, 600], "y_max": [100, 100, 600],
        "label": [EMPTY, OCCUPIED, EMPTY], "score": [0.6, 0.95, 0.7],
    })
    dd = dedup_across_classes(dup)
    assert len(dd) == 2, f"gộp chéo lớp sai: còn {len(dd)} box thay vì 2"
    assert dd[dd.x_min < 500].iloc[0].label == OCCUPIED, "phải giữ box điểm cao (0.95, có xe)"
    assert count_from_predictions(dup)["total"] == 3 and count_from_predictions(dd)["total"] == 2, (
        "gộp chéo lớp phải làm giảm phép đếm — đó chính là lý do nó tồn tại")

    # (c) bù prior nền
    p = np.array([[0.30, 0.20, 0.50]])
    q = correct_negative_sampling(p, classes, neg_rate=0.10)
    assert abs(q.sum() - 1) < 1e-9, "hàng phải tổng bằng 1"
    assert q[0, 2] > p[0, 2], "P(nền) phải TĂNG sau khi bù"
    assert q[0, 0] < p[0, 0] and q[0, 1] < p[0, 1], "hai lớp còn lại phải giảm tương ứng"
    assert np.allclose(correct_negative_sampling(p, classes, neg_rate=1.0), p), (
        "neg_rate=1.0 phải là phép đồng nhất")
    # tỉ lệ giữa hai lớp đối tượng KHÔNG được đổi — chỉ trọng số của nền đổi
    assert abs(q[0, 0] / q[0, 1] - p[0, 0] / p[0, 1]) < 1e-9

    # (d) đường detector đầu-cuối với model giả
    class FakeClf:
        classes_ = classes

        def predict_proba(self, X):
            return np.tile(np.array([[0.05, 0.90, 0.05]]), (len(X), 1))

    bundle = {"clf": FakeClf(), "window_size": config.WINDOW_SIZE,
              "stride": 64, "scales": [1.0], "neg_sample_rate": config.NEG_SAMPLE_RATE}
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
    out = detect_image(img, bundle, n_jobs=1)
    assert list(out.columns) == evaluate_pklot.CSV_COLS, out.columns.tolist()
    assert len(out) > 0 and set(out.label) <= {EMPTY, OCCUPIED}, "lớp nền lọt vào đầu ra"
    assert count_from_predictions(out)["cars"] == len(out), "model giả luôn nói 'có xe'"

    # (e) Nhánh B — ô vuông góc và ô có rotatedRect đều chạy
    slots = [_slot(0, 0, 96, 96, 1), _slot(100, 20, 190, 110, 2, rot=(145, 65, 90, 90, 30.0))]
    sp = classify_slots(img, slots, bundle, n_jobs=1)
    assert len(sp) == 2 and list(sp.columns) == evaluate_pklot.CSV_COLS
    assert (sp.label == OCCUPIED).all(), "model giả luôn nói 'có xe'"
    assert (sp.score > 0.5).all(), "điểm phải được chuẩn hoá trong 2 lớp"

    # (f) vẽ được, không làm hỏng ảnh gốc, và mọi kiểu nhãn đều chạy
    im = draw_boxes(img, sp)
    assert im.size == (300, 200), im.size
    before = img.copy()
    for kind in ANNOTATIONS:
        assert draw_boxes(img, sp, annotate=kind).size == (300, 200), kind
    assert draw_boxes(img, sp, annotate=lambda i, r: f"#{i}").size == (300, 200)
    assert np.array_equal(img, before), "draw_boxes vẽ đè lên mảng ảnh gốc"
    try:
        draw_boxes(img, sp, annotate="khong_ton_tai")
    except ValueError:
        pass
    else:
        raise AssertionError("annotate sai phải báo lỗi, không được im lặng vẽ khung trơn")
    # nhãn của ô sát mép trên phải được lật xuống trong khung, không bị cắt mất
    edge = pd.DataFrame({"image_id": ["i"], "x_min": [0], "y_min": [0], "x_max": [96],
                         "y_max": [96], "label": [EMPTY], "score": [0.9]})
    assert draw_boxes(img, edge, annotate="coords").size == (300, 200)

    # (g) xuất XML rồi đọc lại phải ra đúng các ô ban đầu (khứ hồi)
    xml = slots_to_xml(slots, "test")
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8") as f:
        f.write(xml)
    back = slot_boxes_from_xml(f.name)
    Path(f.name).unlink()
    assert len(back) == len(slots), f"khứ hồi XML mất ô: {len(slots)} -> {len(back)}"
    for a, b in zip(slots, back):
        # rotatedRect phải khứ hồi CHÍNH XÁC — đó là thứ quyết định cách cắt ô lúc suy luận.
        # Ô không khai báo rot_* thì XML suy ra từ hộp vuông góc (tâm, kích thước, góc 0).
        want = {k: a[k] for k in ROT_KEYS} if all(k in a for k in ROT_KEYS) else {
            "rot_cx": (a["x_min"] + a["x_max"]) / 2, "rot_cy": (a["y_min"] + a["y_max"]) / 2,
            "rot_w": a["x_max"] - a["x_min"], "rot_h": a["y_max"] - a["y_min"], "rot_angle": 0.0}
        for k in ROT_KEYS:
            assert abs(want[k] - b[k]) <= 1, (k, want, b)
        # hộp vuông góc chỉ khứ hồi đúng khi ô KHÔNG xoay; ô xoay thì hộp bao được tính lại từ
        # contour (rộng hơn hộp gốc) — đó là hành vi đúng, không phải mất mát
        if abs(b["rot_angle"]) < 1e-6:
            assert (a["x_min"], a["y_min"], a["x_max"], a["y_max"]) == \
                   (b["x_min"], b["y_min"], b["x_max"], b["y_max"]), (a, b)
        else:
            assert b["x_max"] - b["x_min"] >= a["x_max"] - a["x_min"] - 1, (a, b)

    # (h) auto_layout: 3 ảnh giống hệt nhau, model giả luôn nói "có xe" -> phải ra >= 1 ô,
    #     và min_frames cao hơn số ảnh thì không ô nào qua được
    lay = auto_layout([img] * 3, bundle, n_jobs=1)
    assert lay and all(s["n_frames"] == 3 for s in lay), lay[:2]
    assert auto_layout([img] * 3, bundle, min_frames=4, n_jobs=1) == []
    try:
        auto_layout([img, img[:100]], bundle, n_jobs=1)
    except ValueError:
        pass
    else:
        raise AssertionError("ảnh khác kích thước phải bị từ chối")

    print("infer.self_test PASS: đếm khớp evaluate_pklot, gộp chéo lớp giảm trùng, "
          "bù prior nền đúng chiều, cả Nhánh B lẫn Detector chạy đầu-cuối")
    return True


if __name__ == "__main__":
    features.self_test()
    self_test()
