"""pklot_data.py — Parse XML, cắt vùng có nhãn, lấy mẫu thời gian
Chủ sở hữu: P2
"""
import glob
import json
import random
import statistics as st
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import config

# Archive PKLot.tar.gz lồng 2 lớp: raw/PKLot/PKLot/<bãi>/... (không phải lỗi tải,
# cấu trúc gốc của archive UFPR y hệt vậy cả trên local lẫn Drive).
LOT_ROOT = config.RAW / "PKLot" / "PKLot"


UNKNOWN_OCCUPIED = -1  # sentinel: ô có vị trí thật nhưng KHÔNG rõ có xe hay trống (xem ghi chú dưới)


def parse_spaces(xml_path):
    """Đọc 1 file nhãn PKLot -> list dict {id, occupied, x_min, y_min, x_max, y_max,
    rot_cx, rot_cy, rot_w, rot_h, rot_angle} (box axis-aligned từ contour + rotatedRect gốc).

    LƯU Ý: 274/3791 file của UFPR04 dùng tag <Point> (hoa) thay vì <point> (thường) —
    lỗi gốc trong archive UFPR, không phải lỗi tải. Phải tìm cả hai, không chỉ 'point'.

    ~6.3% số file có ô thiếu `occupied` (trạng thái mơ hồ gốc). VẪN GIỮ vị trí ô (occupied=
    UNKNOWN_OCCUPIED=-1) thay vì bỏ hẳn — nếu bỏ, vị trí ô thật biến mất khỏi ground truth, khiến
    một prediction đúng ngay vị trí đó bị chấm là false positive một cách bất công (harness/
    baselines/windows đều lọc riêng các ô -1 ra làm vùng "ignore", không tính TP lẫn FP).

    `rotatedRect` (center/size/angle) đúng chuẩn `cv2.boxPoints`/`cv2.getRotationMatrix2D` (đã
    verify khớp contour thật, sai số ~5.7px/điểm do nhiễu gán nhãn — không cần đổi dấu góc). Dùng
    để cắt ảnh đã "dựng thẳng" ô (loại phần nền dư do box axis-aligned rộng hơn ô thật tới ~93%
    ở góc nghiêng lớn — xem report/data_processing_report.md).
    """
    spaces = ET.parse(xml_path).getroot().findall("space")
    out = []
    for sp in spaces:
        occ_raw = sp.get("occupied")
        occupied = UNKNOWN_OCCUPIED if occ_raw is None else int(occ_raw)
        contour = sp.find("contour")
        pts_el = contour.findall("point") or contour.findall("Point")
        pts = [(int(p.get("x")), int(p.get("y"))) for p in pts_el]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        rr = sp.find("rotatedRect")
        out.append({
            "id": int(sp.get("id")),
            "occupied": occupied,
            "x_min": min(xs), "y_min": min(ys), "x_max": max(xs), "y_max": max(ys),
            "rot_cx": float(rr.find("center").get("x")), "rot_cy": float(rr.find("center").get("y")),
            "rot_w": float(rr.find("size").get("w")), "rot_h": float(rr.find("size").get("h")),
            "rot_angle": float(rr.find("angle").get("d")),
        })
    return out


def axis_aligned_boxes(xml_path):
    """Trả về box (w, h) mỗi ô ĐÃ BIẾT occupied — dùng cho slot_size_report()/perspective_report().
    Loại ô UNKNOWN_OCCUPIED để không đổi số đo đã dùng chốt WINDOW_SIZE/SCALES (Ngày 1, đã khóa)."""
    return [(s["x_max"] - s["x_min"], s["y_max"] - s["y_min"]) for s in parse_spaces(xml_path)
            if s["occupied"] != UNKNOWN_OCCUPIED]


def _sample_xml(lot):
    """1 ảnh đại diện mỗi bãi là đủ — camera cố định nên vị trí + kích thước ô
    giống hệt nhau qua mọi ảnh cùng bãi (xem README § Vì sao phải chia theo bãi đỗ)."""
    files = sorted(glob.glob(str(LOT_ROOT / lot / "*" / "*" / "*.xml")))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy nhãn cho bãi {lot} ở {LOT_ROOT}")
    return files[len(files) // 2]


def _lot_images(lot):
    return sorted(glob.glob(str(LOT_ROOT / lot / "*" / "*" / "*.jpg")))


def slot_size_report(lots=None):
    """p50 kích thước ô — CHỈ tính trên train+val, không đụng test (leakage)."""
    lots = lots or config.TRAIN_LOTS + [l for l in config.VAL_LOTS if l not in config.TRAIN_LOTS]
    all_w, all_h = [], []
    per_lot = {}
    for lot in lots:
        boxes = axis_aligned_boxes(_sample_xml(lot))
        ws, hs = [b[0] for b in boxes], [b[1] for b in boxes]
        per_lot[lot] = {"n": len(boxes), "w_p50": st.median(ws), "h_p50": st.median(hs)}
        all_w += ws
        all_h += hs
    return {"per_lot": per_lot, "w_p50": st.median(all_w), "h_p50": st.median(all_h),
            "w_range": (min(all_w), max(all_w)), "h_range": (min(all_h), max(all_h))}


def _percentile(values, p):
    """Percentile kiểu "nearest-rank", không cần numpy."""
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(p / 100 * (len(s) - 1))))
    return s[k]


def perspective_report(lots=None, threshold=2.0):
    """Đo tỉ lệ méo phối cảnh: area p90 / area p10 trong cùng 1 bãi (KE_HOACH.md §7, §9).
    > threshold (mặc định 2.0) => cần multi-scale. CHỈ dùng train+val — không đụng test (leakage).
    """
    lots = lots or config.TRAIN_LOTS + [l for l in config.VAL_LOTS if l not in config.TRAIN_LOTS]
    per_lot = {}
    worst_ratio = 0.0
    for lot in lots:
        boxes = axis_aligned_boxes(_sample_xml(lot))
        areas = [w * h for w, h in boxes]
        p10, p90 = _percentile(areas, 10), _percentile(areas, 90)
        ratio = p90 / p10
        per_lot[lot] = ratio
        worst_ratio = max(worst_ratio, ratio)
    return {"per_lot": per_lot, "worst_ratio": worst_ratio, "needs_multiscale": worst_ratio > threshold}


def annotated_region(lot, n_samples=10):
    """Bbox vùng có nhãn của 1 bãi (+ margin CROP_MARGIN) — biến nhãn thưa (PUCPR ~100/300 ô)
    thành đầy đủ trong vùng cắt. Trả về (x_min, y_min, x_max, y_max).

    LƯU Ý: UFPR04 có camera bị xê dịch vị trí ít nhất 2 lần trong quá trình ghi hình (phát hiện
    bằng cách so khớp tâm ô giữa các ngày cách xa nhau — lệch tới ~94px, trong khi UFPR05/PUCPR
    xê dịch 0px suốt toàn bộ khoảng thời gian, tức camera THẬT SỰ cố định ở 2 bãi đó). Nếu chỉ
    dùng 1 ảnh đại diện (`_sample_xml()`) để tính vùng cắt như bản cũ, ô ở "kỳ" camera khác kỳ của
    ảnh đại diện có thể bị cắt cụt (đã phát hiện: 57/177 ảnh UFPR04 trong split, mỗi ảnh cụt ~1 ô).
    Sửa bằng cách lấy hợp (union) bbox từ NHIỀU ảnh đại diện trải đều theo thời gian, đủ phủ mọi
    kỳ camera có thể có — không ảnh hưởng UFPR05/PUCPR vì các ảnh của chúng có cùng 1 layout."""
    files = sorted(glob.glob(str(LOT_ROOT / lot / "*" / "*" / "*.xml")))
    if not files:
        raise FileNotFoundError(
            f"Không có file nhãn (.xml) nào cho bãi {lot!r} ở {LOT_ROOT / lot}.\n"
            "Bãi này cần TOÀN BỘ ảnh gốc, không chỉ vài ảnh mẫu — vùng cắt lấy hợp bbox từ nhiều "
            "ảnh trải đều theo thời gian để phủ mọi kỳ camera.\n"
            "Chạy: bash scripts/download_data.sh raw")
    idxs = sorted(set(round(i * (len(files) - 1) / max(1, n_samples - 1)) for i in range(n_samples)))
    x_min = y_min = float("inf")
    x_max = y_max = float("-inf")
    for i in idxs:
        spaces = parse_spaces(files[i])
        x_min = min(x_min, min(s["x_min"] for s in spaces))
        y_min = min(y_min, min(s["y_min"] for s in spaces))
        x_max = max(x_max, max(s["x_max"] for s in spaces))
        y_max = max(y_max, max(s["y_max"] for s in spaces))
    margin = round(config.CROP_MARGIN * max(x_max - x_min, y_max - y_min))
    return (max(0, x_min - margin), max(0, y_min - margin), x_max + margin, y_max + margin)


_MARGIN_REGION_LOTS = ("PUCPR",)  # CHỈ áp dụng cho bãi đã verify bằng mắt là có xe thật chưa gán
                                   # nhãn ở rìa. Đã thử áp dụng cho UFPR04/UFPR05 và render lên ảnh
                                   # thật để kiểm tra: vùng phát hiện ra ở 2 bãi đó chỉ là mặt đường
                                   # trống (background thật), KHÔNG phải xe chưa gán nhãn — do layout
                                   # 2 bãi này nghiêng/không xếp hàng ngang đơn giản như PUCPR, khiến
                                   # cách nhóm theo hàng bên dưới cho kết quả sai (dương tính giả).
                                   # KHÔNG thêm bãi khác vào đây nếu chưa verify lại bằng ảnh thật.


def unlabeled_margin_regions(lot, row_gap=40):
    """Vùng lề CHƯA gán nhãn ở rìa mỗi hàng ô — CHỈ áp dụng cho bãi trong _MARGIN_REGION_LOTS.
    Phát hiện bằng cách vẽ nhãn lên ảnh thật: PUCPR chỉ gán ~20 ô/hàng nhưng ảnh cho thấy còn nhiều
    xe đỗ thêm về phía rìa mỗi hàng, không có nhãn nào (xem report/data_processing_report.md).
    annotated_region() chỉ cắt theo 1 bbox chung nên không loại được phần lề này — nó vẫn nằm trong
    vùng đã cắt.

    Nhóm ô theo HÀNG (cụm theo trục y, do camera nghiêng nên mỗi hàng có mép trái thật khác nhau,
    không thể dùng 1 mép chung). Với mỗi hàng, vùng từ mép trái của annotated_region() tới ô có
    nhãn gần mép trái nhất trong hàng đó = vùng "có thể có xe nhưng không rõ nhãn".

    Trả về list (x_min, y_min, x_max, y_max) theo hệ toạ độ ảnh GỐC (như annotated_region()) — build_gt_rows()
    sẽ trừ offset crop như mọi box khác. Rỗng nếu bãi không trong _MARGIN_REGION_LOTS, hoặc hàng đã
    sát mép crop (không có gì đáng kể để thêm).
    """
    if lot not in _MARGIN_REGION_LOTS:
        return []
    spaces = parse_spaces(_sample_xml(lot))
    region = annotated_region(lot)
    sorted_by_y = sorted(spaces, key=lambda s: s["rot_cy"])
    rows, current = [], [sorted_by_y[0]]
    for s in sorted_by_y[1:]:
        if s["rot_cy"] - current[-1]["rot_cy"] > row_gap:
            rows.append(current)
            current = [s]
        else:
            current.append(s)
    rows.append(current)

    regions = []
    for row in rows:
        row_x_min = min(s["x_min"] for s in row)
        row_y_min = min(s["y_min"] for s in row)
        row_y_max = max(s["y_max"] for s in row)
        if row_x_min - region[0] > 20:  # khoảng trống thật sự đáng kể, không phải sai số làm tròn
            regions.append((region[0], row_y_min, row_x_min, row_y_max))
    return regions


def _parse_timestamp(path):
    return datetime.strptime(Path(path).stem, "%Y-%m-%d_%H_%M_%S")


def temporal_subsample(image_paths, minutes=None):
    """Giữ 1 ảnh mỗi `minutes` phút (mặc định config.TEMPORAL_MINUTES) theo mốc thời gian trong tên file.
    Chống leakage do ảnh gần trùng (chụp 5 phút/lần) — xem README § Bốn đặc thù PKLot."""
    minutes = minutes if minutes is not None else config.TEMPORAL_MINUTES
    dated = sorted(((_parse_timestamp(p), p) for p in image_paths), key=lambda t: t[0])
    kept, last_kept = [], None
    for ts, p in dated:
        if last_kept is None or (ts - last_kept).total_seconds() >= minutes * 60:
            kept.append(p)
            last_kept = ts
    return kept


def _date_of(path):
    """Ngày THẬT lấy từ tên file (timestamp), KHÔNG lấy từ tên thư mục cha.

    Phát hiện 2 thư mục ngày bị đặt sai tên trong archive gốc (không phải lỗi tải):
    - UFPR05/Sunny/2013-14-16/ — tên thư mục có tháng không hợp lệ (14), toàn bộ 53 file thật
      ra là 2013-04-16.
    - PUCPR/Cloudy/2012-09-16/ — tên thư mục ghi 16/09, toàn bộ 145 file thật ra là 2012-10-16
      (khác hẳn 1 tháng), không có thư mục 2012-10-16 riêng nào khác chứa dữ liệu này.
    Nếu gộp theo tên thư mục, ảnh của 1 ngày thật bị trộn nhầm vào nhóm ngày khác (ảnh hưởng
    build_split() khi lot đó là VAL_LOTS). Dùng ngày trong tên file để tránh lỗi này ở mọi bãi.
    """
    return Path(path).stem[:10]  # "YYYY-MM-DD_HH_MM_SS" -> "YYYY-MM-DD"


def _weather_of(path):
    return Path(path).parts[-3]  # .../<lot>/<weather>/<date>/<file>.jpg — 1 ngày chỉ 1 thời tiết


def build_split(n_train=None, n_val=None, n_test=None, val_frac=0.15, seed=None):
    """Chia ảnh theo bãi (KE_HOACH.md §7 Ngày 2) — chỉ chạy 1 lần rồi khóa vào splits.json.

    - TEST: toàn bộ từ TEST_LOTS (PUCPR), tách biệt hoàn toàn theo bãi.
    - VAL: từ VAL_LOTS, giữ theo NGÀY riêng (không trộn ngày với train) — chống leakage thời gian
      khi VAL_LOTS là tập con của TRAIN_LOTS (VD: UFPR04 vừa train vừa val).
    - TRAIN: từ TRAIN_LOTS, loại các ngày đã dùng cho VAL ở bãi trùng.
    Random chỉ dùng để CHỌN ảnh trong từng nhóm đã tách theo bãi/ngày — không dùng để định biên nhóm.

    Mặc định (n_train/n_val/n_test=None) lấy TOÀN BỘ ảnh đã lấy mẫu thời gian (temporal_subsample)
    của nhóm đó — đã bỏ giới hạn N_IMAGES=150 cũ (xem report/data_processing_report.md § 4).
    Khi n_val=None, val_frac quyết định tỉ lệ NGÀY của val_lot dành cho val (mặc định 15%), CHIA
    THEO TỪNG NHÓM THỜI TIẾT riêng (Cloudy/Rainy/Sunny) — tránh val bị lệch tỉ lệ thời tiết so với
    train do 1 ngày chỉ có 1 nhãn thời tiết (nếu chia ngày mù thời tiết, dễ dồn hết ngày Rainy hiếm
    vào 1 bên). Phần ngày còn lại của bãi đó thuộc về train. Truyền số cụ thể (VD n_train=90) để
    tái lập split cũ (khi đó KHÔNG cân bằng thời tiết, giữ đúng hành vi gốc).
    """
    seed = seed if seed is not None else config.RANDOM_SEED
    rng = random.Random(seed)

    def pool(lot):
        return temporal_subsample(_lot_images(lot))

    val_lot = config.VAL_LOTS[0]
    val_pool = pool(val_lot)
    by_date = defaultdict(list)
    for p in val_pool:
        by_date[_date_of(p)].append(p)
    dates = sorted(by_date)
    rng.shuffle(dates)

    if n_val is None:
        dates_by_weather = defaultdict(list)
        for d in dates:
            dates_by_weather[_weather_of(by_date[d][0])].append(d)
        val_dates = set()
        for weather, wdates in sorted(dates_by_weather.items()):
            wdates = list(wdates)
            rng.shuffle(wdates)
            n_w_val = max(1, round(len(wdates) * val_frac))
            val_dates.update(wdates[:n_w_val])
        val_files = sorted(p for d in val_dates for p in by_date[d])
    else:
        val_dates, val_candidates = set(), []
        for d in dates:
            if len(val_candidates) >= n_val:
                break
            val_candidates += by_date[d]
            val_dates.add(d)
        val_files = rng.sample(val_candidates, min(n_val, len(val_candidates)))

    train_pool = []
    for lot in config.TRAIN_LOTS:
        imgs = pool(lot)
        if lot == val_lot:
            imgs = [p for p in imgs if _date_of(p) not in val_dates]
        train_pool += imgs
    train_files = train_pool if n_train is None else rng.sample(train_pool, min(n_train, len(train_pool)))

    test_pool = []
    for lot in config.TEST_LOTS:
        test_pool += pool(lot)
    test_files = test_pool if n_test is None else rng.sample(test_pool, min(n_test, len(test_pool)))

    return {"train": sorted(train_files), "val": sorted(val_files), "test": sorted(test_files)}


def build_full_split(base_split=None):
    """Mở rộng 1 split đã khóa (mặc định đọc từ splits.json) thành bản gồm TOÀN BỘ ảnh gốc — không
    qua temporal_subsample(). Dùng lại ĐÚNG biên ngày val đã khóa (suy ra từ base_split), không random
    lại, không đổi ranh giới train/val/test — chỉ thêm lại các ảnh gần trùng mà temporal_subsample()
    đã lược bớt. raw/ không hề bị đụng tới, chỉ tham chiếu thêm nhiều file hơn trong index.

    Kết quả để tham khảo/dự phòng (VD train_model.py muốn thử toàn bộ dữ liệu) — KHÔNG thay thế
    splits.json chính, không chạy qua cổng kiểm tra Baseline A/harness như splits.json (xem
    report/data_processing_report.md § 6).
    """
    base_split = base_split or load_split()
    val_lot = config.VAL_LOTS[0]
    val_dates = {_date_of(p) for p in base_split["val"]}

    def raw(lot):
        return sorted(_lot_images(lot))

    train_full = []
    for lot in config.TRAIN_LOTS:
        imgs = raw(lot)
        if lot == val_lot:
            imgs = [p for p in imgs if _date_of(p) not in val_dates]
        train_full += imgs

    val_full = [p for p in raw(val_lot) if _date_of(p) in val_dates]

    test_full = []
    for lot in config.TEST_LOTS:
        test_full += raw(lot)

    return {"train": sorted(train_full), "val": sorted(val_full), "test": sorted(test_full)}


def _lot_of(path):
    # LOT_ROOT / <lot> / <weather> / <date> / <file>.jpg
    return Path(path).relative_to(LOT_ROOT).parts[0]


def save_split(split, path=None, overwrite=False):
    """KHÓA VĨNH VIỄN (KE_HOACH.md § Bốn quy tắc #2) — mặc định TỪ CHỐI ghi đè split đã có.
    Chỉ truyền overwrite=True khi thật sự cố ý làm lại từ đầu (và biết mọi feature/model cũ đã lỗi thời)."""
    path = Path(path) if path else config.PROC / "splits.json"
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} đã tồn tại — split đã KHÓA theo quy tắc #2, không được chia lại. "
            "Nếu thật sự cần làm lại (hiếm, và phải trích lại toàn bộ feature), gọi với overwrite=True."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    rel = {name: [str(Path(p).relative_to(LOT_ROOT)) for p in paths] for name, paths in split.items()}
    with open(path, "w") as f:
        json.dump(rel, f, indent=2, ensure_ascii=False)
    return path


def load_split(path=None):
    path = Path(path) if path else config.PROC / "splits.json"
    with open(path) as f:
        rel = json.load(f)
    return {name: [str(LOT_ROOT / p) for p in paths] for name, paths in rel.items()}


def save_crops(lots=None, path=None):
    """Lưu vùng cắt (annotated_region) mỗi bãi vào processed/crops.json."""
    lots = lots or sorted(set(config.TRAIN_LOTS + config.VAL_LOTS + config.TEST_LOTS))
    crops = {lot: list(annotated_region(lot)) for lot in lots}
    path = Path(path) if path else config.PROC / "crops.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(crops, f, indent=2)
    return path


def build_gt_rows(split):
    """Sinh nhãn (image_id,x_min,y_min,x_max,y_max,label,lot,slot_id,split,rot_*) cho toàn bộ ảnh
    trong split. Toạ độ box (kể cả tâm rotatedRect) đã trừ offset annotated_region() -> hệ ảnh
    ĐÃ CẮT (đúng Quy ước README). rot_w/rot_h/rot_angle không đổi qua phép tịnh tiến.

    Với bãi trong _MARGIN_REGION_LOTS, mỗi ảnh còn được thêm 1 dòng "ignore" (label=UNKNOWN_OCCUPIED,
    slot_id âm để không trùng slot_id thật) cho mỗi vùng lề chưa gán nhãn (unlabeled_margin_regions())
    — coi như 1 ô ảo bao trọn cả vùng, để hệ thống ignore-region (containment, không phải IoU) xử lý
    đúng dù vùng này lớn hơn hẳn 1 ô/cửa sổ thật."""
    crop_cache = {}
    margin_cache = {}
    rows = []
    for split_name, paths in split.items():
        for img_path in paths:
            lot = _lot_of(img_path)
            if lot not in crop_cache:
                crop_cache[lot] = annotated_region(lot)
                margin_cache[lot] = unlabeled_margin_regions(lot)
            cx0, cy0, _, _ = crop_cache[lot]
            xml_path = str(Path(img_path).with_suffix(".xml"))
            if not Path(xml_path).exists():
                continue  # 1 ảnh PUCPR thiếu nhãn (đã xác nhận thủ công) — bỏ qua, không suy diễn
            image_id = Path(img_path).stem
            for s in parse_spaces(xml_path):
                rows.append({
                    "image_id": image_id, "lot": lot, "slot_id": s["id"], "split": split_name,
                    "x_min": s["x_min"] - cx0, "y_min": s["y_min"] - cy0,
                    "x_max": s["x_max"] - cx0, "y_max": s["y_max"] - cy0,
                    "label": s["occupied"],
                    "rot_cx": s["rot_cx"] - cx0, "rot_cy": s["rot_cy"] - cy0,
                    "rot_w": s["rot_w"], "rot_h": s["rot_h"], "rot_angle": s["rot_angle"],
                })
            for i, (mx0, my0, mx1, my1) in enumerate(margin_cache[lot]):
                rows.append({
                    "image_id": image_id, "lot": lot, "slot_id": -(i + 1), "split": split_name,
                    "x_min": mx0 - cx0, "y_min": my0 - cy0,
                    "x_max": mx1 - cx0, "y_max": my1 - cy0,
                    "label": UNKNOWN_OCCUPIED,
                    "rot_cx": (mx0 + mx1) / 2 - cx0, "rot_cy": (my0 + my1) / 2 - cy0,
                    "rot_w": mx1 - mx0, "rot_h": my1 - my0, "rot_angle": 0.0,
                })
    return rows


def main():
    """CLI chuẩn bị dữ liệu: raw/PKLot -> processed/{splits,crops}.json + gt.csv.

    Trước 04/09 bước này KHÔNG có entry point nào — chỉ tồn tại dưới dạng đoạn code rời trong
    notebooks/p4_models.ipynb, nên không ai chạy lại được cả quy trình từ đầu.
    """
    import argparse

    import pandas as pd

    ap = argparse.ArgumentParser(description="Chuẩn bị dữ liệu PKLot (P2, Ngày 1-2)")
    ap.add_argument("--overwrite-split", action="store_true",
                    help="Chia lại split. MẶC ĐỊNH TỪ CHỐI — quy tắc #2: split khóa vĩnh viễn. "
                         "Chia lại làm mọi feature và model đã có trở nên lỗi thời.")
    ap.add_argument("--skip-gt", action="store_true", help="chỉ làm split + crops")
    a = ap.parse_args()

    if not LOT_ROOT.exists():
        raise SystemExit(f"Không có ảnh gốc ở {LOT_ROOT}.\n"
                         "Chạy: bash scripts/download_data.sh raw    (~2GB)")

    split_path = config.PROC / "splits.json"
    if split_path.exists() and not a.overwrite_split:
        split = load_split()
        print(f"[1/3] splits.json đã có (KHÓA) — dùng lại: "
              + " ".join(f"{k}={len(v)}" for k, v in split.items()))
    else:
        split = build_split()
        save_split(split, overwrite=a.overwrite_split)
        print(f"[1/3] đã chia split: " + " ".join(f"{k}={len(v)}" for k, v in split.items()))

    print(f"[2/3] crops.json -> {save_crops()}")
    for lot, box in json.loads((config.PROC / 'crops.json').read_text()).items():
        print(f"        {lot}: {box}")

    if a.skip_gt:
        return
    rows = build_gt_rows(split)
    df = pd.DataFrame(rows)
    out = config.PROC / "gt.csv"
    df.to_csv(out, index=False)
    print(f"[3/3] gt.csv -> {out}  ({len(df):,} dòng)")
    print("        nhãn:", df.label.value_counts().to_dict(), " (0=ô trống, 1=có xe, -1=không rõ)")
    print("        ảnh :", df.groupby('split').image_id.nunique().to_dict())


if __name__ == "__main__":
    main()
