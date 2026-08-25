"""build_features.py — Chạy CẢ pipeline: (ảnh + XML)  ->  features.parquet.  Owner: P3.

    ảnh + XML  ->  cửa sổ trượt  ->  gán nhãn IoU  ->  lấy mẫu nền  ->  trích 395 feature  ->  parquet

Đây là "thành phẩm" của Data Pipeline. P4 chỉ cần đọc file parquet này để train.

Dùng:
    python build_features.py --image a.jpg --xml a.xml --out features.parquet
    python build_features.py --folder path/to/PKLot_subset --out features.parquet
"""
import os, glob, argparse
import numpy as np
import pandas as pd
import cv2
import xml.etree.ElementTree as ET

import windows
from features import extract_features, FEATURE_NAMES

# --- hằng số pipeline (khớp config.py của P1) ---
STRIDE          = 16
IOU_POSITIVE    = 0.5
IOU_IGNORE      = 0.3
NEG_SAMPLE_RATE = 0.10
RANDOM_SEED     = 42


def parse_xml(xml_path):
    """PKLot XML -> (boxes Nx4 axis-aligned, labels N,).  Bỏ ô không có 'occupied'."""
    root = ET.parse(xml_path).getroot()
    boxes, labels = [], []
    for sp in root.findall("space"):
        occ = sp.get("occupied")
        cont = sp.find("contour")
        if occ is None or cont is None:
            continue
        xs = [int(p.get("x")) for p in cont.findall("point")]
        ys = [int(p.get("y")) for p in cont.findall("point")]
        boxes.append([min(xs), min(ys), max(xs), max(ys)])
        labels.append(int(occ))                       # 0=empty, 1=occupied
    return np.array(boxes, dtype=np.int32), np.array(labels, dtype=np.int32)


def choose_window_size(boxes, override=None):
    """WINDOW_SIZE = trung vị cạnh dài của ô thật (nếu P2 chưa chốt)."""
    if override:
        return int(override)
    sides = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    return int(np.median(sides))


def process_image(image_path, xml_path, image_id=None, window_size=None):
    image_id = image_id or os.path.splitext(os.path.basename(image_path))[0]
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)        # QUAN TRỌNG: OpenCV đọc BGR -> đổi sang RGB
    H, W = rgb.shape[:2]

    gt_boxes, gt_labels = parse_xml(xml_path)
    win_size = choose_window_size(gt_boxes, window_size)

    wins = windows.generate_windows(H, W, win_size, STRIDE)
    labs = windows.label_windows(wins, gt_boxes, gt_labels, IOU_POSITIVE, IOU_IGNORE)
    wins, labs = windows.sample_windows(wins, labs, NEG_SAMPLE_RATE, RANDOM_SEED)

    feats = np.zeros((len(wins), 395), dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(wins):
        feats[i] = extract_features(rgb[y1:y2, x1:x2])

    meta = pd.DataFrame({
        "image_id": image_id,
        "x1": wins[:, 0], "y1": wins[:, 1], "x2": wins[:, 2], "y2": wins[:, 3],
        "label": labs.astype(np.int8),                # 0=empty, 1=occupied, 2=background
    })
    df = pd.concat([meta, pd.DataFrame(feats, columns=FEATURE_NAMES)], axis=1)
    return df, win_size


def process_folder(folder, window_size=None):
    """Ghép mọi cặp (*.jpg, *.xml) cùng tên trong thư mục."""
    frames = []
    for xmlp in sorted(glob.glob(os.path.join(folder, "*.xml"))):
        stem = os.path.splitext(xmlp)[0]
        imgp = next((stem + e for e in (".jpg", ".jpeg", ".png") if os.path.exists(stem + e)), None)
        if imgp is None:
            print("  (bỏ qua, thiếu ảnh):", os.path.basename(xmlp)); continue
        df, _ = process_image(imgp, xmlp, window_size=window_size)
        frames.append(df)
        print(f"  {os.path.basename(imgp)}: {len(df)} cửa sổ")
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image"); ap.add_argument("--xml")
    ap.add_argument("--folder")
    ap.add_argument("--out", default="features.parquet")
    ap.add_argument("--window-size", type=int, default=None)
    a = ap.parse_args()

    if a.folder:
        df = process_folder(a.folder, a.window_size)
    else:
        df, ws = process_image(a.image, a.xml, window_size=a.window_size)
        print("WINDOW_SIZE dùng:", ws)

    df.to_parquet(a.out)
    print(f"\n✅ Đã lưu {a.out}: {df.shape[0]} dòng x {df.shape[1]} cột")
    print("Phân bố nhãn (0=trống,1=có xe,2=nền):")
    print(df["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
