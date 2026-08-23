"""build_dataset.py — Sinh feature theo shard, lưu parquet
Chủ sở hữu: P2
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import config
import features
import pklot_data
import windows


def _load_crop(img_path, crop_box):
    img = Image.open(img_path).convert("RGB")
    return np.array(img.crop(crop_box))  # crop_box: (x0,y0,x1,y1)


def _rotated_crop(image, cx, cy, w, h, angle):
    """Cắt ảnh 'dựng thẳng' 1 ô nghiêng bằng rotatedRect thật (đã verify khớp cv2.boxPoints/
    getRotationMatrix2D, xem pklot_data.parse_spaces), thay vì box axis-aligned dư nền tới ~93%
    ở góc nghiêng lớn (report/data_processing_report.md)."""
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(image, m, (image.shape[1], image.shape[0]))
    return cv2.getRectSubPix(rotated, (max(1, round(w)), max(1, round(h))), (cx, cy))


def build_shard(shard_id, n_shards, seed=None):
    """Trích feature cho 1 shard ảnh (chia đều & xen kẽ từ processed/splits.json).
    Lưu features/shard_<id>_of_<n>.parquet. Mỗi người chạy 1-2 shard (KE_HOACH.md §7 Ngày 3)."""
    seed = seed if seed is not None else config.RANDOM_SEED
    rng = np.random.default_rng(seed + shard_id)

    split = pklot_data.load_split()
    path_split = {p: name for name, paths in split.items() for p in paths}
    all_paths = sorted(path_split)
    my_paths = all_paths[shard_id::n_shards]

    gt = pd.read_csv(config.PROC / "gt.csv")
    crops = json.load(open(config.PROC / "crops.json"))

    rows = []
    for img_path in my_paths:
        image_id = Path(img_path).stem
        lot = pklot_data._lot_of(img_path)
        g = gt[gt.image_id == image_id]
        if g.empty:
            continue  # ảnh thiếu nhãn (đã xác nhận: 1 ảnh PUCPR) -> bỏ qua

        gt_boxes = list(zip(g.x_min, g.y_min, g.x_max, g.y_max))
        gt_labels = list(g.label)
        gt_rot = list(zip(g.rot_cx, g.rot_cy, g.rot_w, g.rot_h, g.rot_angle))

        cx0, cy0, cx1, cy1 = crops[lot]
        crop = _load_crop(img_path, (cx0, cy0, cx1, cy1))
        ch, cw = crop.shape[:2]

        win_boxes = list(windows.slide_windows(cw, ch))
        labeled = windows.label_windows(win_boxes, gt_boxes, gt_labels)

        for wb, (cls, idx) in zip(win_boxes, labeled):
            if cls == "ignore":
                continue
            if cls == "background" and rng.random() > config.NEG_SAMPLE_RATE:
                continue
            x0, y0, x1, y1 = wb
            if idx is not None:
                # car/empty: cắt "dựng thẳng" theo rotatedRect thật — loại nền dư của box axis-aligned
                rcx, rcy, rw, rh, rangle = gt_rot[idx]
                win_crop = _rotated_crop(crop, rcx, rcy, rw, rh, rangle)
            else:
                win_crop = crop[y0:y1, x0:x1]
            if win_crop.shape[:2] != (config.WINDOW_SIZE, config.WINDOW_SIZE):
                win_crop = np.array(Image.fromarray(win_crop).resize((config.WINDOW_SIZE, config.WINDOW_SIZE)))
            vec = features.extract(win_crop)
            rows.append([image_id, lot, path_split[img_path], cls, x0, y0, x1, y1] + vec.tolist())

    cols = ["image_id", "lot", "split", "class", "x_min", "y_min", "x_max", "y_max"] + features.feature_names()
    df = pd.DataFrame(rows, columns=cols)
    df[features.feature_names()] = df[features.feature_names()].astype(np.float32)  # float64->32: nửa dung lượng, không mất độ chính xác thực dụng

    out_dir = config.FEAT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{shard_id:02d}_of_{n_shards}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path, len(df), len(my_paths)


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--n-shards", type=int, default=8)
    args = parser.parse_args()

    t0 = time.perf_counter()
    path, n_rows, n_imgs = build_shard(args.shard, args.n_shards)
    print(f"shard {args.shard}/{args.n_shards}: {n_imgs} ảnh -> {n_rows} window, "
          f"{time.perf_counter() - t0:.1f}s -> {path}")
