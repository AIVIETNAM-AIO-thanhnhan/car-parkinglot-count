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
import infer
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


def build_shard(shard_id, n_shards, seed=None, axis_aligned=False, out_dir=None, n_jobs=None):
    """Trích feature cho 1 shard ảnh (chia đều & xen kẽ từ processed/splits.json).
    Lưu <out_dir>/shard_<id>_of_<n>.parquet. Mỗi người chạy 1-2 shard (KE_HOACH.md §7 Ngày 3).

    axis_aligned=False (mặc định, bản gốc): ô car/empty được cắt theo rotatedRect ĐÃ XOAY THẲNG
        lấy từ nhãn. Cho accuracy mức cửa sổ cao nhưng dùng thông tin mà lúc suy luận KHÔNG CÓ.

    axis_aligned=True: ô car/empty cắt bằng chính KHUNG TRƯỢT vuông góc, y hệt lúc suy luận.

    🔴 VÌ SAO CÓ CỜ NÀY. Đo trên 60 ảnh train / 10 ảnh val, cùng thuật toán, chỉ khác cách cắt:

                                     xoay thẳng   vuông góc
        accuracy cửa sổ (cắt cùng kiểu)   0.922      0.907
        ô đỗ nhận đúng khi TỰ DÒ          0.000      0.779
        mAP_macro khi TỰ DÒ               0.0054     0.7625

    Bản xoay thẳng nhận đúng 0 ô khi trượt cửa sổ — gọi 100% là nền — vì mọi positive nó từng
    thấy đều đã được xoay ngay ngắn nhờ nhãn. Đổi sang vuông góc chỉ mất 1,5 điểm accuracy mức
    cửa sổ nhưng làm chế độ tự dò từ vô dụng thành dùng được.
    """
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

        # Gom trước rồi trích một lượt: infer.extract_batch chia lô và chạy song song, nhanh ~8x
        # so với gọi features.extract() từng cửa sổ (joblib phải pickle lại ảnh cho mỗi tác vụ).
        # rng.random() vẫn được gọi ĐÚNG THỨ TỰ như bản cũ để tái lập được bộ nền đã lấy mẫu.
        boxes, meta = [], []
        for wb, (cls, idx) in zip(win_boxes, labeled):
            if cls == "ignore":
                continue
            if cls == "background" and rng.random() > config.NEG_SAMPLE_RATE:
                continue
            x0, y0, x1, y1 = wb
            if idx is not None and not axis_aligned:
                # car/empty: cắt "dựng thẳng" theo rotatedRect thật — loại nền dư của box axis-aligned
                rcx, rcy, rw, rh, rangle = gt_rot[idx]
                boxes.append(dict(zip(infer.ROT_KEYS, (rcx, rcy, rw, rh, rangle))))
            else:
                boxes.append((x0, y0, x1, y1))
            meta.append([image_id, lot, path_split[img_path], cls, x0, y0, x1, y1])
        if not boxes:
            continue
        vecs = infer.extract_batch(crop, boxes, n_jobs=n_jobs)
        rows.extend(m + v.tolist() for m, v in zip(meta, vecs))

    cols = ["image_id", "lot", "split", "class", "x_min", "y_min", "x_max", "y_max"] + features.feature_names()
    df = pd.DataFrame(rows, columns=cols)
    df[features.feature_names()] = df[features.feature_names()].astype(np.float32)  # float64->32: nửa dung lượng, không mất độ chính xác thực dụng

    out_dir = Path(out_dir) if out_dir else config.FEAT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{shard_id:02d}_of_{n_shards}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path, len(df), len(my_paths)


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, help="chỉ số shard; bỏ trống = chạy hết")
    parser.add_argument("--n-shards", type=int, default=8)
    parser.add_argument("--axis-aligned", action="store_true",
                        help="cắt ô car/empty bằng khung trượt vuông góc thay vì rotatedRect. "
                             "BẮT BUỘC nếu muốn chế độ tự dò chạy được — xem docstring build_shard().")
    parser.add_argument("--out-dir", default=None, help="mặc định config.FEAT")
    parser.add_argument("--n-jobs", type=int, default=None, help="mặc định config.N_JOBS")
    args = parser.parse_args()

    shards = [args.shard] if args.shard is not None else range(args.n_shards)
    t_all = time.perf_counter()
    total = 0
    for s in shards:
        t0 = time.perf_counter()
        path, n_rows, n_imgs = build_shard(s, args.n_shards, axis_aligned=args.axis_aligned,
                                           out_dir=args.out_dir, n_jobs=args.n_jobs)
        total += n_rows
        print(f"shard {s}/{args.n_shards}: {n_imgs} ảnh -> {n_rows:,} cửa sổ, "
              f"{time.perf_counter() - t0:.0f}s -> {path}", flush=True)
    if len(list(shards)) > 1:
        print(f"\nXONG {total:,} cửa sổ trong {(time.perf_counter() - t_all)/60:.1f} phút"
              f"  (cắt {'VUÔNG GÓC' if args.axis_aligned else 'xoay thẳng'})")
