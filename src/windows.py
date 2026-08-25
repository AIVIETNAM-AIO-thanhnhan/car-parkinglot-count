"""windows.py — Sliding windows + IoU labeling.  Owner: P3 (Pipeline Engineer).

Turns a parking-lot image + its ground-truth boxes (parsed from the PKLot XML)
into labeled candidate windows for training.

Label scheme (khung nhãn):
    1 = occupied (co xe)      -> window overlaps an occupied GT box
    0 = empty    (o trong)    -> window overlaps an empty GT box
    2 = background (nen)      -> window overlaps nothing
   -1 = ignore   (bo qua)     -> overlap in the "grey zone" -> dropped from training
"""
import numpy as np


# ----------------------------------------------------------------------
# 1) Sinh cua so truot / generate candidate windows
# ----------------------------------------------------------------------
def generate_windows(img_h, img_w, window_size, stride, scales=(1.0,)):
    """Return an (N,4) array of (x1, y1, x2, y2) windows tiled over the image."""
    wins = []
    for s in scales:
        w = int(round(window_size * s))
        for y in range(0, img_h - w + 1, stride):
            for x in range(0, img_w - w + 1, stride):
                wins.append((x, y, x + w, y + w))
    return np.asarray(wins, dtype=np.int32)


# ----------------------------------------------------------------------
# 2) IoU cua moi cua so voi moi box that / IoU of every window vs every GT box
# ----------------------------------------------------------------------
def iou_matrix(windows, gt_boxes):
    """windows:(N,4), gt_boxes:(G,4) -> IoU matrix (N,G)."""
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


# ----------------------------------------------------------------------
# 3) Gan nhan cho tung cua so / label each window by IoU
# ----------------------------------------------------------------------
def label_windows(windows, gt_boxes, gt_labels, iou_pos=0.5, iou_ignore=0.3):
    """gt_labels: (G,) array of 0=empty / 1=occupied, aligned with gt_boxes.
    Returns (N,) int8 labels using the scheme in the module docstring."""
    ious = iou_matrix(windows, gt_boxes)                 # (N,G)
    labels = np.full(len(windows), 2, dtype=np.int8)     # default = background
    if ious.shape[1] == 0:
        return labels
    best = ious.max(axis=1)                              # best overlap per window
    arg  = ious.argmax(axis=1)                           # which GT box it matched
    pos = best >= iou_pos
    labels[pos] = gt_labels[arg[pos]].astype(np.int8)    # 0 or 1 from matched box
    ign = (best >= iou_ignore) & (best < iou_pos)
    labels[ign] = -1                                     # grey zone -> ignore
    return labels


# ----------------------------------------------------------------------
# 4) Lay mau negative / down-sample background windows
# ----------------------------------------------------------------------
def sample_windows(windows, labels, neg_rate=0.10, seed=42):
    """Keep all occupied/empty windows, drop all -1, keep only neg_rate of background."""
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(labels), dtype=bool)
    keep[(labels == 0) | (labels == 1)] = True           # keep real classes
    bg = np.where(labels == 2)[0]
    chosen = rng.choice(bg, size=int(len(bg) * neg_rate), replace=False) if len(bg) else []
    keep[chosen] = True
    return windows[keep], labels[keep]


# ----------------------------------------------------------------------
# Demo (chay thu tren du lieu gia / run on a synthetic sample)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 3 fake GT spots: occupied, empty, occupied
    gt_boxes  = np.array([[40, 30, 92, 86], [100, 30, 152, 86], [160, 32, 214, 88]])
    gt_labels = np.array([1, 0, 1])                      # from XML 'occupied' attr

    wins = generate_windows(img_h=580, img_w=1020, window_size=52, stride=16)
    labs = label_windows(wins, gt_boxes, gt_labels, iou_pos=0.5, iou_ignore=0.3)
    wins_s, labs_s = sample_windows(wins, labs, neg_rate=0.10)

    n = len(wins)
    print(f"windows generated : {n}")
    print(f"  occupied (1)    : {(labs == 1).sum()}")
    print(f"  empty    (0)    : {(labs == 0).sum()}")
    print(f"  ignore   (-1)   : {(labs == -1).sum()}")
    print(f"  background(2)   : {(labs == 2).sum()}")
    print(f"after neg-sampling: {len(wins_s)} windows kept")
