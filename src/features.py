"""features.py — Trích 395 đặc trưng cho mỗi cửa sổ.  Owner: P3 (Pipeline Engineer).

395 = hog_(324) + color_(54) + lbp_(10) + tex_(7)

    hog_*   : Histogram of Oriented Gradients  -> cạnh / hình dạng (shape/edges)
    color_* : histogram màu HSV                -> màu (color)          << tín hiệu chính
    lbp_*   : Local Binary Pattern (uniform)   -> kết cấu vi mô (micro-texture)
    tex_*   : thống kê GLCM                     -> kết cấu (texture)

Ô CÓ XE và ô TRỐNG cùng hình chữ nhật -> khác nhau ở MÀU và KẾT CẤU, không phải hình dạng.
"""
import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

FEAT_SIZE = 64          # mọi cửa sổ resize về 64x64 trước khi trích (giữ số chiều ổn định)
LBP_P, LBP_R = 8, 1     # -> P+2 = 10 bins
COLOR_BINS = 18         # 3 kênh HSV x 18 = 54
GLCM_LEVELS = 32

# ---- tên cột cố định (P4 dùng cho SHAP, P3 dùng cho ablation) ----
TEX_NAMES = ["tex_contrast", "tex_dissimilarity", "tex_homogeneity",
             "tex_asm", "tex_energy", "tex_correlation", "tex_entropy"]
FEATURE_NAMES = (
    [f"hog_{i}"   for i in range(324)] +
    [f"color_{i}" for i in range(54)]  +
    [f"lbp_{i}"   for i in range(10)]  +
    TEX_NAMES
)
assert len(FEATURE_NAMES) == 395


def _hog(gray):
    return hog(gray, orientations=9, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2), block_norm="L2-Hys", channel_axis=None)  # -> 324


def _color(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ranges = [(0, 180), (0, 256), (0, 256)]                 # H, S, V ranges in OpenCV
    feats = []
    for ch, (lo, hi) in enumerate(ranges):
        h, _ = np.histogram(hsv[:, :, ch], bins=COLOR_BINS, range=(lo, hi))
        feats.append(h / (h.sum() + 1e-9))
    return np.concatenate(feats)                            # -> 54


def _lbp(gray):
    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method="uniform")
    n_bins = LBP_P + 2
    h, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    return h / (h.sum() + 1e-9)                              # -> 10


def _tex(gray):
    q = (gray.astype(np.uint16) * GLCM_LEVELS // 256).astype(np.uint8)   # quantize -> 32 levels
    glcm = graycomatrix(q, distances=[1], angles=[0], levels=GLCM_LEVELS,
                        symmetric=True, normed=True)
    props = [graycoprops(glcm, p)[0, 0] for p in
             ("contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation")]
    p = glcm[:, :, 0, 0]
    entropy = -np.sum(p * np.log2(p + 1e-12))
    return np.array(props + [entropy])                      # -> 7


def extract_features(win_rgb):
    """win_rgb: HxWx3 uint8 (RGB).  Return np.float32 array of length 395."""
    win = cv2.resize(win_rgb, (FEAT_SIZE, FEAT_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(win, cv2.COLOR_RGB2GRAY)
    vec = np.concatenate([_hog(gray), _color(win), _lbp(gray), _tex(gray)])
    return vec.astype(np.float32)


if __name__ == "__main__":
    # ô "trống" giả (xám, mịn) vs ô "có xe" giả (nhiều màu, gồ ghề)
    rng = np.random.default_rng(0)
    empty = np.full((60, 60, 3), 120, dtype=np.uint8) + rng.integers(-8, 8, (60, 60, 3), dtype=np.int16).astype(np.uint8)
    car   = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)
    fe, fc = extract_features(empty), extract_features(car)
    print("số đặc trưng / n features :", len(fe), "| tên khớp:", len(FEATURE_NAMES) == len(fe))
    ci = [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("color_")]
    print("color std  — ô trống:", round(float(fe[ci].std()), 4), " ô có xe:", round(float(fc[ci].std()), 4))
    print("tex_contrast — ô trống:", round(float(fe[FEATURE_NAMES.index('tex_contrast')]), 2),
          " ô có xe:", round(float(fc[FEATURE_NAMES.index('tex_contrast')]), 2))
