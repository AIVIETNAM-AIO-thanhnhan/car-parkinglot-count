"""features.py — Trích 395 đặc trưng cho mỗi cửa sổ.  Chủ sở hữu: P3 (Pipeline Engineer).

395 = hog(324) + lbp(10) + color(54) + tex(7)

    hog_*   : Histogram of Oriented Gradients  -> cạnh / hình dạng (shape/edges)
    lbp_*   : Local Binary Pattern (uniform)   -> kết cấu vi mô (micro-texture)
    color_* : histogram màu HSV                -> màu (color)          << tín hiệu chính
    tex_*   : thống kê GLCM                     -> kết cấu (texture)

Ô CÓ XE và ô TRỐNG cùng hình chữ nhật -> khác nhau ở MÀU và KẾT CẤU, không phải hình dạng.

🔴 THỨ TỰ NHÓM LÀ HỢP ĐỒNG, KHÔNG ĐƯỢC ĐỔI: hog -> lbp -> color -> tex.
   Đây là thứ tự cột đã nằm trong features/shard_*.parquet (1,9 GB, trích ngày 28/08).
   Một phiên bản trước của file này dùng thứ tự hog -> color -> lbp -> tex: tổng vẫn đúng 395
   nên KHÔNG crash, chỉ âm thầm hoán vị hai khối color/lbp và làm mọi dự đoán thành vô nghĩa.
   `self_test()` đối chiếu trực tiếp với schema parquet để lỗi đó không tái phát.

   Vì lý do đó mọi chi tiết dưới đây cũng phải giữ nguyên bit-for-bit so với lúc trích:
   gray của HOG là float [0,1] (`rgb2gray`), gray của LBP/GLCM là uint8; histogram dùng
   `density=True`; GLCM theo thứ tự contrast, dissimilarity, homogeneity, energy, correlation,
   ASM. Muốn đổi bất cứ thứ gì ở đây thì phải trích lại toàn bộ 1,9 GB.
"""
import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

import config

# Cấu hình khớp đúng số chiều theo README § Quy ước (tiền tố bắt buộc).
_HOG_KW = dict(orientations=9, pixels_per_cell=(16, 16), cells_per_block=(1, 1), feature_vector=True)
# window/16 = 6 -> 6x6 cell * 9 orientation = 324
_LBP_P, _LBP_R = 8, 1  # uniform LBP, P+2 = 10 bin
_COLOR_BINS = 18        # 3 kênh HSV * 18 bin = 54
_GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")  # 6 + entropy = 7
_GLCM_LEVELS = 32       # 256 mức quá chậm (graycomatrix ~256x256); 32 mức đủ cho texture, nhanh ~8x

FEATURE_GROUPS = (("hog", 324), ("lbp", 10), ("color", 54), ("tex", 7))

# Cột KHÔNG phải feature trong parquet — dùng để đối chiếu schema ở self_test().
META_COLS = ("image_id", "lot", "split", "class", "label", "x_min", "y_min", "x_max", "y_max")


def _hog(gray):
    return hog(gray, **_HOG_KW)


def _lbp(gray_u8):
    codes = local_binary_pattern(gray_u8, P=_LBP_P, R=_LBP_R, method="uniform")
    hist, _ = np.histogram(codes, bins=np.arange(0, _LBP_P + 3), density=True)
    return hist  # 10 chiều


def _color(rgb):
    # cv2 nhanh hơn skimage.rgb2hsv ~80x (dùng vì opencv-python-headless đã là dependency bắt buộc)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)  # H:[0,180) S,V:[0,256) — uint8
    ranges = ((0, 180), (0, 256), (0, 256))
    hist = [np.histogram(hsv[..., c], bins=_COLOR_BINS, range=ranges[c], density=True)[0] for c in range(3)]
    return np.concatenate(hist)  # 54 chiều


def _texture(gray_u8):
    quantized = (gray_u8.astype(np.uint16) * _GLCM_LEVELS // 256).astype(np.uint8)
    glcm = graycomatrix(quantized, distances=[1], angles=[0], levels=_GLCM_LEVELS, symmetric=True, normed=True)
    props = [graycoprops(glcm, p)[0, 0] for p in _GLCM_PROPS]
    p = glcm[:, :, 0, 0]
    entropy = -np.sum(p[p > 0] * np.log2(p[p > 0]))
    return np.array(props + [entropy])  # 7 chiều


def feature_names():
    return [f"{prefix}_{i}" for prefix, n in FEATURE_GROUPS for i in range(n)]


FEATURE_NAMES = feature_names()
assert len(FEATURE_NAMES) == 395


def extract(crop_rgb):
    """crop_rgb: ảnh RGB đã resize về (WINDOW_SIZE, WINDOW_SIZE). Trả về vector 395 chiều.

    KHÔNG tự resize — người gọi chịu trách nhiệm (build_dataset.py và infer.py đều resize trước).
    Đưa vào ảnh cỡ khác sẽ ra số chiều HOG khác và assert ở cuối hàm sẽ bắt được.
    """
    gray_f = rgb2gray(crop_rgb)  # float64 [0,1], dùng cho HOG
    gray_u8 = (gray_f * 255).astype(np.uint8)  # dùng cho LBP/GLCM (cần nguyên/rời rạc)
    vec = np.concatenate([_hog(gray_f), _lbp(gray_u8), _color(crop_rgb), _texture(gray_u8)])
    assert vec.shape[0] == sum(n for _, n in FEATURE_GROUPS), (
        f"{vec.shape[0]} chiều thay vì 395 — crop_rgb phải là {config.WINDOW_SIZE}x{config.WINDOW_SIZE}, "
        f"nhận được {crop_rgb.shape}")
    return vec


# Tên cũ của API thế hệ sau (commit 9fc4384). Giữ làm alias để code đang gọi không gãy —
# nhưng CHỈ là alias: chỉ tồn tại một implementation duy nhất, khớp parquet.
extract_features = extract


def self_test():
    """Chốt chặn chống tái phát lỗi hoán vị khối color/lbp.

    So thẳng với schema parquet thật (chỉ đọc metadata, ~mili-giây, không nạp 240 MB).
    """
    assert feature_names() == FEATURE_NAMES
    assert len(FEATURE_NAMES) == 395, len(FEATURE_NAMES)
    assert FEATURE_NAMES[324] == "lbp_0", (
        f"khối sau hog phải là lbp, đang là {FEATURE_NAMES[324]} — thứ tự nhóm bị đổi")
    assert FEATURE_NAMES[334] == "color_0", (
        f"khối sau lbp phải là color, đang là {FEATURE_NAMES[334]} — thứ tự nhóm bị đổi")

    shards = sorted(config.FEAT.glob("shard_*.parquet")) if config.FEAT.exists() else []
    if shards:
        import pyarrow.parquet as pq
        names = pq.ParquetFile(shards[0]).schema.names
        cols = [c for c in names if c not in META_COLS]
        assert cols == FEATURE_NAMES, (
            f"THỨ TỰ FEATURE LỆCH VỚI {shards[0].name}.\n"
            f"  parquet[324:326] = {cols[324:326]}\n"
            f"  code   [324:326] = {FEATURE_NAMES[324:326]}\n"
            "Model train trên parquet sẽ nhận feature hoán vị -> dự đoán vô nghĩa mà không báo lỗi.")
        src = f"khớp schema {shards[0].name}"
    else:
        src = "KHÔNG có parquet để đối chiếu (chỉ kiểm được thứ tự nhóm)"

    n = config.WINDOW_SIZE
    vec = extract(np.zeros((n, n, 3), dtype=np.uint8))
    assert vec.shape == (395,), vec.shape
    assert np.isfinite(vec).all(), "extract() trả về NaN/inf trên ảnh đen"

    print(f"features.self_test PASS: 395 chiều, thứ tự hog->lbp->color->tex, {src}")
    return True


if __name__ == "__main__":
    self_test()

    # ô "trống" giả (xám, mịn) vs ô "có xe" giả (nhiều màu, gồ ghề)
    rng = np.random.default_rng(0)
    n = config.WINDOW_SIZE
    empty = np.full((n, n, 3), 120, dtype=np.uint8) + rng.integers(-8, 8, (n, n, 3), dtype=np.int16).astype(np.uint8)
    car = rng.integers(0, 255, (n, n, 3), dtype=np.uint8)
    fe, fc = extract(empty), extract(car)
    ci = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith("color_")]
    print("color std   — ô trống:", round(float(fe[ci].std()), 4), " ô có xe:", round(float(fc[ci].std()), 4))
    ti = FEATURE_NAMES.index("tex_0")  # contrast
    print("tex contrast— ô trống:", round(float(fe[ti]), 2), " ô có xe:", round(float(fc[ti]), 2))
