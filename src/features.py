"""features.py — Trích 395 feature: HOG, LBP, color, texture
Chủ sở hữu: P3
"""
import cv2
import numpy as np
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


def _hog(gray):
    return hog(gray, **_HOG_KW)


def _lbp(gray):
    codes = local_binary_pattern(gray, P=_LBP_P, R=_LBP_R, method="uniform")
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


FEATURE_GROUPS = (("hog", 324), ("lbp", 10), ("color", 54), ("tex", 7))


def feature_names():
    return [f"{prefix}_{i}" for prefix, n in FEATURE_GROUPS for i in range(n)]


def extract(crop_rgb):
    """crop_rgb: ảnh RGB đã resize về (WINDOW_SIZE, WINDOW_SIZE). Trả về vector 395 chiều."""
    gray_f = rgb2gray(crop_rgb)  # float64 [0,1], dùng cho HOG
    gray_u8 = (gray_f * 255).astype(np.uint8)  # dùng cho LBP/GLCM (cần nguyên/rời rạc)
    vec = np.concatenate([_hog(gray_f), _lbp(gray_u8), _color(crop_rgb), _texture(gray_u8)])
    assert vec.shape[0] == sum(n for _, n in FEATURE_GROUPS), vec.shape
    return vec
