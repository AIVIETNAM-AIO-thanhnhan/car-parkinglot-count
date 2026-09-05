"""train_model.py — DT -> RF -> mining -> LightGBM.  Chủ sở hữu: P4

Ngày 5 (KE_HOACH.md §7): Decision Tree, Baseline B. Kỳ vọng mAP còn THẤP — đó là đúng.
Ngày 6-9: Random Forest (--model rf), model chính.

⚠️ MODEL PHẢI ĐƯỢC LƯU. Trước ngày 04/09 không có model nào tồn tại trên đĩa: mỗi lần chạy đều
   train lại từ đầu, và không có gì để UI hay bất kỳ ai khác dùng lại. Dùng --save để ghi ra
   bundle .joblib; save_model() đóng gói kèm THỨ TỰ CỘT và BẢNG NHÃN, vì đó chính là hai thứ đã
   từng sai âm thầm trong dự án này (xem features.py và load_model()).

⚠️ CHỈ SỐ NÀO MỚI TÍNH?
    Chỉ số của dự án là **mAP_macro** từ evaluate_pklot — đo trên BOX cuối cùng, sau ngưỡng + NMS.
    Accuracy trên cửa sổ (clf.score) KHÔNG phải chỉ số dự án và KHÔNG được điền vào bảng §5:
    lớp "nền" áp đảo nên đoán bừa "tất cả là nền" đã cho accuracy rất cao mà không bắt được ô nào.
    Hàm window_report() dưới đây in luôn accuracy của model-đoán-bừa để so sánh cho rõ.

    Sàn phải vượt trên VAL: Baseline A = 0.5176 mAP_macro (KE_HOACH.md §4/§5).

⚠️ TEST SET: chỉ mở Ngày 12, đúng MỘT lần (KE_HOACH.md §8 quy tắc 3). Muốn chạy trên test phải
   truyền cờ --open-test-set-day-12 một cách có ý thức.

Dùng:
    python train_model.py                        # DT trên train, đánh giá trên val
    python train_model.py --sample 0.25          # lấy mẫu 25% dòng cho nhanh (thử nghiệm)
    python train_model.py --sweep-threshold      # + quét SCORE_THR trên val
    python train_model.py --no-log               # không ghi vào results.csv
    python train_model.py --model rf --save ../models/rf.joblib    # RF + lưu bundle cho UI
"""
import argparse
import csv
import time
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text

import config
import detect
import evaluate_pklot
import features

META_COLS = ["image_id", "lot", "split", "class", "label", "x_min", "y_min", "x_max", "y_max"]
LABEL_NAMES = {0: "ô trống", 1: "có xe", 2: "nền"}

# Shard parquet được sinh bởi build_dataset.py (cột 'class', chuỗi) hoặc build_features.py
# (cột 'label', số). Chấp nhận cả hai để không phụ thuộc phiên bản pipeline đã tạo ra file.
CLASS_TO_LABEL = {
    "empty": 0, "free": 0, "vacant": 0, "trong": 0,
    "car": 1, "occupied": 1, "vehicle": 1, "co_xe": 1,
    "background": 2, "bg": 2, "negative": 2, "neg": 2, "nen": 2,
}


def load_features(splits=("train", "val"), feat_dir=None, sample=None, seed=None):
    """Đọc mọi shard parquet, lọc theo split, chuẩn hoá nhãn về 0/1/2.

    sample: tỉ lệ dòng giữ lại (0-1) để chạy thử nhanh. Lấy mẫu PHÂN TẦNG theo ảnh+lớp để
    không làm lệch phân bố lớp — và chỉ áp cho TRAIN, val/test luôn giữ nguyên 100% dòng,
    nếu không thì con số đánh giá sẽ không so sánh được giữa các lần chạy.
    """
    feat_dir = Path(feat_dir) if feat_dir else config.FEAT
    shards = sorted(feat_dir.glob("shard_*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"Không thấy shard nào trong {feat_dir}. Trên Colab cần mount Drive trước; "
            "trên máy local cần copy thư mục features/ về (xem report/data_processing_report.md §6).")

    frames = []
    for p in shards:
        df = pd.read_parquet(p)
        if "split" not in df.columns:
            raise KeyError(f"{p.name} thiếu cột 'split' — không xác định được ảnh nào thuộc train/val.")
        frames.append(df[df["split"].isin(splits)])
    df = pd.concat(frames, ignore_index=True)

    if "label" not in df.columns:
        if "class" not in df.columns:
            raise KeyError("Parquet thiếu cả 'label' lẫn 'class' — không có nhãn để train.")
        raw = df["class"].astype(str).str.strip().str.lower()
        unknown = sorted(set(raw) - set(CLASS_TO_LABEL))
        if unknown:
            raise ValueError(f"Giá trị 'class' lạ: {unknown}. Bổ sung vào CLASS_TO_LABEL.")
        df["label"] = raw.map(CLASS_TO_LABEL).astype(np.int8)
    df["label"] = df["label"].astype(np.int8)

    bad = sorted(set(df["label"].unique()) - {0, 1, 2})
    if bad:
        raise ValueError(f"Nhãn ngoài {{0,1,2}}: {bad} — cửa sổ 'ignore' (-1) lẽ ra đã bị loại khi trích feature.")

    if sample:
        rng = np.random.default_rng(seed if seed is not None else config.RANDOM_SEED)
        is_train = df["split"] == "train"
        keep = ~is_train
        idx = df.index[is_train]
        strata = df.loc[is_train].groupby(["image_id", "label"], observed=True).indices
        chosen = []
        for pos in strata.values():
            pos = idx[pos]
            n = max(1, int(round(len(pos) * sample)))
            chosen.append(rng.choice(pos, size=n, replace=False))
        keep.loc[np.concatenate(chosen)] = True
        df = df[keep].reset_index(drop=True)

    return df


def split_xy(df, feature_cols=None):
    feature_cols = feature_cols or [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()
    return X, y, feature_cols


def window_report(y_true, y_pred, title="cửa sổ"):
    """In phân bố lớp + báo cáo theo lớp. Có chủ đích in cả accuracy của model đoán-bừa
    (luôn chọn lớp đông nhất) để thấy accuracy tuyệt đối nói lên rất ít ở bài toán này."""
    counts = pd.Series(y_true).value_counts().sort_index()
    share = counts / counts.sum()
    print(f"\n--- Mức {title}: phân bố lớp thật ---")
    for lab, n in counts.items():
        print(f"  {lab} ({LABEL_NAMES.get(lab, '?'):>7}): {n:>9,}  ({share[lab]:6.2%})")
    majority = float(share.max())
    acc = float((np.asarray(y_true) == np.asarray(y_pred)).mean())
    print(f"\n  accuracy của model         : {acc:.4f}")
    print(f"  accuracy nếu đoán bừa lớp đông nhất: {majority:.4f}   <-- mốc so sánh thật")
    if acc <= majority + 0.01:
        print("  ⚠️  Model KHÔNG hơn đoán bừa. Accuracy cao ở đây không có nghĩa gì.")
    print("\n" + classification_report(
        y_true, y_pred, digits=3,
        labels=[0, 1, 2], target_names=[LABEL_NAMES[i] for i in (0, 1, 2)], zero_division=0))
    print("Ma trận nhầm lẫn (hàng = thật, cột = đoán), thứ tự [ô trống, có xe, nền]:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))
    return {"window_accuracy": acc, "majority_accuracy": majority}


def train_decision_tree(X, y, max_depth=12, min_samples_leaf=50, seed=None):
    """Baseline B (KE_HOACH.md §4). class_weight='balanced' vì lớp nền áp đảo — không có nó,
    cây tối ưu accuracy bằng cách đoán 'nền' gần như mọi nơi và không bắt được ô nào.
    max_depth giới hạn để cây còn ĐỌC ĐƯỢC — đó là lý do dùng DT ở ngày 5, không phải vì điểm cao."""
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=seed if seed is not None else config.RANDOM_SEED,
    )
    t0 = time.perf_counter()
    clf.fit(X, y)
    return clf, time.perf_counter() - t0


def train_random_forest(X, y, n_estimators=300, max_depth=None, min_samples_leaf=5, seed=None):
    """Model chính (KE_HOACH.md §3, Ngày 6-9).

    class_weight='balanced_subsample' chứ không phải 'balanced': mỗi cây RF học trên một bootstrap
    khác nhau, cân trọng số theo chính mẫu bootstrap đó mới đúng — 'balanced' tính một lần trên
    toàn bộ y rồi áp cho mọi cây.

    max_depth=None cố ý: RF chống overfit bằng bagging chứ không bằng cắt sâu, và ta KHÔNG cần
    đọc luật ở đây (đó là việc của Decision Tree ở Baseline B).
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        n_jobs=config.N_JOBS,
        random_state=seed if seed is not None else config.RANDOM_SEED,
    )
    t0 = time.perf_counter()
    clf.fit(X, y)
    return clf, time.perf_counter() - t0


def save_model(clf, feature_cols, path, extra=None, crop_mode="rotated"):
    """Lưu một BUNDLE, không phải mỗi clf.

    Chỉ có clf là không đủ: một mảng 395 số không tự nói lên cột nào là cột nào, và một
    `classes_` = [0,1,2] không tự nói 0 nghĩa là gì. Hai thứ đó chính là hai lỗi đã xảy ra
    trong dự án này (hoán vị color/lbp, và bảng nhãn đảo trong bao_cao.md), nên cả hai phải
    được đóng gói cùng model và kiểm lại lúc nạp.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "clf": clf,
        "feature_cols": list(feature_cols),   # khoá thứ tự cột
        "label_names": dict(LABEL_NAMES),     # khoá ý nghĩa nhãn: 0=ô trống, 1=có xe, 2=nền
        "window_size": config.WINDOW_SIZE,
        "stride": config.STRIDE,
        "scales": list(config.SCALES),
        "score_thr": config.SCORE_THR,
        "nms_iou": config.NMS_IOU,
        "neg_sample_rate": config.NEG_SAMPLE_RATE,
        # "rotated": ô car/empty cắt theo rotatedRect từ nhãn -> CHỈ dùng được cho Nhánh B.
        # "axis"   : cắt bằng khung trượt vuông góc -> dùng được cho cả tự dò (infer.auto_layout).
        "crop_mode": crop_mode,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "val_metrics": extra or {},
    }, path, compress=3)
    return path


def load_model(path):
    """Nạp bundle và TỪ CHỐI mọi model không khớp hợp đồng nhãn/feature.

    Thà gãy ở đây còn hơn để detect.py âm thầm xuất vùng nền ra thành box 'ô trống'.
    """
    bundle = joblib.load(path)
    missing = {"clf", "feature_cols", "label_names"} - set(bundle)
    if missing:
        raise ValueError(f"{path}: bundle thiếu khoá {sorted(missing)} — file cũ hoặc không phải bundle.")

    n = len(bundle["feature_cols"])
    if n != 395:
        raise ValueError(f"{path}: bundle có {n} cột feature thay vì 395.")
    if bundle["feature_cols"] != features.FEATURE_NAMES:
        i = next(k for k, (a, b) in enumerate(zip(bundle["feature_cols"], features.FEATURE_NAMES)) if a != b)
        raise ValueError(
            f"{path}: thứ tự feature của model khác features.py tại cột {i} "
            f"({bundle['feature_cols'][i]} vs {features.FEATURE_NAMES[i]}). "
            "Model sẽ nhận feature hoán vị -> dự đoán vô nghĩa. Phải train lại.")

    if {int(k): v for k, v in bundle["label_names"].items()} != LABEL_NAMES:
        raise ValueError(
            f"{path}: model dùng bảng nhãn {bundle['label_names']} thay vì {LABEL_NAMES}. "
            "detect.py giả định 0=ô trống, 1=có xe, 2=nền — bảng khác sẽ làm ĐẢO LỚP: "
            "vùng nền bị xuất ra thành 'ô trống' và toàn bộ xe bị vứt.")

    classes = set(int(c) for c in bundle["clf"].classes_)
    if classes != {0, 1, 2}:
        raise ValueError(
            f"{path}: clf.classes_ = {sorted(classes)}, thiếu lớp {sorted({0,1,2} - classes)}. "
            "detect.build_predictions sẽ im lặng không sinh box cho lớp thiếu.")
    return bundle


def evaluate_detection(clf, df, gt, score_thr=None, nms_iou=None, bg_veto=True):
    """Đường đầy đủ: xác suất -> ngưỡng -> NMS -> mAP. Đây mới là chỉ số của dự án."""
    X, _, _ = split_xy(df)
    proba = clf.predict_proba(X)
    meta = df[["image_id", "x_min", "y_min", "x_max", "y_max"]]
    pred = detect.build_predictions(meta, proba, clf.classes_, score_thr, nms_iou, bg_veto=bg_veto)
    metrics = evaluate_pklot.evaluate(gt, pred)
    return metrics, pred, proba, meta


def load_gt(split_name, gt_path=None):
    gt = pd.read_csv(gt_path or (config.PROC / "gt.csv"))
    return gt[gt["split"] == split_name]


def log_result(row, path=None):
    """Ghi 1 dòng vào results.csv qua module csv (tự bọc dấu ngoặc kép — ghi tay bằng f-string
    sẽ làm lệch cột nếu ghi chú có dấu phẩy; lỗi này đã xảy ra một lần với các dòng Baseline A)."""
    path = Path(path) if path else Path(__file__).resolve().parent.parent / "results.csv"
    header = list(pd.read_csv(path, nrows=0).columns)
    with open(path, "a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=header).writerow({k: row.get(k, "") for k in header})
    return path


def main():
    ap = argparse.ArgumentParser(description="Decision Tree (Baseline B) / Random Forest (model chính)")
    ap.add_argument("--model", default="dt", choices=["dt", "rf"],
                    help="dt = Baseline B (đọc được luật) | rf = model chính")
    ap.add_argument("--eval-split", default="val", choices=["val", "test"])
    ap.add_argument("--open-test-set-day-12", action="store_true",
                    help="Bắt buộc để chạy trên test. KE_HOACH.md §8 quy tắc 3: test mở 1 lần, Ngày 12.")
    ap.add_argument("--max-depth", type=int, default=None,
                    help="mặc định: 12 cho dt (để đọc được luật), None cho rf")
    ap.add_argument("--min-samples-leaf", type=int, default=None, help="mặc định: 50 cho dt, 5 cho rf")
    ap.add_argument("--n-estimators", type=int, default=300, help="chỉ dùng cho rf")
    ap.add_argument("--sample", type=float, default=None, help="tỉ lệ dòng TRAIN giữ lại (0-1)")
    ap.add_argument("--feat-dir", default=None,
                    help="thư mục shard parquet. Mặc định config.FEAT (ô cắt xoay thẳng). "
                         "Dùng ../features_axis cho bộ cắt vuông góc (build_dataset --axis-aligned).")
    ap.add_argument("--crop-mode", default=None, choices=["rotated", "axis"],
                    help="ghi vào bundle để UI biết model có dùng được cho tự dò không. "
                         "Mặc định suy từ tên --feat-dir.")
    ap.add_argument("--score-thr", type=float, default=None)
    ap.add_argument("--nms-iou", type=float, default=None,
                    help="mặc định config.NMS_IOU. ĐỪNG đặt > 0.44: hai cửa sổ đồng tâm ở hai tỉ "
                         "lệ liền nhau có IoU (48/72)^2 = 0.444, đặt cao hơn thì NMS không gộp "
                         "được chúng và mỗi ô sinh nhiều box trùng (xem config.py).")
    ap.add_argument("--sweep-threshold", action="store_true")
    ap.add_argument("--no-bg-veto", action="store_true",
                    help="tắt phủ quyết lớp nền — dùng để tái lập các con số ghi trước 04/09")
    ap.add_argument("--print-rules", type=int, default=3, help="in luật của cây tới độ sâu này (0 = tắt)")
    ap.add_argument("--save", default=None, help="đường dẫn lưu bundle .joblib (VD ../models/rf.joblib)")
    ap.add_argument("--person", default="P4")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()

    # Mặc định theo từng loại model — đặt ở đây thay vì trong add_argument để biết người dùng
    # có truyền tay hay không.
    if a.max_depth is None:
        a.max_depth = 12 if a.model == "dt" else None
    if a.min_samples_leaf is None:
        a.min_samples_leaf = 50 if a.model == "dt" else 5
    bg_veto = not a.no_bg_veto

    if a.eval_split == "test" and not a.open_test_set_day_12:
        raise SystemExit(
            "TỪ CHỐI: đánh giá trên test cần cờ --open-test-set-day-12.\n"
            "KE_HOACH.md §8 quy tắc 3 — test set chỉ mở Ngày 12, đúng một lần. Hôm nay hãy dùng val.")

    features.self_test()      # bắt lỗi hoán vị khối color/lbp so với schema parquet
    detect.self_test()
    evaluate_pklot.self_test()

    crop_mode = a.crop_mode or ("axis" if a.feat_dir and "axis" in str(a.feat_dir) else "rotated")
    print(f"\n[1/4] Đọc feature (train + {a.eval_split}) từ {a.feat_dir or config.FEAT}"
          f"  [ô cắt {'VUÔNG GÓC' if crop_mode == 'axis' else 'xoay thẳng'}]…")
    df = load_features(splits=("train", a.eval_split), feat_dir=a.feat_dir, sample=a.sample)
    train_df = df[df["split"] == "train"]
    eval_df = df[df["split"] == a.eval_split]
    print(f"  train: {len(train_df):,} cửa sổ / {train_df.image_id.nunique()} ảnh"
          f" | {a.eval_split}: {len(eval_df):,} cửa sổ / {eval_df.image_id.nunique()} ảnh")

    X_tr, y_tr, feat_cols = split_xy(train_df)
    X_ev, y_ev, _ = split_xy(eval_df, feat_cols)
    print(f"  {len(feat_cols)} feature")

    if a.model == "rf":
        print(f"\n[2/4] Train Random Forest ({a.n_estimators} cây)…")
        clf, train_time = train_random_forest(
            X_tr, y_tr, a.n_estimators, a.max_depth, a.min_samples_leaf)
        depths = [t.get_depth() for t in clf.estimators_]
        print(f"  xong sau {train_time:.1f}s — độ sâu trung bình {np.mean(depths):.1f}, "
              f"sâu nhất {max(depths)}")
    else:
        print("\n[2/4] Train Decision Tree…")
        clf, train_time = train_decision_tree(X_tr, y_tr, a.max_depth, a.min_samples_leaf)
        print(f"  xong sau {train_time:.1f}s — độ sâu thật {clf.get_depth()}, {clf.get_n_leaves()} lá")

    if a.print_rules and a.model == "rf":
        # RF không có luật để đọc — nhưng độ quan trọng theo nhóm vẫn kiểm chứng được KE_HOACH §9
        by_group = pd.Series(clf.feature_importances_, index=feat_cols).groupby(
            lambda c: c.split("_")[0]).sum().sort_values(ascending=False)
        print("\n  Tổng độ quan trọng theo NHÓM feature (§9 dự đoán color_* > hog_*):")
        print(by_group.to_string(float_format=lambda v: f"{v:.4f}"))
    elif a.print_rules:
        print(f"\n  Luật của cây (cắt ở độ sâu {a.print_rules} cho dễ đọc):")
        print(export_text(clf, feature_names=feat_cols, max_depth=a.print_rules,
                          class_names=[LABEL_NAMES[c] for c in clf.classes_]))
        top = pd.Series(clf.feature_importances_, index=feat_cols).nlargest(15)
        print("  15 feature quan trọng nhất (đối chiếu KE_HOACH.md §9: color_* nên hơn hog_*):")
        print(top.to_string(float_format=lambda v: f"{v:.4f}"))
        by_group = pd.Series(clf.feature_importances_, index=feat_cols).groupby(
            lambda c: c.split("_")[0]).sum().sort_values(ascending=False)
        print("\n  Tổng độ quan trọng theo NHÓM feature:")
        print(by_group.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n[3/4] Mức cửa sổ (THAM KHẢO — không phải chỉ số dự án)")
    win_train = window_report(y_tr, clf.predict(X_tr), "cửa sổ / TRAIN")
    win_eval = window_report(y_ev, clf.predict(X_ev), f"cửa sổ / {a.eval_split.upper()}")

    print(f"\n[4/4] Mức BOX — chỉ số thật ({a.eval_split})…")
    gt = load_gt(a.eval_split)
    metrics, pred, proba, meta = evaluate_detection(
        clf, eval_df, gt, a.score_thr, a.nms_iou, bg_veto=bg_veto)
    print(f"  {len(pred):,} box sau ngưỡng {a.score_thr or config.SCORE_THR}"
          f" + NMS {a.nms_iou or config.NMS_IOU}"
          f" + phủ quyết nền {'BẬT' if bg_veto else 'TẮT'}")
    for k, v in metrics.items():
        print(f"  {k:>18}: {v:.4f}")

    # Đối chứng tác động của phủ quyết nền — dùng lại proba/meta, KHÔNG chạy predict_proba lần hai.
    other = evaluate_pklot.evaluate(gt, detect.build_predictions(
        meta, proba, clf.classes_, a.score_thr, a.nms_iou, bg_veto=not bg_veto))
    print(f"  (đối chứng, phủ quyết nền {'TẮT' if bg_veto else 'BẬT'}: "
          f"mAP_macro {other['mAP_macro']:.4f}, free_slots_MAE {other['free_slots_MAE']:.2f})")
    if abs(other["mAP_macro"] - metrics["mAP_macro"]) < 1e-9 and (a.score_thr or config.SCORE_THR) >= 0.5:
        print("   ↑ giống hệt là ĐÚNG: 3 lớp có tổng xác suất = 1, nên lớp nào đạt >= 0.5 thì tự động"
              "\n     là argmax — ở ngưỡng >= 0.5 phủ quyết nền không thêm gì. Nó chỉ có tác dụng khi"
              "\n     ngưỡng < 0.5 (xem bảng quét ngưỡng bên dưới).")

    if a.eval_split == "val":
        floor = 0.5176
        delta = metrics["mAP_macro"] - floor
        verdict = "✅ VƯỢT sàn" if delta > 0 else "❌ CHƯA vượt sàn"
        print(f"\n  Sàn Baseline A trên val = {floor:.4f} (KE_HOACH.md §4)")
        print(f"  {verdict} — chênh {delta:+.4f}")

    if a.sweep_threshold:
        if a.eval_split != "val":
            raise SystemExit("Chỉ được quét ngưỡng trên val — quét trên test là tinh chỉnh trên test.")
        print("\n[+] Quét SCORE_THR x NMS_IOU trên val (mAP_macro):")
        sweep = detect.sweep_threshold(meta, proba, clf.classes_, gt, bg_veto=bg_veto)
        print(sweep.pivot(index="score_thr", columns="nms_iou", values="mAP_macro").round(4))
        best = sweep.loc[sweep.mAP_macro.idxmax()]
        print(f"  tốt nhất: score_thr={best.score_thr} nms_iou={best.nms_iou} -> "
              f"mAP_macro {best.mAP_macro:.4f} ({int(best.n_pred):,} box)")
        cur = sweep[(sweep.score_thr == (a.score_thr or config.SCORE_THR))
                    & (sweep.nms_iou == (a.nms_iou or config.NMS_IOU))]
        if len(cur):
            print(f"  cấu hình hiện tại (config.py): mAP_macro {cur.iloc[0].mAP_macro:.4f}")

    if a.save:
        path = save_model(clf, feat_cols, a.save, crop_mode=crop_mode, extra={
            "split": a.eval_split, "bg_veto": bg_veto, **{k: round(v, 4) for k, v in metrics.items()}})
        load_model(path)   # nạp lại ngay để hợp đồng nhãn/feature được kiểm ngay lúc lưu
        print(f"\n💾 Đã lưu bundle {path} ({path.stat().st_size / 1e6:.1f} MB) — nạp lại OK")

    if not a.no_log:
        if a.model == "rf":
            cfg = (f"RF n_estimators={a.n_estimators} max_depth={a.max_depth} "
                   f"min_samples_leaf={a.min_samples_leaf} class_weight=balanced_subsample")
            exp = "Random Forest"
        else:
            cfg = (f"DT max_depth={a.max_depth} min_samples_leaf={a.min_samples_leaf} "
                   f"class_weight=balanced")
            exp = "Baseline B (Decision Tree)"
        # nms_iou PHẢI nằm trong nhãn: nó đổi mAP gấp 2,5 lần (0.2743 -> 0.6782 với RF), nên hai
        # dòng cùng model mà khác nms_iou trông sẽ mâu thuẫn nhau nếu không ghi rõ.
        exp += (f" [cắt {crop_mode}, nms={a.nms_iou or config.NMS_IOU}, "
                f"phủ quyết nền {'BẬT' if bg_veto else 'TẮT'}]")
        note = (f"{cfg} score_thr={a.score_thr or config.SCORE_THR} "
                f"nms_iou={a.nms_iou or config.NMS_IOU}; "
                f"phủ quyết nền {'BẬT' if bg_veto else 'TẮT'} (đối chứng phía kia: "
                f"mAP {other['mAP_macro']:.4f}); "
                f"window_acc={win_eval['window_accuracy']:.4f} vs đoán bừa "
                f"{win_eval['majority_accuracy']:.4f} (window_acc CHỈ tham khảo, không phải chỉ số §5); "
                f"train window_acc={win_train['window_accuracy']:.4f}"
                + (f"; sample={a.sample} dòng train" if a.sample else ""))
        path = log_result({
            "date": date.today().isoformat(), "person": a.person,
            "experiment": exp, "split": a.eval_split,
            "mAP_macro": round(metrics["mAP_macro"], 4),
            "AP_occupied": round(metrics["AP_occupied"], 4),
            "AP_empty": round(metrics["AP_empty"], 4),
            "free_slots_MAE": round(metrics["free_slots_MAE"], 2),
            "occupancy_MAE_pp": round(metrics["occupancy_MAE_pp"], 2),
            "train_time": f"{train_time:.0f}s", "notes": note,
        })
        print(f"\n✅ Đã ghi 1 dòng vào {path}")


if __name__ == "__main__":
    main()
