"""mine_hard_negatives.py — Hard negative mining.  Chủ sở hữu: P4

KE_HOACH.md §7 Ngày 6-9 gọi đây là **trục quan trọng nhất**:

    train -> chạy trên ảnh TRAIN -> thu false positive -> thêm vào tập nền -> train lại. Lặp 3 vòng.

⚠️ ĐÀO Ở ĐÂU MỚI ĐÚNG?
   build_dataset.py chỉ giữ NEG_SAMPLE_RATE = 10% cửa sổ nền khi trích feature. 90% còn lại
   (~6 triệu cửa sổ trên 340 ảnh train) CHƯA BAO GIỜ được trích. Đó mới là kho để đào: mẫu model
   chưa từng thấy. Đào lại trong parquet chỉ là upweight mẫu cũ — khác hẳn, và yếu hơn nhiều.
   Script này loại trùng bằng khoá (image_id, x_min, y_min, x_max, y_max) đối chiếu với các dòng
   nền có sẵn, nên chắc chắn chỉ lấy cửa sổ mới.

📊 ĐÃ ĐO (val, RF preset optuna, 60 ảnh/vòng, 40% ứng viên — 05/09):

    vòng  n_train   n_mined  mAP_macro  AP_occ  AP_empty   MAE   tỉ lệ FP
       0   704,515        0     0.7787  0.8162    0.7413  3.31          —
       1   706,987    2,472     0.7868  0.8378    0.7358  3.14      0.69%
       2   709,170    2,183     0.7947  0.8423    0.7471  2.69      0.61%   <- ĐỈNH
       3   710,884    1,714     0.7886  0.8357    0.7415  2.97      0.48%

    Kế hoạch ghi "lặp 3 vòng" nhưng số liệu nói DỪNG Ở VÒNG 2 — vòng 3 thoái lui. Vì vậy
    --rounds mặc định là 2, không phải 3. Muốn tái lập bảng trên thì truyền --rounds 3.

    Tỉ lệ FP giảm đơn điệu qua các vòng = model thật sự học được cách từ chối nền, không phải
    nhiễu. Số box cũng giảm 884 -> 847 mà AP_empty không tụt: box bị loại là box THỪA.

    Vòng 2 đưa free_slots_MAE xuống 2.69 (< 3) nên RF+mining đạt đủ tiêu chí "✅ Tốt" §5
    (mAP > 0.70 VÀ sai số chỗ trống < 3 ô) — điều RF preset đơn thuần không làm được (3.31).

⚠️ MINING ĐẨY MODEL VỀ PHÍA ĐOÁN "NỀN" NHIỀU HƠN. Đó là mục đích của nó, và nó đúng khi lỗi
   là false positive. Nhưng trên bãi PUCPR (test) các model đang THỪA-đoán nền sẵn rồi — sinh
   12-48 box so với 99 ô thật. Ở đó mining nhiều khả năng làm NẶNG THÊM. Dùng --probe-image để
   kiểm chính điều này sau mỗi vòng thay vì đoán.

Dùng:
    python mine_hard_negatives.py                              # 2 vòng, RF preset
    python mine_hard_negatives.py --rounds 3                   # tái lập bảng trên
    python mine_hard_negatives.py --base-model ../models/rf_optuna.joblib   # bỏ qua train vòng 0
    python mine_hard_negatives.py --probe-image ../raw/PKLot/PKLot/PUCPR/Rainy/2012-09-21/2012-09-21_06_10_10.jpg
"""
import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import build_dataset
import config
import detect
import evaluate_pklot
import features
import infer
import pklot_data
import train_model as tm
import windows

FEAT_NAMES = features.feature_names()
META = ["image_id", "lot", "split", "class", "x_min", "y_min", "x_max", "y_max"]

# Bộ tham số RF dùng cho mọi vòng — GIỮ NGUYÊN giữa các vòng, nếu không thì không biết cải thiện
# đến từ mining hay từ đổi tham số. Đây là preset Optuna (xem train_model.OPTUNA_PRESETS).
RF_KW = dict(n_estimators=200, max_depth=26, min_samples_leaf=50,
             criterion="gini", min_samples_split=200, class_weight="balanced")


def seen_background_keys(train_df):
    """Khoá của các cửa sổ NỀN đã nằm trong parquet — để không đào lại chính chúng."""
    bg = train_df.loc[train_df["label"] == 2, ["image_id", "x_min", "y_min", "x_max", "y_max"]]
    bg = bg.astype({"x_min": int, "y_min": int, "x_max": int, "y_max": int})
    return set(map(tuple, bg.values))


def mine_round(clf, paths, seen, gt_full, crops, candidate_frac, fp_thr, rng, columns,
               n_jobs=None, verbose_every=15):
    """Chạy clf trên ảnh train, trả về DataFrame các cửa sổ nền bị đoán nhầm thành ô trống/có xe.

    Chỉ giữ cửa sổ có max(P(ô trống), P(có xe)) >= fp_thr — tức đúng những cái sẽ lọt qua ngưỡng
    của detect.build_predictions và trở thành box rác. Cửa sổ nền mà model đã gán đúng thì không
    có gì để học thêm, đào vào chỉ làm loãng tập train.
    """
    rows, n_cand, n_win = [], 0, 0
    classes = list(clf.classes_)
    col_empty, col_occ = classes.index(0), classes.index(1)

    for k, p in enumerate(paths, 1):
        image_id = Path(p).stem
        lot = pklot_data._lot_of(p)
        g = gt_full[gt_full.image_id == image_id]
        if g.empty:
            continue                      # ảnh thiếu nhãn (đã biết: 1 ảnh PUCPR)
        x0, y0, x1, y1 = crops[lot]
        crop = build_dataset._load_crop(p, (x0, y0, x1, y1))
        ch, cw = crop.shape[:2]

        wins = list(windows.slide_windows(cw, ch))
        n_win += len(wins)
        labeled = windows.label_windows(
            wins, list(zip(g.x_min, g.y_min, g.x_max, g.y_max)), list(g.label))

        cand = [wb for wb, (cls, _) in zip(wins, labeled)
                if cls == "background" and (image_id, *map(int, wb)) not in seen]
        if not cand:
            continue
        take = max(1, int(len(cand) * candidate_frac))
        if take < len(cand):
            cand = [cand[i] for i in rng.choice(len(cand), size=take, replace=False)]
        n_cand += len(cand)

        vecs = infer.extract_batch(crop, cand, n_jobs=n_jobs)
        proba = clf.predict_proba(np.asarray(vecs, dtype=np.float32))
        obj = np.maximum(proba[:, col_empty], proba[:, col_occ])
        for i in np.where(obj >= fp_thr)[0]:
            wb = cand[i]
            rows.append([image_id, lot, "train", "background", *map(int, wb)] + list(vecs[i]))
            seen.add((image_id, *map(int, wb)))   # không đào lại ở vòng sau

        if verbose_every and k % verbose_every == 0:
            print(f"      …{k}/{len(paths)} ảnh, {n_cand:,} ứng viên, {len(rows):,} FP", flush=True)

    if not rows:
        return pd.DataFrame(columns=list(columns)), n_cand, n_win
    # Dựng một lượt bằng concat thay vì gán thêm cột vào frame 403 cột đã tạo — cách kia làm
    # pandas phân mảnh block và bắn PerformanceWarning.
    mined = pd.concat([
        pd.DataFrame([r[:len(META)] for r in rows], columns=META),
        pd.DataFrame(np.asarray([r[len(META):] for r in rows], dtype=np.float32),
                     columns=FEAT_NAMES),
        pd.DataFrame({"label": np.full(len(rows), 2, dtype=np.int8)}),
    ], axis=1)
    return mined[list(columns)], n_cand, n_win


def probe(clf, image_path, gt_full):
    """Chạy model trên MỘT ảnh thật (cả 2 đường) và in kết quả — để thấy mining giúp hay hại
    ở đúng chỗ đang hỏng, thay vì chỉ nhìn mAP tổng hợp trên val."""
    image_id = Path(image_path).stem
    g = gt_full[gt_full.image_id == image_id].copy()
    if g.empty:
        print(f"    [probe] {image_id}: không có trong gt.csv — bỏ qua", flush=True)
        return {}
    image = infer.load_image(image_path)
    slots = infer.slot_boxes_from_gt(image_id, gt_csv=config.PROC / "gt.csv")
    g["image_id"] = "input"
    n_free_gt = int((g.label == 0).sum())

    bundle = {"clf": clf, "feature_cols": FEAT_NAMES, "label_names": dict(tm.LABEL_NAMES),
              "neg_sample_rate": config.NEG_SAMPLE_RATE}
    predB = infer.classify_slots(image, slots, bundle)
    mB = evaluate_pklot.evaluate(g, predB)
    predD = infer.detect_image(image, bundle, image_id="input")
    mD = evaluate_pklot.evaluate(g, predD)

    print(f"    [probe {image_id}] split={g_split(gt_full, image_id)}  ô trống thật={n_free_gt}",
          flush=True)
    print(f"      Nhánh B : đoán {int((predB.label==0).sum()):>3} ô trống | "
          f"mAP {mB['mAP_macro']:.4f} | MAE {mB['free_slots_MAE']:.1f}", flush=True)
    print(f"      Detector: {len(predD):>3} box ({int((predD.label==0).sum())} trống, "
          f"{int((predD.label==1).sum())} xe) | mAP {mD['mAP_macro']:.4f} | "
          f"MAE {mD['free_slots_MAE']:.1f}", flush=True)
    return {"probe_B_MAE": mB["free_slots_MAE"], "probe_B_mAP": mB["mAP_macro"],
            "probe_D_box": len(predD), "probe_D_MAE": mD["free_slots_MAE"]}


def g_split(gt_full, image_id):
    s = gt_full.loc[gt_full.image_id == image_id, "split"].unique()
    return s[0] if len(s) else "?"


def main():
    ap = argparse.ArgumentParser(description="Hard negative mining (KE_HOACH.md §7)")
    ap.add_argument("--rounds", type=int, default=2,
                    help="số vòng. Mặc định 2 — đã đo: vòng 3 thoái lui (xem docstring)")
    ap.add_argument("--images-per-round", type=int, default=60,
                    help="số ảnh train dùng mỗi vòng, không lặp giữa các vòng")
    ap.add_argument("--candidate-frac", type=float, default=0.40,
                    help="tỉ lệ cửa sổ nền-chưa-thấy đem ra chấm điểm. Pool đầy đủ ~6M nên "
                         "không thể lấy hết trong thời gian hợp lý")
    ap.add_argument("--fp-thr", type=float, default=None,
                    help="ngưỡng coi là false positive. Mặc định config.SCORE_THR (0.50) — "
                         "cùng ngưỡng detect.build_predictions dùng")
    ap.add_argument("--base-model", default=None,
                    help="bundle .joblib có sẵn để làm vòng 0 (bỏ qua bước train lại). "
                         "VD ../models/rf_optuna.joblib")
    ap.add_argument("--eval-split", default="val", choices=["val", "test"])
    ap.add_argument("--open-test-set-day-12", action="store_true")
    ap.add_argument("--probe-image", default=None,
                    help="ảnh thật để kiểm sau MỖI vòng (cả Nhánh B lẫn detector). Dùng để thấy "
                         "mining giúp hay hại trên bãi khác — mAP val không nói lên điều đó")
    ap.add_argument("--save-dir", default="../models", help="nơi lưu bundle mỗi vòng")
    ap.add_argument("--mined-dir", default="../mined",
                    help="nơi lưu parquet FP đào được, để vòng/lần sau dùng lại khỏi đào lại")
    ap.add_argument("--sample", type=float, default=None, help="lấy mẫu tập train cho chạy thử")
    ap.add_argument("--person", default="P4")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()

    fp_thr = config.SCORE_THR if a.fp_thr is None else a.fp_thr
    if a.eval_split == "test" and not a.open_test_set_day_12:
        raise SystemExit("TỪ CHỐI: --eval-split test cần --open-test-set-day-12 "
                         "(KE_HOACH.md §8 quy tắc 3).")

    features.self_test()
    detect.self_test()
    evaluate_pklot.self_test()

    rng = np.random.default_rng(config.RANDOM_SEED)
    crops = json.load(open(config.PROC / "crops.json"))
    gt_full = pd.read_csv(config.PROC / "gt.csv")
    split = pklot_data.load_split()
    train_paths = sorted(split["train"])
    rng.shuffle(train_paths)

    print(f"\n[setup] đọc feature (train + {a.eval_split})…", flush=True)
    df = tm.load_features(splits=("train", a.eval_split), sample=a.sample)
    cur = df[df["split"] == "train"].reset_index(drop=True)
    eval_df = df[df["split"] == a.eval_split].reset_index(drop=True)
    gt_eval = tm.load_gt(a.eval_split)
    seen = seen_background_keys(cur)
    print(f"  train {len(cur):,} | {a.eval_split} {len(eval_df):,} | "
          f"cửa sổ nền đã có {len(seen):,}", flush=True)

    save_dir = Path(a.save_dir); mined_dir = Path(a.mined_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    mined_dir.mkdir(parents=True, exist_ok=True)

    # ---------- vòng 0 ----------
    if a.base_model:
        print(f"\n[vòng 0] nạp bundle có sẵn {a.base_model}…", flush=True)
        clf = tm.load_model(a.base_model)["clf"]
        train_time = 0.0
    else:
        print(f"\n[vòng 0] train RF preset trên parquet gốc ({len(cur):,} dòng)…", flush=True)
        clf, train_time = tm.train_random_forest(*tm.split_xy(cur)[:2], **RF_KW)
        print(f"  xong sau {train_time:.0f}s", flush=True)

    hist = []

    def record(r, n_mined, train_time, fp_rate=None):
        m, pred, _, _ = tm.evaluate_detection(clf, eval_df, gt_eval)
        print(f"  vòng {r}: mAP_macro {m['mAP_macro']:.4f} | AP_occ {m['AP_occupied']:.4f} | "
              f"AP_empty {m['AP_empty']:.4f} | MAE {m['free_slots_MAE']:.2f} | "
              f"{len(pred)} box", flush=True)
        extra = probe(clf, a.probe_image, gt_full) if a.probe_image else {}
        path = tm.save_model(clf, FEAT_NAMES, save_dir / f"rf_mined_r{r}.joblib",
                             crop_mode="rotated",
                             extra={"round": r, "n_mined": n_mined,
                                    **{k: round(v, 4) for k, v in m.items()}})
        tm.load_model(path)
        print(f"    💾 {path.name} ({path.stat().st_size/1e6:.1f} MB)", flush=True)
        hist.append(dict(round=r, n_train=len(cur), n_mined=n_mined, fp_rate=fp_rate,
                         train_time=train_time, **m, **extra))
        if not a.no_log:
            note = (f"RF preset + hard negative mining vòng {r}; n_train={len(cur)} "
                    f"n_mined={n_mined}" + (f" fp_rate={fp_rate:.4%}" if fp_rate else "")
                    + f"; {a.images_per_round} ảnh/vòng, candidate_frac={a.candidate_frac}, "
                    f"fp_thr={fp_thr}")
            tm.log_result({
                "date": date.today().isoformat(), "person": a.person,
                "experiment": f"RF + hard negative mining (vòng {r}) "
                              f"[cắt rotated, nms={config.NMS_IOU}, phủ quyết nền BẬT]",
                "split": a.eval_split,
                "mAP_macro": round(m["mAP_macro"], 4),
                "AP_occupied": round(m["AP_occupied"], 4),
                "AP_empty": round(m["AP_empty"], 4),
                "free_slots_MAE": round(m["free_slots_MAE"], 2),
                "occupancy_MAE_pp": round(m["occupancy_MAE_pp"], 2),
                "train_time": f"{train_time:.0f}s", "notes": note,
            })

    record(0, 0, train_time)

    # ---------- các vòng mining ----------
    used = 0
    for r in range(1, a.rounds + 1):
        paths = train_paths[used:used + a.images_per_round]
        used += a.images_per_round
        if not paths:
            print("  hết ảnh train chưa dùng — dừng."); break

        print(f"\n[vòng {r}] đào FP trên {len(paths)} ảnh train…", flush=True)
        t0 = time.perf_counter()
        mined, n_cand, n_win = mine_round(clf, paths, seen, gt_full, crops,
                                          a.candidate_frac, fp_thr, rng, cur.columns)
        dt = time.perf_counter() - t0
        fp_rate = len(mined) / max(n_cand, 1)
        print(f"  {n_cand:,} ứng viên (từ {n_win:,} cửa sổ) -> {len(mined):,} FP "
              f"({fp_rate:.2%}) trong {dt:.0f}s", flush=True)
        if not len(mined):
            print("  không đào được FP nào — model đã sạch trên vùng này, dừng."); break

        mp = mined_dir / f"mined_r{r}.parquet"
        mined.to_parquet(mp, index=False)
        print(f"  💾 {mp.name} ({mp.stat().st_size/1e6:.1f} MB)", flush=True)

        cur = pd.concat([cur, mined], ignore_index=True)
        print(f"  train: {len(cur):,} dòng (nền {int((cur.label==2).sum()):,})", flush=True)
        t0 = time.perf_counter()
        clf, train_time = tm.train_random_forest(*tm.split_xy(cur)[:2], **RF_KW)
        print(f"  train lại {train_time:.0f}s", flush=True)
        record(r, len(mined), train_time, fp_rate)

    # ---------- tổng hợp ----------
    print("\n\n########## BIỂU ĐỒ mAP QUA TỪNG VÒNG "
          f"({a.eval_split}) ##########", flush=True)
    h = pd.DataFrame(hist)
    cols = ["round", "n_train", "n_mined", "fp_rate", "mAP_macro", "AP_occupied",
            "AP_empty", "free_slots_MAE"]
    cols += [c for c in h.columns if c.startswith("probe_")]
    print(h[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    base = h.mAP_macro.iloc[0]
    for _, r in h.iloc[1:].iterrows():
        print(f"  vòng {int(r['round'])}: {r['mAP_macro']-base:+.4f} mAP so với vòng 0")
    best = h.loc[h.mAP_macro.idxmax()]
    print(f"\n  ĐỈNH: vòng {int(best['round'])} — mAP {best['mAP_macro']:.4f}, "
          f"MAE {best['free_slots_MAE']:.2f}  ->  models/rf_mined_r{int(best['round'])}.joblib")
    out = mined_dir / "mining_history.csv"
    h.to_csv(out, index=False)
    print(f"  💾 {out}")


if __name__ == "__main__":
    main()
