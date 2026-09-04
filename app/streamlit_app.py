"""streamlit_app.py — UI đếm chỗ đỗ xe.  Chủ sở hữu: P1.

    ảnh bãi đỗ  ->  SỐ XE  /  SỐ CHỖ TRỐNG  /  TỈ LỆ LẤP ĐẦY   (KE_HOACH.md §1)

Chạy:  streamlit run app/streamlit_app.py

Toàn bộ phần tính toán nằm ở src/infer.py — file này chỉ lo giao diện. Nhờ vậy con số hiện trên
màn hình đi qua đúng công thức mà evaluate_pklot.evaluate() dùng để chấm điểm dự án.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config          # noqa: E402
import infer           # noqa: E402
import windows         # noqa: E402

st.set_page_config(page_title="Đếm chỗ đỗ xe", page_icon="🅿️", layout="wide")

MODE_SLOTS = "Nhánh B — phân loại từng ô (chính xác hơn)"
MODE_DETECT = "Detector — trượt cửa sổ (ảnh nào cũng chạy)"


@st.cache_resource(show_spinner="Đang nạp model…")
def get_model(path, mtime):
    """mtime nằm trong chữ ký để model được nạp lại khi file thay đổi."""
    return infer.load_model(path)


@st.cache_data(show_spinner=False)
def read_image(data):
    import io
    return infer.load_image(io.BytesIO(data))


def fmt_int(n):
    return f"{n:,}".replace(",", ".")


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.title("🅿️ Cấu hình")

models = infer.available_models()
if not models:
    st.sidebar.error(f"Không có model nào trong `{config.MODELS}`.")
    st.error(
        f"**Chưa có model.** Thư mục `{config.MODELS}` trống.\n\n"
        "Train và lưu một model trước:\n\n"
        "```\ncd src\npython train_model.py --model rf --save ../models/rf.joblib\n```")
    st.stop()

model_path = st.sidebar.selectbox("Model", models, format_func=lambda p: p.name)
try:
    bundle = get_model(str(model_path), model_path.stat().st_mtime)
except Exception as e:
    st.sidebar.error("Model không hợp lệ")
    st.error(f"**Không nạp được `{model_path.name}`**\n\n```\n{e}\n```")
    st.stop()

with st.sidebar.expander("Thông tin model", expanded=False):
    st.write({
        "train lúc": bundle.get("trained_at", "?"),
        "cửa sổ": bundle.get("window_size"),
        "scales": bundle.get("scales"),
        "lớp": sorted(int(c) for c in bundle["clf"].classes_),
        **{f"val {k}": v for k, v in (bundle.get("val_metrics") or {}).items()},
    })

mode = st.sidebar.radio("Chế độ", [MODE_SLOTS, MODE_DETECT],
                        help="Nhánh B cần biết trước vị trí ô. Detector tự tìm, không cần gì.")

if mode == MODE_DETECT:
    st.sidebar.warning(
        "Detector kém hơn Nhánh B rất nhiều. Model được train trên các ô đã **xoay thẳng** theo "
        "nhãn thật, còn cửa sổ trượt thì **vuông góc** — đo trên 84 ô: đúng 81% vs **3,6%**.",
        icon="⚠️")
    st.sidebar.subheader("Tham số detector")
    fast = st.sidebar.checkbox("Chế độ nhanh (1 tỉ lệ)", value=False,
                               help="Chỉ quét tỉ lệ 1.0 — nhanh hơn ~8 lần, bỏ sót ô quá to/nhỏ")
    scales = [1.0] if fast else list(bundle.get("scales") or config.SCALES)
    score_thr = st.sidebar.slider("Ngưỡng điểm", 0.05, 0.95, float(config.SCORE_THR), 0.05)
    nms_iou = st.sidebar.slider(
        "NMS IoU", 0.05, 0.9, float(config.NMS_IOU), 0.05,
        help="Gộp box chồng nhau cùng lớp. Đặt trên 0.44 sẽ KHÔNG gộp được hai cửa sổ đồng tâm "
             "ở hai tỉ lệ liền nhau — (48/72)² = 0.444 — nên mỗi ô sinh nhiều box trùng. "
             "Đó là lý do mặc định là 0.20 chứ không phải 0.45.")

    st.sidebar.markdown("**Xử lý lớp nền**")
    correct_prior = st.sidebar.checkbox(
        "Bù lệch prior nền", value=False,
        help=f"Lúc trích feature chỉ giữ {config.NEG_SAMPLE_RATE:.0%} cửa sổ nền, nên model học "
             "với ~80% nền trong khi ảnh thật có ~98%. Về lý thuyết cần bù — NHƯNG đo trên 10 ảnh "
             "val thì bù prior làm tệ hơn (0 box, MAE 18,8 ô vs 2,5 box, MAE 18,1 ô), vì lỗi chủ "
             "đạo của detector là bỏ sót chứ không phải báo thừa. Mặc định TẮT theo số đo.")
    bg_veto = st.sidebar.checkbox(
        "Phủ quyết lớp nền", value=True,
        help="Bỏ cửa sổ mà lớp nền thắng argmax. Chỉ có tác dụng khi ngưỡng < 0.5.")
    dedup = st.sidebar.checkbox(
        "Gộp box chồng nhau khác lớp", value=True,
        help="NMS chạy riêng từng lớp, nên một ô có thể vừa bị đếm là xe vừa là chỗ trống.")
else:
    scales, score_thr, nms_iou = None, None, None
    correct_prior = bg_veto = False
    dedup = st.sidebar.checkbox("Gộp box chồng nhau khác lớp", value=False,
                                help="Các ô đỗ vốn không chồng nhau — thường không cần.")

# ----------------------------------------------------------------------
# Đầu vào
# ----------------------------------------------------------------------
st.title("Đếm chỗ đỗ xe")
st.caption("Ảnh bãi đỗ → số xe, số chỗ trống, tỉ lệ lấp đầy")

col_img, col_layout = st.columns(2)
with col_img:
    up = st.file_uploader("Ảnh bãi đỗ", type=["jpg", "jpeg", "png"])

slots = None
layout_note = ""
auto_crop = None      # hộp cắt phải áp lên ảnh trước khi chạy (chỉ với layout từ gt.csv)
if mode == MODE_SLOTS:
    with col_layout:
        src = st.radio("Vị trí các ô lấy từ đâu?",
                       ["File XML PKLot", "Bãi PKLot có sẵn", "Dán toạ độ JSON"],
                       horizontal=False)
        if src == "File XML PKLot":
            xml = st.file_uploader("File .xml đi kèm ảnh", type=["xml"])
            if xml is not None:
                tmp = Path(st.session_state.setdefault("_tmpdir", "."))
                p = tmp / "_uploaded.xml"
                p.write_bytes(xml.getvalue())
                try:
                    slots = infer.slot_boxes_from_xml(p)
                    layout_note = f"{len(slots)} ô từ `{xml.name}` (toạ độ ảnh gốc)"
                except Exception as e:
                    st.error(f"Không đọc được XML: {e}")
        elif src == "Bãi PKLot có sẵn":
            try:
                gt = pd.read_csv(config.PROC / "gt.csv")
                ids = sorted(gt.image_id.unique())
                image_id = st.selectbox("Ảnh trong gt.csv", ids)
                slots = infer.slot_boxes_from_gt(image_id)
                lot = gt.loc[gt.image_id == image_id, "lot"].iloc[0]
                auto_crop = infer.crop_for_lot(lot)
                layout_note = (f"{len(slots)} ô của ảnh `{image_id}` (bãi {lot}). "
                               f"Toạ độ trong `gt.csv` ở hệ ĐÃ CẮT, nên ảnh sẽ được tự động cắt "
                               f"theo `crops.json[{lot}]` = {list(auto_crop)}.")
            except Exception as e:
                st.error(f"Không đọc được gt.csv: {e}")
        else:
            txt = st.text_area("JSON: `[[x_min,y_min,x_max,y_max], …]`", height=120,
                               placeholder='[[100,50,190,140], [200,50,290,140]]')
            if txt.strip():
                try:
                    slots = infer.slot_boxes_from_json(txt)
                    layout_note = f"{len(slots)} ô từ JSON dán tay"
                except Exception as e:
                    st.error(f"JSON không hợp lệ: {e}")

if up is None:
    st.info("Tải lên một ảnh bãi đỗ để bắt đầu.")
    st.stop()

image = read_image(up.getvalue())
orig_h, orig_w = image.shape[:2]
if auto_crop is not None:
    # PHẢI dùng infer.crop_image (ngữ nghĩa PIL: đệm đen khi hộp vượt biên). Cắt bằng lát numpy
    # sẽ cụt mất 45 dòng ở UFPR04 và lệch toàn bộ toạ độ ô — xem docstring crop_image().
    image = infer.crop_image(image, auto_crop)
h, w = image.shape[:2]
st.caption(f"Ảnh `{up.name}` — {orig_w}×{orig_h} px"
           + (f" → đã cắt còn {w}×{h}" if auto_crop is not None else ""))
if layout_note:
    st.caption(layout_note)

if mode == MODE_SLOTS and not slots:
    st.warning("Chế độ Nhánh B cần biết vị trí các ô. Chọn một nguồn layout ở trên.")
    st.stop()

if mode == MODE_DETECT:
    n_win = windows.count_windows(w, h, bundle.get("window_size"), bundle.get("stride"), scales)["total"]
    est = n_win * 3.4 / 1000 / max(1, os.cpu_count() or 1) * 2.5   # 3.4 ms/cửa sổ, đo trên máy này
    st.caption(f"Sẽ quét **{fmt_int(n_win)}** cửa sổ ở {len(scales)} tỉ lệ — ước tính ~{est:.0f}s")
    if n_win == 0:
        st.error(f"Ảnh {w}×{h} nhỏ hơn mọi cửa sổ ({bundle.get('window_size')}px). Dùng ảnh lớn hơn.")
        st.stop()

if not st.button("▶️ Chạy", type="primary", use_container_width=True):
    st.stop()

# ----------------------------------------------------------------------
# Chạy
# ----------------------------------------------------------------------
t0 = time.perf_counter()
try:
    if mode == MODE_SLOTS:
        with st.spinner(f"Phân loại {len(slots)} ô…"):
            pred = infer.classify_slots(image, slots, bundle)
    else:
        bar = st.progress(0.0, text="Đang trích đặc trưng…")
        pred = infer.detect_image(
            image, bundle, scales=scales, score_thr=score_thr, nms_iou=nms_iou,
            correct_prior=correct_prior, bg_veto=bg_veto,
            progress=lambda d, t: bar.progress(min(1.0, d / t), text=f"Cửa sổ {fmt_int(d)}/{fmt_int(t)}"))
        bar.empty()
except Exception as e:
    st.error(f"**Chạy thất bại**\n\n```\n{type(e).__name__}: {e}\n```")
    st.stop()

raw_n = len(pred)
if dedup:
    pred = infer.dedup_across_classes(pred, nms_iou)
counts = infer.count_from_predictions(pred)
elapsed = time.perf_counter() - t0

# ----------------------------------------------------------------------
# Kết quả
# ----------------------------------------------------------------------
st.divider()
c1, c2, c3, c4 = st.columns(4)
c1.metric("🚗 Số xe", fmt_int(counts["cars"]))
c2.metric("🅿️ Số chỗ trống", fmt_int(counts["empty"]))
c3.metric("📊 Tỉ lệ lấp đầy", f"{counts['occupancy_pct']:.1f}%")
c4.metric("Tổng ô nhận ra", fmt_int(counts["total"]), help=f"chạy trong {elapsed:.1f}s")

if dedup and raw_n != len(pred):
    st.caption(f"Đã gộp {raw_n - len(pred)} box chồng nhau khác lớp (từ {raw_n} xuống {len(pred)}).")

if mode == MODE_DETECT:
    if counts["total"] < 5:
        st.error(
            f"**Chỉ nhận ra {counts['total']} ô — detector gần như không thấy gì.** Đây là hạn chế "
            "đã biết, không phải lỗi cấu hình: đo trên 84 ô đỗ thật, model đoán đúng **81%** khi ô "
            "được cắt xoay thẳng (như lúc train) nhưng chỉ **3,6%** khi cắt vuông góc (như khi "
            "trượt cửa sổ) — 96% bị gọi là 'nền'. Hãy dùng **Nhánh B**.")
    elif counts["total"] > 400:
        st.warning(
            f"**{fmt_int(counts['total'])} box là quá nhiều** cho một bãi đỗ thật. Thử nâng ngưỡng "
            "điểm, bật 'Bù lệch prior nền', hoặc dùng Nhánh B.")

st.image(infer.draw_boxes(image, pred), use_container_width=True,
         caption="🟩 ô trống  🟥 có xe")

with st.expander(f"Chi tiết {fmt_int(len(pred))} box"):
    show = pred.copy()
    show["nhãn"] = show.label.map(infer.LABEL_TEXT)
    st.dataframe(show[["nhãn", "score", "x_min", "y_min", "x_max", "y_max"]],
                 use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ Tải predictions.csv", pred.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv", mime="text/csv",
    help="Đúng định dạng evaluate_pklot.CSV_COLS — nạp thẳng vào harness chấm điểm được.")
