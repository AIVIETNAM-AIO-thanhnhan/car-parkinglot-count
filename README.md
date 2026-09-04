# Phát hiện và Đếm Chỗ Đỗ Xe

Phát hiện ô đỗ xe từ ảnh camera giám sát, phân loại **có xe / còn trống**, và đếm số chỗ trống — chỉ dùng học máy cổ điển, không dùng CNN.

Đồ án Module 3 · nhóm 4 người · 23/08 → 07/09/2026

---

## Bài toán

```
Ảnh bãi đỗ  →  danh sách ô đỗ  →  3 con số
```

| Output | Ví dụ |
|---|---|
| Box + nhãn + độ tin cậy | `(120,80,165,110, "có xe", 0.91)` × N ô |
| Số xe | 42 |
| **Số chỗ trống** | 18 |
| Tỉ lệ lấp đầy | 70% |

Ba con số cuối là thứ hiển thị trên bảng điện tử đầu bãi.

**Dataset:** [PKLot](https://web.inf.ufpr.br/vri/databases/parking-lot-database) — 12.417 ảnh 1280×720, 3 bãi đỗ, 3 điều kiện thời tiết, giấy phép CC BY 4.0.

---

## Quy trình đầy đủ

```
  raw/PKLot/*.jpg + *.xml                                   ~2 GB, tải 1 lần
        │
        │  [1] pklot_data.py          parse XML · cắt vùng có nhãn · lấy mẫu thời gian · chia split
        ▼
  processed/  splits.json · crops.json · gt.csv             579 ảnh · 34.686 ô
        │
        │  [2] build_dataset.py       trượt cửa sổ · gán nhãn IoU · lấy mẫu nền · trích 395 feature
        ▼
  features/shard_*.parquet                                  8 shard · 1,9 GB · 1,18 triệu cửa sổ
        │
        │  [3] baselines.py           SÀN phải vượt
        │  [4] train_model.py         Decision Tree → Random Forest → lưu bundle
        ▼
  models/rf.joblib  +  results.csv                          53 MB · bảng kết quả
        │
        │  [5] infer.py               1 ảnh → cửa sổ/ô → feature → model → box → ĐẾM
        ▼
  app/streamlit_app.py                                      số xe / số chỗ trống / % lấp đầy
```

Bước 1–2 chạy **một lần** rồi cache. Bước 3–4 chạy lại mỗi khi đổi model. Bước 5 chạy mỗi ảnh.

### Cài đặt

```bash
git clone https://github.com/AIVIETNAM-AIO-thanhnhan/car-parkinglot-count.git
cd car-parkinglot-count
pip install -r requirements.txt
```

> **Windows:** console mặc định dùng cp1252, không in được tiếng Việt và sẽ ném
> `UnicodeEncodeError` ở dòng `print` cuối. Đặt `PYTHONIOENCODING=utf-8` trước mọi lệnh
> `python` bên dưới (hoặc `$env:PYTHONIOENCODING='utf-8'` trong PowerShell).

### [1] Chuẩn bị dữ liệu — P2, một lần

```bash
bash scripts/download_data.sh raw          # ~2 GB, 15-40 phút, cần wget
cd src && python pklot_data.py             # -> processed/{splits,crops}.json + gt.csv
```

Bước này làm bốn việc, mỗi việc xử lý một đặc thù của PKLot (xem bảng §"Bốn đặc thù"):
parse XML lấy cả `rotatedRect`, cắt ảnh về vùng có nhãn, giữ 1 ảnh mỗi 2 giờ, và chia split
**theo bãi đỗ**.

`splits.json` được **KHÓA vĩnh viễn** (quy tắc #2). Lệnh trên tự động dùng lại nếu file đã có;
`--overwrite-split` chia lại nhưng làm mọi feature và model đã có trở nên lỗi thời.

Ba file `processed/*.json` đã có sẵn trong Git để tái lập được — chỉ `gt.csv` (4 MB, dẫn xuất)
là phải sinh lại.

### [2] Trích feature — cả nhóm, một lần

```bash
cd src
python build_dataset.py --shard 0 --n-shards 8    # mỗi người chạy 1-2 shard
```

Mỗi ảnh sinh ~18.000 cửa sổ ở 5 tỉ lệ; cửa sổ được gán nhãn theo IoU với ô thật
(`0` ô trống / `1` có xe / `2` nền / `-1` bỏ qua), giữ lại 10% cửa sổ nền, rồi trích 395 đặc trưng.

Kết quả: `features/shard_*.parquet`, 8 file, tổng 1,9 GB, 1,18 triệu cửa sổ.

> ⚠️ Thứ tự 395 cột là **hợp đồng**: `hog(324) → lbp(10) → color(54) → tex(7)`.
> `features.self_test()` đối chiếu trực tiếp với schema parquet. Một phiên bản trước đã hoán vị
> hai khối `color`/`lbp` — tổng vẫn đúng 395 nên **không crash, chỉ ra kết quả sai âm thầm**.

### [3] Cổng kiểm tra — chạy TRƯỚC khi train

```bash
cd src && python -c "import features,windows,detect,evaluate_pklot,baselines,infer; \
  [f() for f in (features.self_test, windows.self_test, detect.self_test, \
                 evaluate_pklot.self_test, baselines.self_test, infer.self_test)]"
```

Sáu cổng phải PASS hết. FAIL nghĩa là mọi số đo sau đó không đáng tin (quy tắc #1).

```bash
python baselines.py        # Baseline A — SÀN mà mọi model phải vượt
```

### [4] Train

```bash
cd src
python train_model.py --model dt                                  # Baseline B, ~8 phút
python train_model.py --model rf --sweep-threshold \
                      --save ../models/rf.joblib                  # model chính, ~6 phút
```

Mỗi lần chạy tự ghi một dòng vào `results.csv` (dùng `--no-log` để không ghi).
`--sweep-threshold` quét **2 chiều** `score_thr × nms_iou` trên val.

`--save` ghi ra một **bundle**, không phải mỗi classifier: bundle khoá kèm *thứ tự 395 cột* và
*bảng nhãn* `{0: ô trống, 1: có xe, 2: nền}`. `load_model()` từ chối model không khớp — đây đúng
là hai thứ đã từng sai âm thầm trong dự án
([chi tiết](report/model_pipeline_audit.md)).

> 🔒 Test set chỉ mở Ngày 12, một lần. Muốn chạy trên test phải truyền
> `--eval-split test --open-test-set-day-12` một cách có ý thức.

### [5] Giao diện

```bash
streamlit run app/streamlit_app.py
```

Tải lên ảnh bãi đỗ → **số xe / số chỗ trống / tỉ lệ lấp đầy**, kèm ảnh có vẽ box và nút tải
`predictions.csv` (đúng định dạng harness). Hai chế độ:

| Chế độ | Cần gì | Sai số (10 ảnh val) | Tốc độ |
|---|---|---|---|
| **Nhánh B** — phân loại từng ô | vị trí ô: XML PKLot / `gt.csv` / JSON tự dán | **0,30 ô** | ~0,2 s |
| **Detector** — trượt cửa sổ | không cần gì, ảnh nào cũng chạy | 18,1 ô | ~8 s |

> ⚠️ **Detector kém hơn hẳn, và mAP trong `results.csv` nói quá về nó.** Lúc trích feature, ô
> car/empty được cắt bằng `rotatedRect` **lấy từ nhãn thật** (`build_dataset._rotated_crop`), và
> phép đo mAP chạy trên chính những feature đó — model được cho sẵn thông tin góc xoay của nhãn.
> Khi trượt cửa sổ trên ảnh thật thì không có nhãn nên mọi cửa sổ đều vuông góc. Đo trên 84 ô đỗ
> thật, cùng một model: cắt xoay thẳng đúng **81,0%**, cắt vuông góc chỉ **3,6%** — 96% bị gọi là
> "nền". Nhánh B không dính vì nó dùng đúng `rotatedRect` từ layout.
> [Chi tiết](report/model_pipeline_audit.md).

Toàn bộ tính toán nằm ở `src/infer.py`; `app/streamlit_app.py` chỉ lo giao diện. Phép đếm dùng
đúng công thức của `evaluate_pklot.evaluate()` nên số trên UI không mâu thuẫn với bảng kết quả.

`src/infer.py` cũng dùng được trực tiếp:

```python
import infer
bundle = infer.load_model("models/rf.joblib")
img    = infer.load_image("bai_xe.jpg")
slots  = infer.slot_boxes_from_xml("bai_xe.xml")     # hoặc slot_boxes_from_json([...])
pred   = infer.classify_slots(img, slots, bundle)
print(infer.count_from_predictions(pred))
# {'cars': 17, 'empty': 11, 'total': 28, 'occupancy_pct': 60.71428571428572}
```

Ảnh dùng XML thì toạ độ ở **hệ ảnh gốc** — đưa thẳng ảnh vào, không cắt. Chỉ khi lấy layout từ
`gt.csv` (toạ độ ở hệ **đã cắt**) mới phải cắt trước bằng
`infer.crop_image(img, infer.crop_for_lot("UFPR04"))`; UI tự làm việc này.

---

## Phân vai

| | Vai trò | Sở hữu | File |
|---|---|---|---|
| **P1** | Tech Lead | Harness, config, quét ngưỡng, tích hợp, báo cáo | `evaluate*.py`, `config.py`, `baselines.py`, `infer.py` |
| **P2** | Data Engineer | Parse XML, cắt vùng, split, canh test set | `pklot_data.py`, `build_dataset.py` |
| **P3** | Pipeline Engineer | Cửa sổ trượt, feature, tốc độ, ablation | `windows.py`, `features.py` |
| **P4** | Model Engineer | DT → RF → mining → LightGBM | `train_model.py` |

> ⚠️ **P4 là điểm nghẽn** — gánh chuỗi model liên tục Ngày 5→12. P2 chuyển sang hỗ trợ P4 từ Ngày 6. P1 nên đọc hiểu code của P4 từ Ngày 7 để tiếp quản được nếu cần.

**Notebook:** mỗi người dùng file riêng (`p1_*`, `p2_*`...). Notebook lưu cả output nên hai người sửa cùng file sẽ conflict liên tục.

---

## Cấu trúc

```
car-parkinglot-count/        ← Git: chỉ code
├── src/
│   ├── config.py            🔒 hằng số dùng chung (P1)
│   ├── pklot_data.py        [1] parse XML, cắt vùng, split      + CLI
│   ├── windows.py           [2] cửa sổ trượt, gán nhãn IoU
│   ├── features.py          [2] 395 đặc trưng — thứ tự cột là hợp đồng
│   ├── build_dataset.py     [2] sinh shard parquet               + CLI
│   ├── baselines.py         [3] Baseline A — sàn                 + CLI
│   ├── evaluate.py          [3] IoU, average precision
│   ├── evaluate_pklot.py    [3] harness chấm điểm — nguồn của MỌI con số
│   ├── train_model.py       [4] DT / RF, lưu + nạp bundle        + CLI
│   ├── detect.py            [5] ngưỡng + NMS  (hậu xử lý, không phải model)
│   └── infer.py             [5] 1 ảnh → 3 con số  ← UI gọi hàm này
├── app/streamlit_app.py     [5] giao diện demo
├── notebooks/               mỗi người 1 file
├── scripts/download_data.sh
├── processed/*.json         🔒 commit vào Git để tái lập được
├── report/                  báo cáo + rà soát kỹ thuật
└── results.csv              🔑 bảng theo dõi của P1

KHÔNG vào Git (xem .gitignore) — local hoặc MyDrive/pklot_project/
├── raw/PKLot/               ⛔ KHÔNG BAO GIỜ SỬA
├── processed/gt.csv         dẫn xuất, sinh lại được ở bước [1]
├── features/*.parquet       cache 1,9 GB, sinh 1 lần ở bước [2]
└── models/*.joblib          bundle, sinh ở bước [4]
```

---

## Bốn quy tắc không được vi phạm

1. **Harness viết xong và PASS self-test trước khi ai train** (Ngày 2)
2. **Chia split theo bãi đỗ, khóa vĩnh viễn** — không bao giờ chia ngẫu nhiên
3. **Test set chỉ mở Ngày 12, một lần duy nhất**
4. **CODE FREEZE Ngày 11** — sau đó chỉ sửa bug và viết báo cáo

---

## Vì sao phải chia theo bãi đỗ

Camera PKLot **cố định**, chụp 5 phút/lần suốt 30 ngày. Vị trí ô đỗ giống hệt nhau trong mọi ảnh của cùng một bãi, và xe đỗ lại hàng giờ.

Nên một chương trình chỉ **ghi nhớ vị trí ô** — không nhìn một pixel nào của ảnh test — vẫn đạt điểm cao:

| Cách chia test | mAP của Baseline A |
|---|---|
| Ngẫu nhiên, cùng ngày | **0.76** |
| 10 ngày sau, cùng bãi | 0.48 |
| **Bãi đỗ khác** | **0.00** |

Chạy `baselines.run_all()` ở Ngày 2. Nếu điểm cao → split đang sai, phải chia lại.

> **val cố ý cùng bãi với train** (UFPR04, khác NGÀY) để có tập tinh chỉnh nhanh. Đổi lại val có
> **sàn 0.5176** chứ không phải 0 — mọi model phải vượt rõ con số đó. **test = PUCPR** mới là bãi
> khác hoàn toàn, và đó mới là phép đo thật.

---

## Model

| Ngày | Model | Học ở | Bắt buộc? |
|---|---|---|---|
| 5 | Decision Tree | W1 ✅ | ✅ |
| 6–7 | Random Forest | W2 ✅ | ✅ |
| 8–9 | + Hard negative mining | W2 ✅ | ✅ |
| 10 | LightGBM | W3 (28/08) | Nếu kịp |
| 11 | Optuna | W3 (29/08) | Nếu kịp |

### Kết quả đo được (val, 04/09)

| Model | mAP_macro | AP_xe | AP_trống | free_slots_MAE | occupancy_MAE_pp | |
|---|---:|---:|---:|---:|---:|---|
| **Baseline A (đoán vị trí) — SÀN** | **0.5176** | 0.3901 | 0.6451 | 7.17 | 25.64 | 🔴 phải vượt |
| Baseline B (Decision Tree) | 0.2769 | 0.2693 | 0.2845 | 38.76 | 19.30 | ❌ dưới sàn |
| **Random Forest** (200 cây) | **0.6782** | 0.7296 | 0.6269 | 6.76 | 2.58 | ✅ **+0.161** |

> 🔴 `NMS_IOU` đã đổi **0.45 → 0.20** ngày 04/09 — đây là việc "quét ngưỡng **+ NMS**" mà
> KE_HOACH §7 Ngày 6–9 giao P1 nhưng chưa từng làm (`sweep_threshold()` cũ chỉ quét `score_thr`).
> Cửa sổ **đồng tâm ở hai tỉ lệ liền nhau** có IoU đúng bằng (48/72)² = **0.444**, nằm ngay dưới
> 0.45 nên NMS không gộp được, mỗi ô đỗ sinh nhiều box trùng ở các tỉ lệ khác nhau.
> Riêng thay đổi này đưa RF từ **0.2743 → 0.6782** trong khi recall gần như không đổi
> (0.686 → 0.665) — tức 978 box bị loại toàn là **bản sao của cùng một ô**.
> Không làm mất giá trị parquet vì NMS là **hậu xử lý**, không phải hằng số trích feature.
> [Chi tiết](report/model_pipeline_audit.md).

Sửa NMS giúp RF rất nhiều nhưng gần như không giúp DT (0.1676 → 0.2769): box thừa của DT nằm rải
rác ở các vị trí khác nhau — false positive **thật**, NMS không gộp được; box thừa của RF thì
chồng lên nhau ở cùng một ô.

### Tiêu chí thành công

Ngưỡng khác nhau giữa val và test — val cùng bãi với train nên dễ hơn hẳn:

| Mức | val (sàn 0.5176) | test — Ngày 12 |
|---|---|---|
| ❌ Thất bại | không vượt sàn Baseline A | — |
| ⚠️ Tối thiểu | mAP > 0.60 | mAP > 0.35 |
| ✅ Tốt | mAP > 0.70 và sai số chỗ trống < 3 ô | mAP > 0.50 và < 3 ô |

Hiện tại: RF đạt **⚠️ Tối thiểu** trên val (0.6782). Chưa đạt "✅ Tốt" — cần mAP > 0.70 **và**
`free_slots_MAE < 3` (đang 6.76).

---

## Bốn đặc thù PKLot đã xử lý

| Vấn đề | Giải pháp | Hàm |
|---|---|---|
| PUCPR chỉ gán nhãn ~100/300 ô | Cắt về vùng có nhãn → nhãn thành đầy đủ | `annotated_region()` |
| Ảnh chụp 5 phút/lần, gần trùng nhau | Giữ 1 ảnh/2 giờ (156 → 7 ảnh/ngày) | `temporal_subsample()` |
| Camera chéo, ô gần to hơn ô xa | Đo tỉ lệ méo, bật multi-scale nếu > 2.0 | `perspective_report()` |
| Hai lớp cùng hình dạng | Tín hiệu ở **màu và kết cấu**, không phải hình dạng | — |

Thêm một đặc thù phát hiện sau: **camera UFPR04 bị xê dịch ít nhất 2 lần** (tới ~94 px), trong
khi UFPR05/PUCPR cố định tuyệt đối. `annotated_region()` vì thế lấy **hợp** bbox từ nhiều ảnh
trải đều theo thời gian, không dùng một ảnh đại diện.

---

## Quy ước

**File dự đoán:**
```csv
image_id,x_min,y_min,x_max,y_max,label,score
UFPR04_2012-09-11_15_16_58,120,80,165,110,1,0.91
```
`label`: **0 = ô trống, 1 = có xe, 2 = nền, -1 = không rõ**. Lớp nền không bao giờ được sinh box.
Tọa độ trong hệ ảnh **đã cắt**.

> ⚠️ Quy ước này do code thi hành (`detect.py`, `windows.py`, `train_model.py`). `bao_cao.md`
> dùng thứ tự ngược (`background=0, empty=1, car=2`) — **không khớp**, đừng dùng.

**Tên file:** `<model>__<split>__<tham_số>.csv` — hai gạch dưới ngăn cách.

**Tên feature:** tiền tố nhóm bắt buộc, đúng thứ tự này — `hog_` (324), `lbp_` (10), `color_` (54),
`tex_` (7). Tổng 395 chiều. SHAP gộp theo tiền tố.

---

## Nhịp làm việc

| | Tần suất |
|---|---|
| Cập nhật `results.csv` | Cuối mỗi ngày |
| Họp đứng 20 phút | Thứ 3 & Thứ 6 |
| Review chéo code | P1→P2→P3→P4→P1 |

Mọi số đưa vào báo cáo **phải sinh ra từ `evaluate_pklot.py`**. Số tự tính tay không được dùng.

---

## Gỡ lỗi thường gặp

**`UnicodeEncodeError` khi chạy `python` trên Windows** — console dùng cp1252, không in được
tiếng Việt. Đặt `PYTHONIOENCODING=utf-8`. Code vẫn chạy đúng, chỉ chết ở dòng `print`.

**`ModuleNotFoundError: cv2` / `skimage`** — `pip install opencv-python-headless scikit-image`.
Bản `headless` là cố ý (không cần GUI).

**`FileNotFoundError: Không thấy shard nào`** — chưa chạy bước [2], hoặc trên Colab chưa mount Drive.

**`load_model` báo "thứ tự feature của model khác features.py"** — model được train bằng phiên
bản `features.py` khác. Train lại; đừng bỏ qua, dự đoán sẽ vô nghĩa mà không báo lỗi.

**Notebook conflict khi `git pull`** — Kernel → Restart & Clear Output rồi commit lại.

**`splits.json` không lên Git** — `.gitignore` dùng `processed/*` chứ không phải `processed/`, vì Git không cho lấy lại file nếu cả thư mục cha bị loại trừ. Cũng lưu ý `.gitignore` **không hỗ trợ comment cuối dòng**.

**Colab ngắt session** — feature đã cache trên Drive nên chỉ mất shard đang chạy (~1 phút). Chạy lại `build_dataset.py` với đúng `--shard`.

**mAP = 0 mà không rõ vì sao** — kiểm tra tọa độ box có nằm trong hệ ảnh **đã cắt** không. Đây là lỗi phổ biến nhất.

**Ảnh cắt bị lệch ~45 dòng** — dùng `infer.crop_image()`, đừng cắt bằng lát numpy.
`crops.json["UFPR04"]` cao 721 px trong khi ảnh chỉ cao 720 px, và `PIL.Image.crop()` (thứ mà
`build_dataset.py` dùng) **cho phép vượt biên rồi đệm đen**.

---

## Tài liệu

| File | Nội dung |
|---|---|
| `KE_HOACH.md` | Kế hoạch 16 ngày, tiêu chí thành công, phân vai |
| `report/model_pipeline_audit.md` | 🔴 Rà soát pipeline 04/09 — các lỗi đã tìm và sửa, kèm cách kiểm chứng |
| `report/data_processing_report.md` | Chi tiết xử lý dữ liệu (P2) |
| `report/BAO_CAO.md` | Báo cáo cuối |
| `bao_cao.md` | Ghi chép của P4 về lệch phân phối val/test — ⚠️ bảng nhãn trong file này ngược với code |
