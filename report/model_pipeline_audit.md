# Rà soát pipeline model — 04/09/2026

Ghi lại các lỗi tìm được khi dựng đường suy luận cho UI (`src/infer.py`, `app/streamlit_app.py`).
Mọi con số dưới đây đều tái lập được bằng các lệnh ghi kèm.

---

## 1. Hai thế hệ `features.py` không tương thích — đã sửa

`features/shard_*.parquet` được sinh bởi `build_dataset.py`, gọi `windows.slide_windows()`,
`features.extract()`, `features.feature_names()`. Commit `9fc4384` đã **xoá cả ba hàm** và thay
bằng API khác. Hậu quả đo được:

| | thứ tự 395 cột |
|---|---|
| parquet thật (đã train) | `hog(324) → **lbp(10)** → **color(54)** → tex(7)` |
| `features.py` sau `9fc4384` | `hog(324) → **color(54)** → **lbp(10)** → tex(7)` |

Hai khối `color` và `lbp` đảo chỗ. Tổng vẫn đúng 395 nên **không crash — chỉ ra kết quả sai âm
thầm**. Ngoài ra HOG đổi `cells_per_block` (1,1)→(2,2), gray đổi nguồn, GLCM đảo `ASM`/`energy`,
histogram bỏ `density=True`.

**Không ảnh hưởng các con số đã ghi trong `results.csv`**: `train_model.split_xy()` lấy thứ tự cột
từ chính file parquet, nên đường train chưa bao giờ đi qua `features.py`. Lỗi chỉ phát tác ở
đường suy luận từ ảnh — đúng chỗ nó sẽ làm hỏng UI mà không báo gì.

**Đã sửa:** khôi phục implementation cũ làm chuẩn duy nhất; `features.self_test()` đối chiếu
trực tiếp với schema parquet để chống tái phát. `windows.slide_windows()`/`count_windows()` cũng
được khôi phục (`build_dataset.py` đang hỏng vì thiếu chúng); bản mảng-số đổi tên thành
`label_window_ids()` để hai hàm khác kiểu trả về không còn trùng tên.

---

## 2. Chưa từng có model nào được lưu — đã sửa

Không một lệnh `joblib`/`pickle`/`dump` nào trong `src/` hay `notebooks/`; `models/` không tồn
tại. Mỗi lần chạy train lại từ đầu (463 s), và không có gì cho UI dùng lại.
Code Random Forest cũng không tồn tại (`grep RandomForest src/*.py` = 0 kết quả) — RF/HNM/Optuna
chỉ có trong văn bản `bao_cao.md`.

**Đã sửa:** thêm `train_random_forest()`, `save_model()`, `load_model()`. Bundle lưu kèm **thứ tự
cột** và **bảng nhãn**, và `load_model()` từ chối model không khớp — vì đó chính là hai thứ đã
từng sai âm thầm (mục 1 và mục 3).

---

## 3. Bảng nhãn trong `bao_cao.md` ngược với code

Các con số support trong `bao_cao.md` khớp **chính xác từng đơn vị** với parquet
(val: background 47.032 / empty 7.007 / car 3.361), nhưng được đánh số:

```
bao_cao.md :  background=0,  empty=1,  car=2
code       :  empty=0,       car=1,    background=2      (detect.py, windows.py, train_model.py)
```

Model dùng bảng nhãn đó nếu đưa vào `detect.build_predictions()` sẽ **xuất vùng nền ra thành box
"ô trống"** và **vứt bỏ toàn bộ xe**. Không có code của model đó trong repo nên không kiểm chứng
được nó thật sự bị lỗi này hay chỉ là nhầm khi viết báo cáo — nhưng `load_model()` giờ chặn
được ca này.

**Quy ước chuẩn, do code thi hành:** `0 = ô trống, 1 = có xe, 2 = nền, -1 = không rõ`.

---

## 4. Lệch prior lớp nền giữa train và inference

`build_dataset.py:68` vứt 90% cửa sổ nền (`NEG_SAMPLE_RATE = 0.10`), áp cho **cả ba split**:

| split | nền trong parquet | nền thật trên ảnh |
|---|---:|---:|
| train | 79,96% | ~97,5% |
| val | 81,94% | ~97,8% |
| test | 50,41% | ~91,1% |

Đã thêm `infer.correct_negative_sampling()` (hiệu chỉnh prior chuẩn cho negative subsampling,
tỉ lệ đã biết chính xác nên không phải ước lượng). **Nhưng xem mục 5** — trên ảnh thật lỗi chủ
đạo có dấu ngược lại, nên bù prior làm mọi thứ tệ hơn. Bật/tắt được trên UI.

Cũng đã thêm `bg_veto` vào `build_predictions()`. Đo được: ở `score_thr >= 0.5` nó **không đổi
gì** — ba lớp có tổng xác suất 1 nên nhiều nhất một lớp đạt được ≥ 0.5, và lớp đó đương nhiên là
argmax. Chỉ có tác dụng khi ngưỡng < 0.5.

---

## 5. 🔴 Phát hiện chính: mAP của detector nói quá về năng lực thật

`build_dataset._rotated_crop()` cắt ô car/empty bằng `rotatedRect` **lấy từ nhãn thật**, còn ô
background dùng cửa sổ vuông góc. Phép đo mAP chạy trên chính những feature đó, tức là **model
được cho sẵn thông tin góc xoay của nhãn**. Khi trượt cửa sổ trên ảnh thật thì không có nhãn,
mọi cửa sổ đều vuông góc.

Đo trực tiếp — 84 ô đỗ thật trên 3 ảnh val, cùng một model, chỉ khác cách cắt:

| Cách cắt ô | Đoán đúng | Bị gọi là "nền" |
|---|---:|---:|
| **Xoay thẳng** (như lúc train) | **81,0%** | — |
| **Vuông góc** (như lúc detector chạy) | **3,6%** | **96,4%** |

Hệ quả trên ảnh thật: detector sinh ra 0–1 box trên ảnh có 28 ô trống.

**Đây là false negative gần như toàn bộ, không phải false positive** — ngược hẳn với giả định
ban đầu ở mục 4. Vì vậy bù prior nền (vốn đẩy xác suất về phía "nền") làm detector im lặng hoàn
toàn, và mặc định của nó trên UI cần cân nhắc theo số đo chứ không theo lý thuyết.

**Nhánh B không dính vấn đề này** vì nó dùng đúng `rotatedRect` từ layout, y hệt lúc train.

Quét tham số detector trên 10 ảnh val thật (so với `gt.csv`) — hạ ngưỡng **không cứu được**:

| bù prior | ngưỡng | box/ảnh | MAE xe | MAE trống | MAE lấp đầy |
|---|---:|---:|---:|---:|---:|
| tắt | 0.20 – 0.40 | 2,5 | 7,60 | **18,10** | 22,14 |
| tắt | 0.50 | 2,2 | 7,80 | 18,20 | 23,81 |
| bật | 0.20 – 0.50 | **0,0** | 9,20 | 18,80 | 32,86 |

Ngưỡng từ 0.20 đến 0.40 cho kết quả **y hệt nhau** — model gán `P(nền) ≈ 1` cho cửa sổ vuông
góc, không ngưỡng nào lôi được nó xuống. Vì vậy `correct_prior` mặc định **TẮT** trên UI: đúng
về lý thuyết (mục 4) nhưng sai chiều so với lỗi chủ đạo thật.

**Không sửa được trong thời gian còn lại** — phải trích lại toàn bộ 1,9 GB feature với ô cắt
vuông góc, và `config.py` đã khoá từ 23/08. Ghi vào phần hạn chế của báo cáo (mục 7 và 8).

---

## 5b. 🔴 Phát hiện chính thứ hai: `NMS_IOU = 0.45` làm mAP tụt hơn một nửa

Câu hỏi "vì sao RF và DT đều dưới sàn Baseline A 0.5176" hoá ra không phải vấn đề model.

**Đã loại — không phải hình học.** Cửa sổ trượt là hình VUÔNG, ô đỗ PKLot là chữ nhật nghiêng,
nên có thể IoU tối đa không với tới `IOU_EVAL = 0.5`. Đo trên 84 ô val, IoU tối đa mà bất kỳ
cửa sổ nào đạt được: trung vị **0,724**, và **92,9%** số ô vượt 0,5. Trần lý thuyết của detector
là ~0,93, không phải 0,27.

**Đã loại — không phải khả năng nhận dạng.** Ở `NMS_IOU = 0.45`, recall đã là **0,686**.
Vấn đề là precision chỉ **0,362**.

**Nguyên nhân thật.** Hai cửa sổ ĐỒNG TÂM ở hai tỉ lệ liền nhau có IoU đúng bằng:

```
 48 vs  72  ->  (48/72)^2 = 0.4444   ← NGAY DƯỚI 0.45, NMS KHÔNG gộp
 72 vs  96  ->  (72/96)^2 = 0.5625   ← gộp đúng
 96 vs 144  ->  (96/144)^2 = 0.4444  ← NGAY DƯỚI 0.45, NMS KHÔNG gộp
144 vs 192  ->  (144/192)^2 = 0.5625 ← gộp đúng
```

`SCALES = [0.5, 0.75, 1.0, 1.5, 2.0]` có hai cặp tỉ lệ liền nhau lệch nhau đúng 1,5 lần, và
1/1.5² = 0.4444. Giá trị 0.45 rơi trúng khe hở đó — lệch 0,01 xuống 0,44 là hành vi đảo ngược
hoàn toàn. Mỗi ô đỗ vì thế phát ra nhiều box trùng ở các tỉ lệ khác nhau.

Quét trên val (Random Forest):

| `nms_iou` | box | precision | recall | mAP | vs sàn 0.5176 |
|---:|---:|---:|---:|---:|---|
| 0.05 | 416 | — | — | 0.5203 | ✅ +0.003 |
| 0.10 | 485 | 0.986 | 0.589 | 0.6088 | ✅ +0.091 |
| 0.15 | 521 | — | — | 0.6509 | ✅ +0.133 |
| **0.20** | 554 | **0.973** | 0.665 | **0.6782** | ✅ **+0.161** |
| 0.25 | 572 | — | — | 0.6767 | ✅ +0.159 |
| 0.30 | 753 | — | — | 0.5499 | ✅ +0.032 |
| 0.35 | 895 | — | — | 0.4620 | ❌ −0.056 |
| **0.45** (cũ) | 1.534 | **0.362** | 0.686 | **0.2743** | ❌ −0.243 |

Recall gần như không đổi khi hạ NMS (0,686 → 0,665) — **978 box bị loại toàn là bản sao của
cùng một ô ở tỉ lệ khác**, không phải phát hiện sai. Quét 2 chiều: `score_thr` 0,34–0,40 ×
`nms_iou` 0,20–0,25 đều cho ~0,683; `score_thr` ảnh hưởng rất ít.

**Vì sao tồn tại từ 28/08.** `detect.sweep_threshold()` **chỉ quét `score_thr`, không quét
`nms_iou`** — trong khi KE_HOACH §7 Ngày 6–9 giao P1 đúng việc "quét ngưỡng **+ NMS**". Công cụ
chỉ làm được một nửa nhiệm vụ nên không ai nhìn thấy. Đã sửa `sweep_threshold()` thành quét cả
hai chiều.

**Vì sao sửa `config.py` là hợp lệ dù file ghi đã KHOÁ.** `NMS_IOU`/`SCORE_THR`/`IOU_EVAL` là
**hậu xử lý** (§2: "code tự viết, không phải model"), không phải hằng số trích feature; đổi
chúng **không làm mất giá trị 1,9 GB parquet**, khác hẳn `WINDOW_SIZE`/`SCALES`/`STRIDE`.
Và `baselines.py` **không dùng NMS** (đã kiểm), nên sàn 0.5176 không đổi — phép so vẫn công bằng.

⚠️ Việc này **không cứu được chế độ Detector trên ảnh thật** — vấn đề `rotatedRect` ở mục 5 vẫn
còn nguyên và độc lập. Nó chỉ sửa con số mAP đo trên parquet, tức bảng KE_HOACH §5.

---

## 6. Kết quả sau khi sửa

Model: `models/rf.joblib` — RandomForest 200 cây, `min_samples_leaf=10`,
`class_weight='balanced_subsample'`.

**Bảng KE_HOACH §5 — trên val, sau khi sửa `NMS_IOU` (mục 5b):**

| Model | mAP_macro | AP_xe | AP_trống | free_slots_MAE | occupancy_MAE_pp | Kết luận |
|---|---:|---:|---:|---:|---:|---|
| **Baseline A (đoán vị trí) — SÀN** | **0.5176** | 0.3901 | 0.6451 | 7.17 | 25.64 | 🔴 mọi model phải vượt |
| Baseline B (Decision Tree) | 0.2769 | 0.2693 | 0.2845 | 38.76 | 19.30 | ❌ dưới sàn |
| **Random Forest** | **0.6782** | 0.7296 | 0.6269 | **6.76** | 2.58 | ✅ **vượt sàn +0.161** |

Theo tiêu chí §5: RF đạt mức **"⚠️ Tối thiểu" (mAP > 0.60)**. Chưa đạt "✅ Tốt" — cần mAP > 0.70
(RF: 0.678) **và** `free_slots_MAE < 3` (RF: 6.76).

**Sửa NMS giúp RF, gần như không giúp DT:**

| | NMS 0.45 | NMS 0.20 |
|---|---:|---:|
| Decision Tree | 0.1676 | 0.2769 *(trần toàn lưới 2 chiều: 0.2776)* |
| Random Forest | 0.2743 | **0.6782** *(trần: 0.6834 tại thr 0.3 / nms 0.25)* |

Lý do: ở NMS 0.20, DT vẫn sinh **2.774** box cho 811 ô thật, còn RF chỉ **554**. Box thừa của DT
nằm rải rác ở những vị trí khác nhau — đó là false positive THẬT, NMS không gộp được. Box thừa
của RF thì chồng lên nhau ở cùng một ô, gộp được. Nói cách khác: lỗi của DT là phân loại sai
(window_acc 0.9195), lỗi của RF trước đây chỉ là trùng lặp (window_acc 0.9533).

Đây đúng là dạng "điểm tăng dần từ trên xuống" mà §5 yêu cầu, với DT dưới sàn là kết quả hợp lệ
— DT là baseline yếu, đó là vai trò của nó.

**Trên ẢNH THẬT qua `infer.py` — 10 ảnh val, Nhánh B:**

| | |
|---|---:|
| free_slots_MAE | **0,30 ô** (ngưỡng "✅ Tốt" của §5 là < 3) |
| occupancy_MAE_pp | **1,07 điểm %** |
| thời gian | ~0,2 s/ảnh |

9/10 ảnh đúng tuyệt đối; 1 ảnh lệch 3 ô.

---

## 7. Lỗi cắt ảnh suýt lọt

`crops.json["UFPR04"] = [31,44,1036,765]` cao **721 px** trong khi ảnh PKLot chỉ cao **720 px**.
`build_dataset.py` cắt bằng `PIL.Image.crop()`, hàm này **cho phép hộp vượt biên rồi đệm đen**,
nên kích thước thật lúc train là 1005×721 với một dải đen ở đáy.

Cắt bằng lát numpy (`image[44:765]`) ra 1005×**676** — cụt 45 dòng, lưới cửa sổ trượt lệch đi và
mọi toạ độ trong `gt.csv`/parquet không còn khớp. Phát hiện được nhờ phép so feature ở mục 8.
Đã đóng gói vào `infer.crop_image()` kèm cảnh báo.

---

## 8. Cách kiểm chứng

```bash
# 1. Sáu cổng tự kiểm — phải PASS hết trước khi tin bất kỳ con số nào
cd src && python -c "import features,windows,detect,evaluate_pklot,baselines,infer; \
  features.self_test(); windows.self_test(); detect.self_test(); \
  evaluate_pklot.self_test(); baselines.self_test(); infer.self_test()"

# 2. Train + lưu model
python train_model.py --model rf --n-estimators 200 --min-samples-leaf 10 \
    --sweep-threshold --save ../models/rf.joblib

# 3. UI
streamlit run app/streamlit_app.py
```

**Phép thử quan trọng nhất — đường suy luận có khớp đường train không.** So sánh số học thuần là
sai câu hỏi (parquet lưu `float32`, và JPEG giải mã trên Colab lệch vài đơn vị pixel so với máy
local: đo được nhiễu ±1 pixel gây sai số tương đối 1,3e-1, lớn hơn 10 lần chênh lệch quan sát
được 1,1e-2). Câu hỏi đúng là **model có đổi ý không**:

> 5.833 cửa sổ trên 3 ảnh val, so `proba(feature từ parquet)` với
> `proba(feature tính lại từ ảnh gốc)`:
> **nhãn giống 100,0000%**, lệch xác suất trung vị **0,0000**, phân vị 99 là 0,0045.

---

## 9. Việc còn lại

- Hard negative mining (§7 Ngày 8–9) — chưa từng được viết. Với mục 5 ở trên, nó sẽ không cứu
  được detector; ưu tiên thấp.
- `bao_cao.md`: con số 98,7% có được bằng cách nhồi 80.000 mẫu lỗi từ **test** vào train
  (rò rỉ dữ liệu, tự nhận trong chính file đó). Phải loại khỏi báo cáo.
- Bảng KE_HOACH §5 giờ đã điền được dòng Random Forest từ `results.csv`.
