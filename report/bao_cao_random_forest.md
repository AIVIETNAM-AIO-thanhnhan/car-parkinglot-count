# BÁO CÁO TỐI ƯU HÓA OPTUNA VÀ ĐÁNH GIÁ MÔ HÌNH: RANDOM FOREST

> **⚠️ ĐÃ ĐÁNH SỐ LẠI NHÃN (05/09, lúc merge main → feature/nhan-ui).**
> Bản gốc dùng `0=background, 1=empty, 2=car`. Toàn bộ dự án (`windows.py`, `detect.py`,
> `train_model.py`, `infer.py`, UI) dùng **`0 = ô trống (empty)`, `1 = có xe (car)`,
> `2 = nền (background)`**, nên báo cáo này đã được đánh số lại theo quy ước đó.
> **Các chỉ số không đổi** — chúng gắn với TÊN lớp, không gắn với chỉ số. Chỉ có số thứ tự
> và thứ tự hàng/cột của ma trận nhầm lẫn được hoán vị.
>
> Hệ quả cần nhớ: **file `.pkl` sinh ra từ lần chạy này (link ở `models/models.md`) vẫn theo
> quy ước CŨ** và sẽ bị `train_model.load_model()` từ chối. Muốn dùng cho UI phải train lại.

## 1. Cấu trúc Dữ liệu và Thiết lập Mục tiêu
Dự án sử dụng Optuna để tự động tìm kiếm bộ tham số tối ưu cho thuật toán Random Forest (Rừng ngẫu nhiên) nhằm phân loại 3 lớp đặc trưng HOG: `0` (ô trống / empty), `1` (có xe / car) và `2` (nền / background).

* **Tập Train:** 69.541 (0) | 71.609 (1) | 563.365 (2) — *Tổng: 704.515 mẫu*
* **Tập Validation:** 7.007 (0) | 3.361 (1) | 47.032 (2) — *Tổng: 57.400 mẫu*
* **Tập Test:** 114.116 (0) | 93.323 (1) | 210.846 (2) — *Tổng: 418.285 mẫu*

> **Nguồn dữ liệu đã xác minh (05/09):** sweep chạy trên `features/shard_*.parquet` do
> `build_dataset.py` sinh — 403 cột = 8 cột meta + 395 feature. Cả 9 con số phân bố lớp ở trên
> khớp tuyệt đối với dữ liệu thật. Do đó cách chọn cột `iloc[:, 8:]` trong `train_model.py`
> bản main lấy **đúng 395 feature, KHÔNG rò rỉ nhãn** — bộ tham số dưới đây hợp lệ.
> (Cảnh báo: cách chọn cột đó chỉ đúng với parquet của `build_dataset.py`. Parquet của
> `build_features.py` có thêm cột `label` và sẽ bị kéo nhãn vào làm feature.)

Hàm mục tiêu (`objective`) được thiết lập lấy **Macro F1-score** trên tập Validation làm thước đo đánh giá. Việc dùng Macro F1 giúp tạo áp lực buộc mô hình phải học cách phân biệt tốt cả các lớp thiểu số thay vì tập trung "ăn điểm" ở lớp đa số (`nền`).

> **⚠️ Macro F1 ở đây KHÔNG phải chỉ số của dự án.** Nó đo ở **mức cửa sổ trượt**, trước
> ngưỡng và trước NMS, và cộng cả lớp `nền` vào điểm. Chỉ số §5 của `KE_HOACH.md` là
> **`mAP_macro` mức box** (trung bình `AP_occupied` + `AP_empty` sau ngưỡng + NMS), sàn
> Baseline A = 0.5176 trên val. Hai con số không so sánh được với nhau.
> Macro F1 ở đây chỉ dùng làm **proxy rẻ để dò tham số** — xem §6 cho mAP thật.

## 2. Quá trình Dò tìm Tham số bằng Optuna (10 Trials)
Không gian tìm kiếm được giới hạn trong 10 lượt chạy (Trials) bằng thuật toán TPESampler. Dưới đây là lịch sử đánh giá trên tập Validation:

| Trial | n_estimators | criterion | max_depth | min_samples_split | min_samples_leaf | class_weight | Macro F1 (Val) |
|---:|---:|---|---:|---:|---:|---|---:|
| 0 | 100 | entropy | 8 | 100 | 40 | balanced | 0,8630 |
| 1 | 200 | gini | 7 | 1600 | 30 | balanced | 0,8457 |
| 2 | 50 | entropy | 10 | 1700 | 90 | None | 0,8651 |
| 3 | 150 | gini | 18 | 2000 | 40 | None | 0,8593 |
| **4** | **200** | **gini** | **26** | **200** | **50** | **balanced** | **0,9519** |
| 5 | 200 | gini | 20 | 600 | 150 | balanced | 0,9262 |
| 6 | 50 | entropy | 12 | 1100 | 170 | None | 0,8886 |
| 7 | 100 | gini | 26 | 1400 | 70 | None | 0,8795 |
| 8 | 200 | entropy | 14 | 1200 | 200 | balanced | 0,9008 |
| 9 | 150 | gini | 21 | 2000 | 10 | balanced | 0,9128 |

## 3. Bộ Tham số Tối ưu
**Trial 4** tạo ra sự bứt phá hoàn toàn với điểm Validation cao nhất (0.9519). Mô hình cần số lượng cây lớn, độ sâu sâu hơn đáng kể so với Cây quyết định đơn lẻ để bắt chi tiết, kết hợp với ngưỡng mẫu tách hợp lý để chống nhiễu:
* `n_estimators`: 200
* `criterion`: gini
* `max_depth`: 26
* `min_samples_split`: 200
* `min_samples_leaf`: 50
* `class_weight`: balanced

## 4. Kết quả Chi tiết trên Tập Test
Khi mang bộ tham số xuất sắc này đi dự đoán tập Test thực tế, mô hình ghi nhận sự sụt giảm hiệu suất mạnh, đặc biệt ở năng lực bắt lỗi lớp đỗ xe trống.

**Tổng quan điểm số:**
* **Train Accuracy:** 0.9935
* **Test Accuracy:** 0.8018
* **Test Macro F1:** 0.77

**Classification Report (Tập Test):**
| Class | Precision | Recall | F1-score | Support |
| :--- | ---: | ---: | ---: | ---: |
| **ô trống (0)** | 0.97 | 0.37 | 0.54 | 114.116 |
| **có xe (1)** | 0.95 | 0.92 | 0.93 | 93.323 |
| **nền (2)** | 0.73 | 0.98 | 0.84 | 210.846 |

**Confusion Matrix (Tập Test):**
| Thực tế \ Dự đoán | ô trống (0) | có xe (1) | nền (2) |
| :--- | ---: | ---: | ---: |
| **ô trống (0)** | 42.287 | 1.368 | **70.461** |
| **có xe (1)** | 842 | 85.871 | 6.610 |
| **nền (2)** | 395 | 3.222 | 207.229 |

## 5. Phân tích Hiện trạng
* Mặc dù Random Forest thể hiện năng lực tổng quát hóa tốt hơn Decision Tree (thể hiện qua các chỉ số F1 tổng thể), nó vẫn gục ngã trước bài toán **Lệch phân phối dữ liệu (Distribution Shift)**.
* Chỉ số Recall của lớp `ô trống` chênh lệch cực lớn (đạt 0.88 ở Validation nhưng rơi tự do xuống 0.37 ở Test). 
* Hệ quả là có tới **70.461** ô đỗ xe trống bị mô hình nhận diện sai thành nền. Ranh giới đặc trưng học được từ môi trường Train/Val hoàn toàn không đủ bao quát để xử lý các biến thể hình ảnh mới xuất hiện trong môi trường Test.
* Lưu ý đọc số: Test Accuracy 0.8018 nghe khá, nhưng lớp `nền` chiếm 50,4% tập test nên mốc đoán-bừa đã là 0.504; và recall 0.37 của lớp `ô trống` nghĩa là **mô hình bỏ sót 63% số ô trống** — vốn chính là đầu ra của sản phẩm. Đây đúng là lý do dự án không dùng accuracy/macro F1 làm chỉ số §5.

---

## 6. Kiểm chứng bằng mAP — chỉ số thật của dự án (05/09)

Bộ tham số §3 đã được port vào `train_model.py` (`--preset optuna`) và chạy qua **toàn bộ đường
dự đoán** — `predict_proba` → ngưỡng → NMS → `evaluate_pklot.evaluate` — trên **val**, để có
`mAP_macro` thay vì chỉ macro F1 mức cửa sổ.

```bash
cd src
python train_model.py --model rf --preset optuna --sweep-threshold --save ../models/rf_optuna.joblib
python train_model.py --model rf --sweep-threshold          # đối chứng: tham số mặc định
```

### Kết quả trên val (`score_thr=0.5`, `nms_iou=0.20`, phủ quyết nền BẬT)

| | **Preset Optuna** (§3) | Mặc định dự án | Chênh |
|---|---:|---:|---:|
| `n_estimators` / `max_depth` / `min_samples_leaf` | 200 / 26 / 50 | 300 / None / 5 | |
| **`mAP_macro`** | **0.7787** | 0.6042 | **+0.1745** |
| `AP_occupied` | 0.8162 | 0.6487 | +0.1675 |
| `AP_empty` | 0.7413 | 0.5597 | +0.1816 |
| **`free_slots_MAE`** (ô) | **3.31** | 8.00 | **−4.69** |
| `occupancy_MAE_pp` | 4.00 | 2.92 | +1.08 |
| Số box sinh ra | 884 | 491 | |
| Thời gian train | **332s** | 598s | −45% |
| macro F1 mức cửa sổ (val) | 0.955 | 0.855 | +0.100 |

**Kết luận: bộ tham số Optuna thắng rõ rệt** — hơn 0.1745 mAP, sai số đếm chỗ trống giảm từ
8,00 xuống 3,31 ô, và train nhanh hơn gấp rưỡi.

Đối chiếu tiêu chí `KE_HOACH.md` §5 (val): sàn Baseline A = 0.5176 → **vượt +0.2611**;
ngưỡng "⚠️ Tối thiểu" mAP > 0.60 → **đạt**; ngưỡng "✅ Tốt" mAP > 0.70 → **đạt**,
nhưng điều kiện đi kèm *sai số chỗ trống < 3 ô* thì **3,31 vẫn chưa đạt** (thiếu 0,31 ô).

### Vì sao mặc định thua: quá khớp

`max_depth=None` + `min_samples_leaf=5` cho cây sâu trung bình **84,1** (sâu nhất 118) và
**accuracy trên train = 1.0000** — thuộc lòng tập train. Trên val nó trở nên quá dè dặt:
precision 1.000 cho cả hai lớp đối tượng nhưng recall chỉ 0.642 (`ô trống`) và 0.693 (`có xe`),
chỉ sinh 491 box. Preset chặn `max_depth=26` và `min_samples_leaf=50` nên tổng quát tốt hơn hẳn.

### Quét ngưỡng × NMS (val, `mAP_macro`) — preset

| `score_thr` \ `nms_iou` | 0.10 | 0.15 | **0.20** | 0.25 | 0.30 | 0.45 |
|---|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.7048 | 0.7670 | 0.8073 | 0.8056 | 0.6461 | 0.3220 |
| **0.4** | 0.7044 | 0.7685 | **0.8111** | 0.8090 | 0.6450 | 0.3222 |
| 0.5 | 0.6876 | 0.7485 | 0.7787 | 0.7749 | 0.6306 | 0.3143 |
| 0.6 | 0.6498 | 0.7060 | 0.7399 | 0.7368 | 0.5947 | 0.2905 |
| 0.7 | 0.5849 | 0.6202 | 0.6453 | 0.6450 | 0.5220 | 0.2603 |
| 0.8 | 0.4036 | 0.4335 | 0.4560 | 0.4536 | 0.3646 | 0.1793 |
| 0.9 | 0.1982 | 0.2093 | 0.2137 | 0.2114 | 0.1855 | 0.0886 |

Tốt nhất **0.8111** tại `score_thr=0.4, nms_iou=0.20` (976 box) — hơn cấu hình hiện tại của
`config.py` thêm +0.0324. Cột `nms_iou=0.45` sụp xuống ~0.32 tái xác nhận vách
`(48/72)² = 0.4444` đã ghi trong `config.py`: trên ngưỡng đó NMS không gộp được cửa sổ đồng tâm
ở hai tỉ lệ liền nhau.

### Ghi chú về độ tin cậy của macro F1 làm proxy

Ở lần chạy này macro F1 mức cửa sổ **xếp hạng đúng** hai cấu hình (0.955 > 0.855, cùng chiều với
mAP 0.7787 > 0.6042) — nên việc dùng nó làm proxy rẻ để dò 10 trial là hợp lý. Nhưng **độ lớn thì
sai lệch nặng**: 0.9519 của Trial 4 không hề tương ứng với mAP 0.95, mà là 0.7787. Dùng macro F1
để *xếp hạng ứng viên* thì được; dùng nó để *báo cáo kết quả* thì không.

> Dữ liệu thô: `results.csv`, hai dòng ngày 2026-09-05 (`Random Forest + preset optuna [cắt
> rotated…]` và `Random Forest [cắt rotated…]`). Dòng đối chứng tái lập **chính xác** dòng
> 2026-09-04 (0.6042 / 0.6487 / 0.5597 / MAE 8.00), xác nhận việc port tham số không làm đổi
> hành vi mặc định.
