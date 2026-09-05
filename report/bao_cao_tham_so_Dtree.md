# BÁO CÁO TỐI ƯU HÓA OPTUNA VÀ ĐÁNH GIÁ MÔ HÌNH: DECISION TREE

> **⚠️ ĐÃ ĐÁNH SỐ LẠI NHÃN (05/09, lúc merge main → feature/nhan-ui).**
> Bản gốc dùng `0=background, 1=empty, 2=car`. Toàn bộ dự án dùng **`0 = ô trống (empty)`,
> `1 = có xe (car)`, `2 = nền (background)`**, nên báo cáo này đã đánh số lại theo quy ước đó.
> **Các chỉ số không đổi** — chúng gắn với TÊN lớp. Chỉ số thứ tự và thứ tự hàng/cột của ma
> trận nhầm lẫn được hoán vị. Chi tiết xem `report/bao_cao_random_forest.md`.

## 1. Cấu trúc Dữ liệu và Thiết lập Mục tiêu
Dự án sử dụng Optuna để tự động tìm kiếm bộ tham số tối ưu cho thuật toán Decision Tree (Cây quyết định) nhằm phân loại 3 lớp đặc trưng HOG: `0` (ô trống / empty), `1` (có xe / car) và `2` (nền / background).

* **Tập Train:** 69.541 (0) | 71.609 (1) | 563.365 (2) — *Tổng: 704.515 mẫu*
* **Tập Validation:** 7.007 (0) | 3.361 (1) | 47.032 (2) — *Tổng: 57.400 mẫu*
* **Tập Test:** 114.116 (0) | 93.323 (1) | 210.846 (2) — *Tổng: 418.285 mẫu*

Hàm mục tiêu (`objective`) được thiết lập lấy **Macro F1-score** trên tập Validation làm thước đo duy nhất để ép mô hình phải học tốt cả lớp thiểu số (`ô trống`, `có xe`), tránh xu hướng chỉ đoán lớp đa số (`nền`).

> **⚠️ Macro F1 mức cửa sổ, KHÔNG phải chỉ số §5.** Chỉ số của dự án là `mAP_macro` mức box
> (sau ngưỡng + NMS), sàn Baseline A = 0.5176 trên val. Xem `bao_cao_random_forest.md` §6.
>
> Bộ tham số §3 đã được port sang `train_model.py` (`--preset optuna`, dùng kèm `--model dt`)
> nhưng **CHƯA được đo bằng mAP**. Chạy để có con số thật:
> ```bash
> cd src && python train_model.py --model dt --preset optuna --sweep-threshold
> ```
> Với Random Forest, preset đã được kiểm chứng và hơn tham số mặc định +0.1745 mAP
> (0.7787 vs 0.6042) — xem `bao_cao_random_forest.md` §6.

## 2. Quá trình Dò tìm Tham số bằng Optuna (10 Trials)
Không gian tìm kiếm được giới hạn trong 10 lượt chạy (Trials). Dưới đây là lịch sử đánh giá trên tập Validation:

| Trial | criterion | max_depth | min_samples_split | min_samples_leaf | class_weight | Macro F1 (Val) |
|---:|---|---:|---:|---:|---|---:|
| **0** | **entropy** | **13** | **100** | **20** | **balanced** | **0,8660** |
| 1 | entropy | 27 | 1400 | 90 | balanced | 0,7944 |
| 2 | gini | 29 | 500 | 180 | None | 0,8207 |
| 3 | entropy | 11 | 300 | 100 | balanced | 0,8243 |
| 4 | gini | 6 | 1000 | 30 | None | 0,7568 |
| 5 | entropy | 18 | 1000 | 100 | balanced | 0,8073 |
| 6 | gini | 8 | 1300 | 50 | balanced | 0,7705 |
| 7 | entropy | 6 | 600 | 190 | balanced | 0,7269 |
| 8 | gini | 19 | 1700 | 40 | None | 0,8060 |
| 9 | entropy | 10 | 400 | 100 | None | 0,8481 |

## 3. Bộ Tham số Tối ưu
**Trial 0** mang lại kết quả Validation tốt nhất. Thuật toán chọn độ sâu cây ở mức trung bình (13), yêu cầu số lượng mẫu đủ lớn ở các nút lá để chống nhiễu, đồng thời kích hoạt trọng số cân bằng:
* `criterion`: entropy
* `max_depth`: 13
* `min_samples_split`: 100
* `min_samples_leaf`: 20
* `class_weight`: balanced

## 4. Kết quả Chi tiết trên Tập Test
Khi mang bộ tham số tối ưu này huấn luyện và dự đoán trên tập Test, Decision Tree gặp khó khăn lớn trong việc khái quát hóa dữ liệu so với Random Forest.

**Tổng quan điểm số:**
* **Train Accuracy:** 0.9695
* **Test Accuracy:** 0.7804
* **Test Macro F1:** 0.76

**Classification Report (Tập Test):**
| Class | Precision | Recall | F1-score | Support |
| :--- | ---: | ---: | ---: | ---: |
| **ô trống (0)** | 0.83 | 0.52 | 0.64 | 114.116 |
| **có xe (1)** | 0.86 | 0.76 | 0.81 | 93.323 |
| **nền (2)** | 0.74 | 0.93 | 0.82 | 210.846 |

**Confusion Matrix (Tập Test):**
| Thực tế \ Dự đoán | ô trống (0) | có xe (1) | nền (2) |
| :--- | ---: | ---: | ---: |
| **ô trống (0)** | 59.675 | 2.663 | **51.778** |
| **có xe (1)** | 5.449 | 71.305 | 16.569 |
| **nền (2)** | 6.439 | 8.951 | 195.456 |

## 5. Phân tích Hiện trạng
Giống như Random Forest, Cây quyết định cũng đang vấp phải bài toán "Lệch phân phối dữ liệu" (Distribution Shift) kết hợp với hạn chế nội tại của một cây đơn lẻ:
* Dù đã bật `class_weight='balanced'`, mô hình vẫn nhận diện sai **51.778** ô đỗ xe trống (`ô trống`) thành nền (`nền`), kéo Recall của lớp này tụt xuống mức 0.52.
* Những đặc trưng hình ảnh của tập Test rõ ràng nằm ngoài ranh giới mà cây quyết định đã học được từ tập Validation (nơi nó đạt Macro F1 0.86).
* **Định hướng xử lý:** Đối với Decision Tree, việc áp dụng Hard Negative Mining (lấy các mẫu đoán sai) từ tập Validation sẽ gặp rào cản tương tự như Random Forest (số lượng mẫu sai ở Val quá ít để phản ánh được độ khó của tập Test). Nếu dự án chọn con đường "Domain Adaptation", 51.778 mẫu lỗi này chính là kho dữ liệu quý giá cần được gộp lại vào Train để định hình lại các nhánh cây quyết định.