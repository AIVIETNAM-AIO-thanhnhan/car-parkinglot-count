# BÁO CÁO TỐI ƯU HÓA OPTUNA VÀ ĐÁNH GIÁ MÔ HÌNH: RANDOM FOREST

## 1. Cấu trúc Dữ liệu và Thiết lập Mục tiêu
Dự án sử dụng Optuna để tự động tìm kiếm bộ tham số tối ưu cho thuật toán Random Forest (Rừng ngẫu nhiên) nhằm phân loại 3 lớp đặc trưng HOG: `0` (background), `1` (empty) và `2` (car).

* **Tập Train:** 563.365 (0) | 69.541 (1) | 71.609 (2)
* **Tập Validation:** 47.032 (0) | 7.007 (1) | 3.361 (2) — *Tổng: 57.400 mẫu*
* **Tập Test:** 210.846 (0) | 114.116 (1) | 93.323 (2) — *Tổng: 418.285 mẫu*

Hàm mục tiêu (`objective`) được thiết lập lấy **Macro F1-score** trên tập Validation làm thước đo đánh giá. Việc dùng Macro F1 giúp tạo áp lực buộc mô hình phải học cách phân biệt tốt cả các lớp thiểu số thay vì tập trung "ăn điểm" ở lớp đa số (`background`).

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
| **background (0)** | 0.73 | 0.98 | 0.84 | 210.846 |
| **empty (1)** | 0.97 | 0.37 | 0.54 | 114.116 |
| **car (2)** | 0.95 | 0.92 | 0.93 | 93.323 |

**Confusion Matrix (Tập Test):**
| Thực tế \ Dự đoán | background (0) | empty (1) | car (2) |
| :--- | ---: | ---: | ---: |
| **background (0)**| 207.229 | 395 | 3.222 |
| **empty (1)** | **70.461** | 42.287 | 1.368 |
| **car (2)** | 6.610 | 842 | 85.871 |

## 5. Phân tích Hiện trạng
* Mặc dù Random Forest thể hiện năng lực tổng quát hóa tốt hơn Decision Tree (thể hiện qua các chỉ số F1 tổng thể), nó vẫn gục ngã trước bài toán **Lệch phân phối dữ liệu (Distribution Shift)**.
* Chỉ số Recall của lớp `empty` chênh lệch cực lớn (đạt 0.88 ở Validation nhưng rơi tự do xuống 0.37 ở Test). 
* Hệ quả là có tới **70.461** ô đỗ xe trống bị mô hình nhận diện sai thành bối cảnh. Ranh giới đặc trưng học được từ môi trường Train/Val hoàn toàn không đủ bao quát để xử lý các biến thể hình ảnh mới xuất hiện trong môi trường Test.
