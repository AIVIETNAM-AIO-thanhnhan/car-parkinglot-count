# BÁO CÁO TỐI ƯU HÓA OPTUNA VÀ ĐÁNH GIÁ MÔ HÌNH: DECISION TREE

## 1. Cấu trúc Dữ liệu và Thiết lập Mục tiêu
Dự án sử dụng Optuna để tự động tìm kiếm bộ tham số tối ưu cho thuật toán Decision Tree (Cây quyết định) nhằm phân loại 3 lớp đặc trưng HOG: `0` (background), `1` (empty) và `2` (car).

* **Tập Train:** 563.365 (0) | 69.541 (1) | 71.609 (2)
* **Tập Validation:** 47.032 (0) | 7.007 (1) | 3.361 (2) — *Tổng: 57.400 mẫu*
* **Tập Test:** 210.846 (0) | 114.116 (1) | 93.323 (2) — *Tổng: 418.285 mẫu*

Hàm mục tiêu (`objective`) được thiết lập lấy **Macro F1-score** trên tập Validation làm thước đo duy nhất để ép mô hình phải học tốt cả lớp thiểu số (`empty`, `car`), tránh xu hướng chỉ đoán lớp đa số (`background`).

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
| **background (0)** | 0.74 | 0.93 | 0.82 | 210.846 |
| **empty (1)** | 0.83 | 0.52 | 0.64 | 114.116 |
| **car (2)** | 0.86 | 0.76 | 0.81 | 93.323 |

**Confusion Matrix (Tập Test):**
| Thực tế \ Dự đoán | background (0) | empty (1) | car (2) |
| :--- | ---: | ---: | ---: |
| **background (0)**| 195.456 | 6.439 | 8.951 |
| **empty (1)** | **51.778** | 59.675 | 2.663 |
| **car (2)** | 16.569 | 5.449 | 71.305 |

## 5. Phân tích Hiện trạng
Giống như Random Forest, Cây quyết định cũng đang vấp phải bài toán "Lệch phân phối dữ liệu" (Distribution Shift) kết hợp với hạn chế nội tại của một cây đơn lẻ:
* Dù đã bật `class_weight='balanced'`, mô hình vẫn nhận diện sai **51.778** ô đỗ xe trống (`empty`) thành bối cảnh (`background`), kéo Recall của lớp này tụt xuống mức 0.52.
* Những đặc trưng hình ảnh của tập Test rõ ràng nằm ngoài ranh giới mà cây quyết định đã học được từ tập Validation (nơi nó đạt Macro F1 0.86).
* **Định hướng xử lý:** Đối với Decision Tree, việc áp dụng Hard Negative Mining (lấy các mẫu đoán sai) từ tập Validation sẽ gặp rào cản tương tự như Random Forest (số lượng mẫu sai ở Val quá ít để phản ánh được độ khó của tập Test). Nếu dự án chọn con đường "Domain Adaptation", 51.778 mẫu lỗi này chính là kho dữ liệu quý giá cần được gộp lại vào Train để định hình lại các nhánh cây quyết định.