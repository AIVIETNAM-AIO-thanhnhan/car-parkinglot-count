# BÁO CÁO PHÂN TÍCH HIỆN TƯỢNG LỆCH PHÂN PHỐI (DISTRIBUTION SHIFT)

## 1. Phát hiện Bất thường giữa Validation và Test
Sau khi tinh chỉnh tham số bằng Optuna cho mô hình Random Forest, quá trình đánh giá chéo đã bộc lộ độ lệch nghiêm trọng về phân phối dữ liệu (Distribution Shift) giữa tập Validation và tập Test.

*   **Validation Macro F1:** 0.95
*   **Test Macro F1:** 0.77

## 2. Kết quả Chi tiết trên Tập Validation
Mô hình hoạt động xuất sắc trên tập Validation, không hề có dấu hiệu bỏ rơi lớp `empty` như khi dự đoán trên tập Test.

| Class | Precision | Recall | F1-score | Support |
| :--- | ---: | ---: | ---: | ---: |
| **background (0)** | 0.98 | 0.99 | 0.99 | 47.032 |
| **empty (1)** | 0.95 | 0.88 | 0.92 | 7.007 |
| **car (2)** | 0.95 | 0.98 | 0.96 | 3.361 |

*Tổng số mẫu (Accuracy = 0.98): 57.400*

## 3. Kết quả Chi tiết trên Tập Test
Trái ngược với Validation, mô hình sụp đổ hoàn toàn ở năng lực nhận diện lớp `empty` trên tập Test thực tế.

**Classification Report:**
| Class | Precision | Recall | F1-score | Support |
| :--- | ---: | ---: | ---: | ---: |
| **background (0)** | 0.73 | 0.98 | 0.84 | 210.846 |
| **empty (1)** | 0.97 | 0.37 | 0.54 | 114.116 |
| **car (2)** | 0.95 | 0.92 | 0.93 | 93.323 |

*Tổng số mẫu (Accuracy = 0.80): 418.285*

**Confusion Matrix:**
| Thực tế \ Dự đoán | background (0) | empty (1) | car (2) |
| :--- | ---: | ---: | ---: |
| **background (0)**| 207.229 | 395 | 3.222 |
| **empty (1)** | **70.461** | 42.287 | 1.368 |
| **car (2)** | 6.610 | 842 | 85.871 |

## 4. Phân tích Nguyên nhân
Sự chênh lệch Recall của lớp `empty` (0.88 trên Validation so với 0.37 trên Test) phản ánh rõ rệt hiện tượng ranh giới đặc trưng bị thay đổi:
*   Tập Test (hơn 418.000 mẫu) mang những biến thể hình ảnh rất nặng (như góc camera khác, điều kiện ánh sáng, thời tiết, hoặc thiết kế bãi đỗ xe mới) mà tập Train và Validation không hề có.
*   Mô hình thực chất không bị overfitting (học thuộc lòng) theo định nghĩa cơ bản. Thay vào đó, nó đã học rất tốt và khái quát hóa thành công ranh giới của "Môi trường A" (tập Train/Val), nhưng lại bị đem đi đánh giá ở "Môi trường B" (tập Test).

## 5. Nghịch lý của Kỹ thuật Hard Negative Mining hiện tại
Kế hoạch trích xuất các dự đoán sai (Hard Negatives) để huấn luyện tăng cường đang vấp phải một mâu thuẫn lớn về cấu trúc dữ liệu:
*   **Khai thác trên Validation:** Tuân thủ tuyệt đối việc cách ly tập thi. Tuy nhiên, với Recall 0.88, tập Validation chỉ sinh ra khoảng hơn 800 mẫu lỗi. Số lượng mẫu này không mang đặc trưng của Môi trường B, nên việc nhồi chúng vào tập Train sẽ không giúp mô hình giải quyết được hơn 70.000 ca dự đoán sai trong tập Test.
*   **Khai thác trên Test (Rò rỉ dữ liệu):** Việc bóc tách 80.000 mẫu lỗi từ Test để nhồi vào Train đã đẩy điểm Test vọt lên ngưỡng ảo 98.7%. Việc lấy chính dữ liệu thi để dạy cho mô hình gây ra rò rỉ (Data Leakage), làm mất đi tính khách quan của bộ test.

## 6. Đề xuất Giải pháp Xử lý
Dự án đứng trước hai hướng rẽ để giải quyết hiện trạng:
1.  **Chiến lược 1 :** Vẫn tiến hành HNM trên tập Validation. Chấp nhận điểm số tập Test có thể dậm chân tại chỗ để giữ nguyên tính trung thực của quy trình đánh giá.
2.  **Chiến lược 2 :** Phá vỡ quy tắc tập Test hiện tại. Lấy chính 80.000 mẫu khó của tập Test gộp vào Train để ép mô hình thích nghi với môi trường mới. Tuy nhiên, bắt buộc phải loại bỏ vai trò của tập Test này và thu thập/sử dụng một tập dữ liệu Test hoàn toàn mới để chấm điểm năng lực cuối cùng.
3.  **Chiến lược 3 :** Không tiến hành HNM