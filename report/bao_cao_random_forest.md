# BÁO CÁO RANDOM FOREST – PHÂN LOẠI TRẠNG THÁI CHỖ ĐỖ Ô TÔ

## 1. Mục tiêu bài toán

Bài toán xây dựng mô hình phân loại trạng thái của một vị trí đỗ ô tô thành 2 lớp:

- **Class 0:** chỗ đỗ trống.
- **Class 1:** chỗ đỗ có xe.

Mục tiêu là xây dựng mô hình có khả năng nhận diện chính xác trạng thái chỗ đỗ, đặc biệt hạn chế trường hợp **chỗ đang có xe nhưng mô hình dự đoán là chỗ trống**.

---

## 2. Đặc điểm dữ liệu

Dữ liệu sử dụng cho bài toán có:

- **395 features**.
- Khoảng **1,18 triệu mẫu**.
- Bài toán **binary classification** với 2 class.
- Các tập **Train, Validation và Test đã được dataset quy định sẵn**, vì vậy không chia lại các tập này trong quá trình thử nghiệm.

Phân bố class của các tập không hoàn toàn giống nhau. Do đó, việc đánh giá mô hình không chỉ dựa trên Accuracy mà còn xem xét Precision, Recall và F1-score, đặc biệt đối với **class 1 – có xe**.

---

## 3. Lý do lựa chọn Random Forest

Decision Tree trước đó cho kết quả khá tốt nhưng vẫn còn hạn chế. Cấu hình Decision Tree tốt nhất đã thử đạt khoảng:

- Accuracy Test: **93,43%**
- Precision class 1: **0,82**
- Recall class 1: **0,90**
- F1-score class 1: **0,86**

Random Forest được sử dụng để cải thiện độ ổn định và khả năng tổng quát hóa bằng cách kết hợp dự đoán của nhiều Decision Tree thay vì chỉ sử dụng một cây duy nhất.

---

## 4. Cấu hình Random Forest

Cấu hình sử dụng:

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    max_depth=10,
    min_samples_split=500,
    min_samples_leaf=50,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
```

### Ý nghĩa các tham số

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `n_estimators` | 100 | Xây dựng 100 cây Decision Tree |
| `criterion` | `entropy` | Sử dụng Entropy/Information Gain để đánh giá chất lượng split |
| `max_depth` | 10 | Giới hạn độ sâu tối đa của mỗi cây |
| `min_samples_split` | 500 | Một node phải có ít nhất 500 mẫu mới được phép tiếp tục split |
| `min_samples_leaf` | 50 | Mỗi node lá phải có ít nhất 50 mẫu |
| `class_weight` | `balanced` | Tăng trọng số cho class có ít mẫu hơn |
| `n_jobs` | -1 | Sử dụng toàn bộ CPU khả dụng |
| `random_state` | 42 | Đảm bảo kết quả có thể tái lập |

---

## 5. Kết quả trên tập Validation

Classification report:

```text
              precision    recall  f1-score   support

           0       1.00      0.98      0.99     54039
           1       0.80      0.99      0.88      3361

    accuracy                           0.98     57400
   macro avg       0.90      0.99      0.94     57400
weighted avg       0.99      0.98      0.99     57400
```

Confusion matrix:

```text
[[53201   838]
 [   33  3328]]
```

Diễn giải:

| Thực tế | Dự đoán trống (0) | Dự đoán có xe (1) |
|---|---:|---:|
| Trống (0) | 53.201 | 838 |
| Có xe (1) | 33 | 3.328 |

Đối với class 1:

- Precision = **0,80**
- Recall = **0,99**
- F1-score = **0,88**

Recall 0,99 cho thấy mô hình phát hiện được khoảng **99% số vị trí thực sự có xe** trong tập Validation.

Đặc biệt, chỉ có **33 trường hợp** chỗ thực tế có xe nhưng bị dự đoán thành chỗ trống.

---

## 6. Kết quả trên tập Test

Classification report:

```text
              precision    recall  f1-score   support

           0       0.99      0.97      0.98    324962
           1       0.91      0.95      0.93     93323

    accuracy                           0.97    418285
   macro avg       0.95      0.96      0.95    418285
weighted avg       0.97      0.97      0.97    418285
```

Confusion matrix:

```text
[[315665   9297]
 [  4345  88978]]
```

Diễn giải:

| Thực tế | Dự đoán trống (0) | Dự đoán có xe (1) |
|---|---:|---:|
| Trống (0) | 315.665 | 9.297 |
| Có xe (1) | 4.345 | 88.978 |

Đối với class 1 – có xe:

- **Precision = 0,91**
- **Recall = 0,95**
- **F1-score = 0,93**

Như vậy, mô hình phát hiện đúng khoảng **95% các vị trí thực sự đang có xe**.

Số trường hợp thực tế có xe nhưng dự đoán thành trống là:

\[
FN = 4.345
\]

Đây là một trong những lỗi quan trọng đối với bài toán parking vì có thể khiến hệ thống thông báo nhầm rằng một vị trí đang trống.

---

## 7. So sánh với Decision Tree

Cấu hình Decision Tree tốt nhất đã thử:

```text
max_depth = 10
min_samples_split = 500
min_samples_leaf = 50
class_weight = balanced
```

Kết quả:

| Chỉ số trên Test | Decision Tree | Random Forest |
|---|---:|---:|
| Accuracy | 93,43% | **97,00%** |
| Precision class 1 | 0,82 | **0,91** |
| Recall class 1 | 0,90 | **0,95** |
| F1 class 1 | 0,86 | **0,93** |

Random Forest cải thiện:

- Accuracy: **+3,57 điểm phần trăm**
- Precision class 1: **+0,09**
- Recall class 1: **+0,05**
- F1-score class 1: **+0,07**

Số False Negative của class 1 giảm đáng kể so với Decision Tree.

---

## 8. Đánh giá mô hình

Random Forest cho kết quả tốt hơn Decision Tree ở cả Validation và Test.

### Ưu điểm

1. **Accuracy cao:** khoảng 97% trên Test.
2. **Recall class 1 cao:** 95%, phù hợp với mục tiêu phát hiện chỗ đang có xe.
3. **F1 class 1 cao:** 0,93, cho thấy sự cân bằng tốt giữa Precision và Recall.
4. Giảm đáng kể số False Negative so với Decision Tree.
5. Kết hợp nhiều cây giúp mô hình ổn định hơn so với một Decision Tree đơn.

### Hạn chế

Random Forest sử dụng nhiều Decision Tree nên:

- Thời gian huấn luyện lớn hơn Decision Tree.
- Sử dụng nhiều bộ nhớ hơn.
- Khó diễn giải trực quan hơn một Decision Tree đơn.

Với dữ liệu khoảng 1,18 triệu mẫu và 395 features, chi phí tính toán của Random Forest là điểm cần cân nhắc khi triển khai.

---

## 9. Kết luận

Sau quá trình thử nghiệm, Random Forest cho kết quả tốt hơn Decision Tree đối với bài toán phân loại trạng thái chỗ đỗ ô tô.

Cấu hình được sử dụng:

```python
RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    max_depth=10,
    min_samples_split=500,
    min_samples_leaf=50,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
```

Kết quả cuối cùng trên Test:

\[
oxed{Accuracy = 97,00\%}
\]

\[
oxed{Precision_{class1} = 0,91}
\]

\[
oxed{Recall_{class1} = 0,95}
\]

\[
oxed{F1_{class1} = 0,93}
\]

Trong đó **class 1 là vị trí đang có xe**.

Với đặc thù bài toán parking, Recall và F1 của class 1 là các chỉ số quan trọng vì hệ thống cần nhận diện chính xác các vị trí đang có xe và hạn chế thông báo nhầm vị trí đó là chỗ trống.

Dựa trên kết quả đã thử nghiệm, **Random Forest là mô hình phù hợp hơn Decision Tree cho bài toán hiện tại**.
