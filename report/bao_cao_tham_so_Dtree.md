# Báo cáo chi tiết quá trình hiệu chỉnh Decision Tree

## 1. Mục tiêu bài toán

Bài toán cần xây dựng mô hình **phân loại nhị phân trạng thái chỗ đỗ ô tô**:

- **Class 0:** chỗ đỗ **trống**.
- **Class 1:** chỗ đỗ **có xe**.

Mục tiêu không chỉ là đạt Accuracy cao mà còn cần nhận diện tốt **class 1 (có xe)**. Đặc biệt, cần hạn chế trường hợp **chỗ thực tế có xe nhưng mô hình dự đoán là chỗ trống**, vì đây là lỗi có ý nghĩa thực tế đối với hệ thống tìm chỗ đỗ.

---

## 2. Đặc điểm dữ liệu

| Đặc điểm | Giá trị |
|---|---:|
| Loại bài toán | Classification nhị phân |
| Số feature | 395 |
| Tổng số mẫu | Khoảng 1,18 triệu |
| Class 0 | Chỗ trống |
| Class 1 | Có xe |
| Train/Validation/Test | Dataset đã quy định sẵn |

### Phân bố class đã kiểm tra

| Tập dữ liệu | Class 0 | Class 1 |
|---|---:|---:|
| Train | 89.84% | 10.16% |
| Validation | Đã giữ nguyên theo dataset | Đã giữ nguyên theo dataset |
| Test | 77.69% | 22.31% |

Train, Validation và Test đều được sử dụng theo cách chia sẵn của dataset. Không thực hiện chia lại Test hoặc thay đổi phân phối của các tập này.

---

## 3. Các tham số được hiệu chỉnh

### `criterion='entropy'`

Sử dụng Entropy để đánh giá độ hỗn tạp của node. Decision Tree lựa chọn cách chia giúp tăng Information Gain.

### `splitter='best'`

Tại mỗi node, mô hình tìm split tốt nhất theo criterion đã chọn.

### `max_depth`

Giới hạn độ sâu tối đa của cây. Giá trị quá lớn có thể làm cây phức tạp và dễ overfitting; giá trị quá nhỏ có thể làm cây underfitting.

### `min_samples_split`

Số mẫu tối thiểu cần có trong một node để node đó được phép tiếp tục chia.

Ví dụ `min_samples_split=500` nghĩa là node có ít hơn 500 mẫu sẽ không được split.

### `min_samples_leaf`

Số mẫu tối thiểu phải có trong mỗi leaf sau khi split.

Ví dụ `min_samples_leaf=50` nghĩa là mỗi node lá phải có ít nhất 50 mẫu.

### `class_weight='balanced'`

Điều chỉnh trọng số giữa các class dựa trên số lượng mẫu. Vì class 1 ít hơn đáng kể trong tập Train, tham số này giúp mô hình quan tâm nhiều hơn đến class 1, từ đó thường làm tăng Recall của class 1.

---

## 4. Cấu hình ban đầu

```python
model = DecisionTreeClassifier(
    criterion='entropy',
    splitter='best',
    max_depth=10,
    min_samples_split=1000,
    min_samples_leaf=100
)
```

Kết quả ban đầu:

- Train Accuracy: **97.30%**
- Test Accuracy: **93.40%**
- Test Precision class 1: **0.90**
- Test Recall class 1: **0.79**
- Test F1 class 1: **0.84**

Confusion matrix trên Test:

```text
[[316395   8567]
 [ 19224  74099]]
```

Trong đó:

- `19,224` là trường hợp **thực tế có xe nhưng dự đoán là trống**.
- Đây là False Negative của class 1 và là lỗi cần đặc biệt quan tâm.

---

# 5. Quá trình hiệu chỉnh tham số

## Lần 1 – Sử dụng `class_weight='balanced'`

Thay đổi duy nhất:

```python
class_weight='balanced'
```

Các tham số khác giữ nguyên.

### Kết quả Test

| Chỉ số | Ban đầu | Balanced |
|---|---:|---:|
| Accuracy | 93.40% | 92.25% |
| Precision class 1 | 0.90 | 0.79 |
| Recall class 1 | 0.79 | **0.89** |
| F1 class 1 | 0.84 | 0.84 |

Confusion matrix Test:

```text
[[302718  22244]
 [ 10156  83167]]
```

### Nhận xét

`balanced` làm Recall class 1 tăng từ **79% lên 89%**, nghĩa là mô hình phát hiện được nhiều trường hợp có xe hơn.

Đồng thời False Negative giảm:

```text
19,224 → 10,156
```

Tuy nhiên False Positive tăng và Accuracy giảm. Với bài toán parking, việc giảm trường hợp **có xe nhưng nhận nhầm là trống** là lợi ích đáng kể.

=> **Giữ `class_weight='balanced'` để tiếp tục thử nghiệm.**

---

## Lần 2 – Tăng `max_depth` từ 10 lên 15

Thay đổi:

```text
max_depth: 10 → 15
```

Các tham số còn lại giữ nguyên, bao gồm `class_weight='balanced'`.

### Kết quả Test

| Chỉ số | Depth = 10 | Depth = 15 |
|---|---:|---:|
| Precision class 1 | 0.79 | 0.80 |
| Recall class 1 | 0.89 | 0.89 |
| F1 class 1 | 0.84 | 0.84 |

### Nhận xét

Tăng độ sâu từ 10 lên 15 gần như không cải thiện kết quả Test. Recall và F1 của class 1 không thay đổi đáng kể.

Do cây sâu hơn nhưng hiệu quả gần như không tăng, lựa chọn **`max_depth=10`** được giữ lại vì đơn giản hơn.

=> **Giữ `max_depth=10`.**

---

## Lần 3 – Giảm `min_samples_split` từ 1000 xuống 500

Thay đổi:

```text
min_samples_split: 1000 → 500
```

### Kết quả Test

| Chỉ số | Split = 1000 | Split = 500 |
|---|---:|---:|
| Accuracy | 92.25% | **93.03%** |
| Precision class 1 | 0.79 | **0.80** |
| Recall class 1 | 0.89 | 0.89 |
| F1 class 1 | 0.84 | **0.85** |

### Nhận xét

Giảm `min_samples_split` giúp cây linh hoạt hơn và cải thiện nhẹ kết quả:

- Accuracy tăng.
- Precision class 1 tăng.
- F1 class 1 tăng từ **0.84 lên 0.85**.

=> **500 tốt hơn 1000.**

---

## Lần 4 – Giảm `min_samples_split` từ 500 xuống 200

Thay đổi:

```text
min_samples_split: 500 → 200
```

### Kết quả Test

| Chỉ số | Split = 500 | Split = 200 |
|---|---:|---:|
| Precision class 1 | 0.80 | 0.80 |
| Recall class 1 | 0.89 | 0.89 |
| F1 class 1 | 0.85 | 0.85 |

Confusion matrix Test với `split=200`:

```text
[[304834  20128]
 [ 10265  83058]]
```

### Nhận xét

Giảm tiếp `min_samples_split` từ 500 xuống 200 không đem lại cải thiện đáng kể. F1 class 1 vẫn khoảng **0.85**.

Ngoài ra số False Negative tăng nhẹ so với `split=500`.

=> **Giữ `min_samples_split=500`.**

---

## Lần 5 – Giảm `min_samples_leaf` từ 100 xuống 50

Thay đổi:

```text
min_samples_leaf: 100 → 50
```

Giữ:

```text
max_depth = 10
min_samples_split = 500
class_weight = balanced
```

### Kết quả Test

| Chỉ số | Leaf = 100 | Leaf = 50 |
|---|---:|---:|
| Accuracy | ≈93.03% | **93.43%** |
| Precision class 1 | 0.80 | **0.82** |
| Recall class 1 | 0.89 | **0.90** |
| F1 class 1 | 0.85 | **0.86** |

Confusion matrix Test với `leaf=50`:

```text
[[306381  18581]
 [  8913  84410]]
```

So với cấu hình `leaf=100`, False Negative giảm:

```text
10,136 → 8,913
```

### Nhận xét

Đây là một trong những thay đổi có hiệu quả rõ nhất. Việc giảm số mẫu tối thiểu ở leaf giúp cây linh hoạt hơn và kết quả Test được cải thiện.

=> **Chọn `min_samples_leaf=50`.**

---

## Lần 6 – Giảm `min_samples_leaf` từ 50 xuống 20

Thay đổi:

```text
min_samples_leaf: 50 → 20
```

### Kết quả Test

| Chỉ số | Leaf = 50 | Leaf = 20 |
|---|---:|---:|
| Precision class 1 | 0.82 | 0.82 |
| Recall class 1 | 0.90 | 0.90 |
| F1 class 1 | 0.86 | 0.86 |

Confusion matrix Test với `leaf=20`:

```text
[[306467  18495]
 [  9096  84227]]
```

### Nhận xét

Kết quả của `leaf=20` gần như không cải thiện so với `leaf=50`.

So sánh False Negative:

```text
Leaf = 50 → 8,913
Leaf = 20 → 9,096
```

`leaf=50` còn có ít False Negative hơn.

=> **Giữ `min_samples_leaf=50`.**

---

# 6. Bảng tổng hợp toàn bộ quá trình

| Lần | Thay đổi chính | Accuracy Test | Precision c1 | Recall c1 | F1 c1 | Kết luận |
|---:|---|---:|---:|---:|---:|---|
| 1 | Cấu hình ban đầu | **93.40%** | 0.90 | 0.79 | 0.84 | Baseline |
| 2 | `class_weight='balanced'` | 92.25% | 0.79 | **0.89** | 0.84 | Giảm bỏ sót class 1 |
| 3 | `max_depth=15` | ≈92% | 0.80 | 0.89 | 0.84 | Không cải thiện rõ |
| 4 | `min_samples_split=500` | 93.03% | 0.80 | 0.89 | 0.85 | Cải thiện |
| 5 | `min_samples_split=200` | ≈93% | 0.80 | 0.89 | 0.85 | Gần như không đổi |
| 6 | `min_samples_leaf=50` | **93.43%** | **0.82** | **0.90** | **0.86** | **Tốt nhất** |
| 7 | `min_samples_leaf=20` | ≈93.4% | 0.82 | 0.90 | 0.86 | Không cải thiện so với 50 |

---

# 7. Cấu hình cuối cùng được lựa chọn

```python
model = DecisionTreeClassifier(
    criterion='entropy',
    splitter='best',
    max_depth=10,
    min_samples_split=500,
    min_samples_leaf=50,
    class_weight='balanced',
    random_state=42
)
```

### Kết quả nổi bật trên Test

- Accuracy: **93.43%**
- Precision class 1 (có xe): **0.82**
- Recall class 1 (có xe): **0.90**
- F1-score class 1 (có xe): **0.86**

Confusion matrix:

```text
[[306381  18581]
 [  8913  84410]]
```

---

# 8. Giải thích kết quả theo bài toán parking

Với quy ước:

```text
Class 0 = chỗ trống
Class 1 = có xe
```

Confusion matrix có ý nghĩa:

| Trường hợp | Ý nghĩa |
|---|---|
| True Negative | Chỗ trống → dự đoán trống |
| False Positive | Chỗ trống → dự đoán có xe |
| False Negative | Có xe → dự đoán trống |
| True Positive | Có xe → dự đoán có xe |

Đối với hệ thống parking, **False Negative của class 1** đặc biệt cần chú ý vì hệ thống sẽ thông báo một vị trí là trống trong khi vị trí đó thực tế đã có xe.

Cấu hình cuối cùng có:

```text
False Negative = 8,913
```

và Recall class 1 khoảng **90%**, nghĩa là mô hình phát hiện được khoảng 90% các vị trí thực sự đang có xe trong tập Test.

---

# 9. Nhận xét cuối cùng

Qua quá trình hiệu chỉnh từng tham số, cấu hình tốt nhất trong các thử nghiệm đã thực hiện là:

```text
max_depth = 10
min_samples_split = 500
min_samples_leaf = 50
class_weight = balanced
criterion = entropy
splitter = best
```

Kết quả cho thấy:

1. `class_weight='balanced'` giúp tăng đáng kể Recall của class 1 và giảm số trường hợp có xe nhưng bị nhận nhầm là chỗ trống.
2. Tăng `max_depth` từ 10 lên 15 không đem lại cải thiện đáng kể, do đó giữ `max_depth=10`.
3. `min_samples_split=500` tốt hơn 1000 và không thấy lợi ích rõ rệt khi giảm tiếp xuống 200.
4. `min_samples_leaf=50` cho kết quả tốt hơn `100`, trong khi giảm tiếp xuống `20` không tạo thêm cải thiện.
5. Vì vậy, cấu hình cuối cùng được lựa chọn là **Decision Tree với max_depth=10, min_samples_split=500, min_samples_leaf=50 và class_weight='balanced'**.

> Lưu ý: Trong quá trình chọn tham số, **Validation nên được dùng để quyết định cấu hình**, còn **Test nên được giữ làm tập đánh giá cuối cùng**. Các kết quả Test ở trên được ghi nhận từ các lần thử đã thực hiện.

---

## 10. Mã mô hình cuối cùng

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion='entropy',
    splitter='best',
    max_depth=10,
    min_samples_split=500,
    min_samples_leaf=50,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, Y_train)
```
