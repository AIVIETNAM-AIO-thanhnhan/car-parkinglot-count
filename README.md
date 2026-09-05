# Đếm Chỗ Đỗ Xe

Đưa vào một ảnh bãi đỗ, nhận về **số xe · số chỗ trống · tỉ lệ lấp đầy**.

Dùng học máy cổ điển (HOG + LBP + màu + kết cấu → Random Forest). Không cần GPU.

---

## Cài đặt

```bash
git clone https://github.com/AIVIETNAM-AIO-thanhnhan/car-parkinglot-count.git
cd car-parkinglot-count
pip install -r requirements.txt
```

Cần Python 3.9+. Trên Windows đặt `PYTHONIOENCODING=utf-8` trước khi chạy `python`.

---

## Chuẩn bị model

Repo **không kèm sẵn model** (file ~180 MB). Cần train một lần, tổng khoảng **1,5 giờ**:

```bash
bash scripts/download_data.sh raw                     # tải PKLot ~2 GB        (15-40 phút)
cd src
python pklot_data.py                                  # đọc nhãn, chia dữ liệu     (2 phút)
python build_dataset.py --axis-aligned \
       --out-dir ../features_axis                     # trích đặc trưng           (~36 phút)
python train_model.py --model rf --feat-dir ../features_axis \
       --save ../models/rf_axis.joblib                # train                      (~8 phút)
```

Bước trích đặc trưng chiếm gần hết thời gian: 1,18 triệu khung ảnh, mỗi khung 395 đặc trưng.
Chỉ phải chạy một lần — kết quả được cache lại.

> `--axis-aligned` là bắt buộc nếu muốn dùng chế độ **tự dò layout**. Bỏ cờ này thì ô đỗ được
> cắt theo góc xoay lấy từ nhãn — chỉ dùng được khi đã biết trước vị trí ô.

---

## Dùng

### Giao diện

```bash
streamlit run app/streamlit_app.py
```

Tải ảnh lên → ba con số, ảnh có vẽ khung (🟩 trống, 🟥 có xe), và nút tải kết quả dạng CSV.
Có thể ghi thêm lên từng khung: toạ độ, kích thước, độ tin cậy hoặc số thứ tự.

### Trong code

```python
import sys; sys.path.insert(0, "src")
import infer

bundle = infer.load_model("models/rf_axis.joblib")
img    = infer.load_image("bai_xe.jpg")
slots  = infer.slot_boxes_from_xml("bai_xe.xml")     # vị trí các ô đỗ

pred = infer.classify_slots(img, slots, bundle)
print(infer.count_from_predictions(pred))
# {'cars': 17, 'empty': 11, 'total': 28, 'occupancy_pct': 60.71428571428572}

infer.draw_boxes(img, pred).save("ket_qua.jpg")
```

Chưa có file XML? Để nó tự tìm vị trí ô từ nhiều ảnh cùng camera, rồi lưu lại dùng mãi:

```python
slots = infer.auto_layout([infer.load_image(p) for p in anh_cung_camera], bundle)
open("layout.xml", "w", encoding="utf-8").write(infer.slots_to_xml(slots, "bai_cua_toi"))
```

---

## Hai chế độ

| | Cần gì | Sai số chỗ trống | Tốc độ |
|---|---|---:|---|
| **Ô cố định** *(khuyên dùng)* | vị trí các ô đỗ | **0,4 ô** | 0,2 giây/ảnh |
| **Tự dò layout** | 5–20 ảnh cùng camera | **0,8 ô** | 8 giây/ảnh |

Camera bãi đỗ cố định, nên vị trí các ô chỉ cần xác định **một lần** rồi dùng mãi.

**Tự dò** làm việc đó tự động: đưa vào nhiều ảnh chụp khác thời điểm của cùng camera, nó gom dự
đoán qua các ảnh và giữ những vị trí xuất hiện lặp lại. Xe đổi chỗ giữa các ảnh nên ô bị che ở
ảnh này sẽ lộ ra ở ảnh khác. Đo trên 29 ảnh UFPR04: tìm ra **đúng 28 ô** (thực tế 28), và đếm
bằng layout tự dò cho sai số **0,8 ô** — gần bằng dùng layout thật (0,4 ô).

Sau khi dò, bấm **⬇️ Tải layout (.xml)** để dùng lại cho những lần sau mà không phải dò lại.

Cũng có thể khai báo tay: file XML định dạng PKLot, hoặc dán toạ độ JSON `[[x1,y1,x2,y2], ...]`.

> ⚠️ Tự dò cần model huấn luyện từ **ô cắt vuông góc** (`models/rf_axis.joblib`). Model cắt xoay
> thẳng (`rf.joblib`) không tự dò được — giao diện sẽ báo rõ nếu bạn chọn nhầm.
>
> Layout tự dò dùng khung **vuông**, bám đúng vị trí nhưng không ôm khít ô đỗ nghiêng
> (IoU trung vị 0,46). Đủ tốt để đếm, nhưng nên xem lại bằng mắt trước khi dùng lâu dài.

---

## Cách hoạt động

```
ảnh  →  cắt từng ô  →  395 đặc trưng mỗi ô  →  Random Forest  →  đếm
                       HOG · LBP · màu · kết cấu
```

Xe và chỗ trống có cùng hình chữ nhật, nên tín hiệu phân biệt nằm ở **màu và kết cấu** chứ không
phải hình dạng.

Chế độ "tự dò" thêm một bước phía trước: quét khung trượt ở 5 kích cỡ khắp ảnh (~18.000 khung),
lọc theo ngưỡng tin cậy rồi gộp các khung chồng nhau.

---

## Dữ liệu huấn luyện

579 ảnh (lọc từ 12.417 ảnh gốc, giữ 1 ảnh mỗi 2 giờ vì ảnh chụp 5 phút/lần gần như trùng nhau):

| Tập | Bãi | Ảnh |
|---|---|---:|
| **train** | UFPR04 + UFPR05 | 340 |
| **val** | UFPR04 — *khác ngày* với train | 29 |
| **test** | **PUCPR — bãi khác hoàn toàn** | 210 |

Chia **theo bãi đỗ**, không chia ngẫu nhiên. Camera cố định chụp suốt 30 ngày nên vị trí ô giống
hệt nhau ở mọi ảnh cùng bãi; chia ngẫu nhiên thì một chương trình *chỉ ghi nhớ vị trí ô* — không
nhìn một pixel nào — cũng đạt điểm cao. Kiểm bằng đúng phép thử đó:

| Cách chia | Điểm của "chỉ nhớ vị trí" |
|---|---:|
| Ngẫu nhiên, cùng ngày | 0.76 ← rò rỉ dữ liệu |
| Khác ngày, cùng bãi (= val) | 0.52 |
| **Bãi khác (= test)** | **0.00** ✅ |

---

## Độ chính xác

Đo trên **val** — bãi UFPR04, những ngày không dùng để huấn luyện, 28 ô đỗ mỗi ảnh:

| | Sai số số chỗ trống | Ghi chú |
|---|---:|---|
| **Ô cố định** (layout thật) | **0,41 ô** | 98,6% ô phân loại đúng |
| **Tự dò layout** | **0,76 ô** | tự tìm ra đúng 28/28 ô |

Cả hai đều dưới ngưỡng 3 ô — mức "tốt" theo tiêu chí đặt ra ban đầu của dự án.

### Khi đổi sang bãi hoàn toàn khác

Đo trên **test** (PUCPR — camera khác, độ cao khác, ô đỗ nhỏ hơn một nửa):

| | Cùng bãi (val) | Bãi khác (test) |
|---|---:|---:|
| **Phân loại** — đã biết vị trí ô | 98,6% | **96,5%** |
| **Tự dò** — tự tìm vị trí ô | 28 ô đúng | **hỏng** |

**Phân loại gần như không suy giảm** — phân biệt xe/chỗ trống dựa vào màu và kết cấu, khá phổ quát.

**Tự dò thì không dùng được ở bãi lạ.** Kích thước khung quét (96 px) được đo theo ô đỗ của UFPR;
ô của PUCPR chỉ 46×52 px, và "nền" mà model học là nhựa đường/cây cối của UFPR.

→ Với camera mới, hãy **khai báo vị trí ô một lần** thay vì dựa vào tự dò.

---

## Định dạng kết quả

```csv
image_id,x_min,y_min,x_max,y_max,label,score
bai_xe,120,80,165,110,1,0.91
```

`label`: **0 = ô trống, 1 = có xe**. `score` là độ tin cậy 0–1.

---

## Gặp lỗi?

| Lỗi | Cách xử lý |
|---|---|
| **Khung vẽ lên cây, lối đi, mặt đường** | vị trí ô đang lấy từ **bãi khác** với ảnh. Dùng đúng cặp ảnh + XML, hoặc chọn đúng tên ảnh trong danh sách |
| `UnicodeEncodeError` trên Windows | đặt `PYTHONIOENCODING=utf-8` |
| `ModuleNotFoundError: cv2` | `pip install opencv-python-headless scikit-image` |
| Giao diện báo "Chưa có model" | chưa chạy bước train ở trên |
| `Không thấy shard nào` | chưa chạy `build_dataset.py` |
| `thứ tự feature của model khác features.py` | model cũ không còn tương thích — train lại |
| Tự dò báo "Model này không tự dò được" | dùng `rf_axis.joblib`, xem mục "Hai chế độ" |

---

## Nguồn & giấy phép

Dữ liệu: [PKLot](https://web.inf.ufpr.br/vri/databases/parking-lot-database) — 12.417 ảnh, 3 bãi
đỗ, 3 điều kiện thời tiết. CC BY 4.0, Đại học Liên bang Paraná.
