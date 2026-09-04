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

Repo **không kèm sẵn model** (file 53 MB). Cần train một lần, tổng khoảng **2 giờ**:

```bash
bash scripts/download_data.sh raw                             # tải PKLot ~2 GB   (15-40 phút)
cd src
python pklot_data.py                                          # đọc nhãn, chia dữ liệu  (2 phút)
python build_dataset.py --shard 0 --n-shards 1                # trích đặc trưng   (~70 phút)
python train_model.py --model rf --save ../models/rf.joblib   # train             (~6 phút)
```

Bước trích đặc trưng chiếm gần hết thời gian: 1,18 triệu khung ảnh, mỗi khung 395 đặc trưng.
Chỉ phải chạy một lần — kết quả được cache lại.

> Muốn nhanh hơn: `--n-shards 8` chia bước này ra 8 phần chạy song song trên nhiều máy
> (mỗi máy một `--shard 0..7`), còn ~9 phút mỗi máy.

---

## Dùng

### Giao diện

```bash
streamlit run app/streamlit_app.py
```

Tải ảnh lên → ba con số, ảnh có vẽ khung (🟩 trống, 🟥 có xe), và nút tải kết quả dạng CSV.

### Trong code

```python
import sys; sys.path.insert(0, "src")
import infer

bundle = infer.load_model("models/rf.joblib")
img    = infer.load_image("bai_xe.jpg")
slots  = infer.slot_boxes_from_xml("bai_xe.xml")     # vị trí các ô đỗ

pred = infer.classify_slots(img, slots, bundle)
print(infer.count_from_predictions(pred))
# {'cars': 17, 'empty': 11, 'total': 28, 'occupancy_pct': 60.71428571428572}

infer.draw_boxes(img, pred).save("ket_qua.jpg")
```

---

## Hai chế độ

| | Cần gì | Sai số | Tốc độ |
|---|---|---|---|
| **Ô cố định** *(khuyên dùng)* | vị trí các ô đỗ | **0,3 ô** | 0,2 giây |
| **Tự dò** *(thử nghiệm)* | chỉ cần ảnh | 18 ô | 8 giây |

Camera bãi đỗ thường cố định, nên vị trí các ô chỉ cần khai báo **một lần** rồi dùng mãi — đó là
lý do chế độ "ô cố định" vừa nhanh vừa chính xác hơn hẳn. Khai báo bằng một trong ba cách: file
XML định dạng PKLot, dán toạ độ JSON `[[x1,y1,x2,y2], ...]`, hoặc vẽ trực tiếp trên giao diện.

> ⚠️ **Chế độ "tự dò" chưa dùng được cho việc thật** — nó bỏ sót phần lớn ô đỗ. Model được huấn
> luyện trên các ô đã xoay thẳng theo nhãn, còn khi tự dò thì khung quét luôn vuông góc. Đo trên
> 84 ô: nhận đúng 81% với ô xoay thẳng, chỉ 3,6% với khung vuông góc.

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

## Độ chính xác

Đo trên 10 ảnh kiểm định — bãi UFPR04, những ngày không dùng để huấn luyện:

| | Sai số số chỗ trống | Sai số tỉ lệ lấp đầy |
|---|---:|---:|
| **Ô cố định** | **0,3 ô** | 1,1 điểm % |
| Tự dò | 18 ô | 22 điểm % |

Với chế độ ô cố định, 9/10 ảnh cho kết quả đúng tuyệt đối.

Model **chưa được kiểm trên bãi đỗ khác**. Kỳ vọng độ chính xác giảm khi đổi bãi, đổi góc camera
hoặc gặp thời tiết khác dữ liệu huấn luyện.

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
| `UnicodeEncodeError` trên Windows | đặt `PYTHONIOENCODING=utf-8` |
| `ModuleNotFoundError: cv2` | `pip install opencv-python-headless scikit-image` |
| Giao diện báo "Chưa có model" | chưa chạy bước train ở trên |
| `Không thấy shard nào` | chưa chạy `build_dataset.py` |
| `thứ tự feature của model khác features.py` | model cũ không còn tương thích — train lại |
| Chế độ tự dò gần như không thấy ô nào | hạn chế đã biết, xem mục "Hai chế độ" |

---

## Nguồn & giấy phép

Dữ liệu: [PKLot](https://web.inf.ufpr.br/vri/databases/parking-lot-database) — 12.417 ảnh, 3 bãi
đỗ, 3 điều kiện thời tiết. CC BY 4.0, Đại học Liên bang Paraná.
