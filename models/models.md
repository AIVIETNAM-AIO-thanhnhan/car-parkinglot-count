# Model đã train

## Thư mục Drive (nhánh main, trước merge 05/09)

https://drive.google.com/drive/u/0/folders/1ivznBvMAa1vFePhgBizQHu7HstSIy-Fw

> **⚠️ KHÔNG dùng trực tiếp cho UI.** Các file `.pkl` ở đây là `joblib.dump(clf)` trần, sinh ra
> từ `train_model.py` bản main, và mang **bảng nhãn cũ `0=nền, 1=ô trống, 2=có xe`** — ngược
> với quy ước toàn dự án (`0=ô trống, 1=có xe, 2=nền`).
>
> Nạp thẳng vào `detect.py` / `infer.py` / UI sẽ làm **đảo lớp im lặng**: vùng nền bị xuất ra
> thành "ô trống" và toàn bộ xe bị vứt. Không có lỗi nào được báo.
>
> Chúng cũng thiếu `feature_cols` và `label_names`, nên `train_model.load_model()` sẽ **từ chối**
> chúng — đó là hành vi đúng, đừng gỡ kiểm tra đó ra để "cho nó chạy".

## Bundle dùng được cho UI

Sinh bằng `--save`, `save_model()` đóng gói kèm thứ tự cột + bảng nhãn + cấu hình hậu xử lý:

```bash
cd src
python train_model.py --model rf --preset optuna --save ../models/rf_optuna.joblib
```

`models/*.joblib` không được commit (xem `.gitignore`) — sinh lại tại chỗ hoặc lấy từ Drive khi
đã có bundle đúng chuẩn.
