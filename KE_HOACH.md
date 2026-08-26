# Đếm Chỗ Đỗ Xe (Có xe / Còn trống) — Kế hoạch gọn

| | |
|---|---|
| **Dataset** | PKLot — `web.inf.ufpr.br/vri/databases/parking-lot-database` (CC BY 4.0, tải tự do) |
| **Thời gian** | 23/08 → 07/09/2026 (16 ngày) |
| **Nhân sự** | 4 người |
| **Phần cứng** | Colab Free, **không cần GPU** |

---

## 1. OUTPUT — Dự án làm ra cái gì

```
Ảnh bãi đỗ  →  danh sách ô đỗ  →  3 con số
```

| Output | Ví dụ |
|---|---|
| Box + nhãn + độ tin cậy | `(120,80,165,110, "có xe", 0.91)` × N ô |
| **Số xe** | 42 |
| **Số chỗ trống** | 18 |
| **Tỉ lệ lấp đầy** | 70% |

Ba con số cuối là thứ hiển thị trên bảng điện tử đầu bãi. Đó là sản phẩm cuối.

---

## 2. CÁCH LÀM

```
Ảnh 1280×720
   ↓  cắt về vùng có nhãn          (P2)
   ↓  trượt cửa sổ 48px, bước 16   → ~1200 cửa sổ/ảnh   (P3)
   ↓  trích 395 feature/cửa sổ     HOG, LBP, màu, texture (P3)
   ↓  MODEL phân loại 3 lớp        nền / có xe / ô trống  (P4)
   ↓  lọc theo ngưỡng + NMS        ← code tự viết, không phải model
   ↓
box cuối cùng → đếm
```

Model chỉ làm bước "cửa sổ → xác suất". Ba bước sau là code tự viết.

---

## 3. MODEL — thứ tự thử

| Ngày | Model | Học ở | Vai trò |
|---|---|---|---|
| 5 | Decision Tree | W1 ✅ | Baseline nội bộ, đọc được luật |
| 6–9 | Random Forest (+ hard negative mining) | W2 ✅ | Model chính |
| 10 | LightGBM | W3 (28/08) | Nếu kịp học |
| 11 | Optuna | W3 (29/08) | Nếu kịp học |

> LightGBM và Optuna **không bắt buộc**. Dự án vẫn hoàn chỉnh nếu dừng ở Random Forest + mining.

---

## 4. BASELINE — hai cái, để biết model có thật sự tốt không

### Baseline A — "Đoán theo vị trí" (kiểm tra split có hợp lệ)

Không nhìn ảnh. Chỉ nhớ vị trí các ô từ tập train rồi lặp lại y nguyên.

Camera PKLot **cố định**, vị trí ô luôn giống nhau, xe đỗ lại hàng giờ. Nên baseline này rất mạnh nếu chia dữ liệu sai:

| Cách chia | mAP của Baseline A | Đo thật (23/08) |
|---|---|---|
| Random split cùng bãi (SAI) | **0.76** | — |
| **val = UFPR04, khác NGÀY với train** (cùng bãi) | cao — dự kiến | **0.5176** |
| **test = PUCPR, chia theo bãi** (ĐÚNG) | **0.00** | **0.0000** ✅ |

> Chạy `baselines.py` ở **Ngày 2**. Nếu điểm cao → split đang sai, phải chia lại theo bãi.
>
> 🔴 **Đã chạy Ngày 1. Kết quả cần hiểu đúng:**
> - **test (PUCPR) = 0.00** → split chính **hợp lệ**, không leak vị trí. Đây là điều kiện đã PASS.
> - **val (UFPR04) = 0.5176** → val **cố ý** cùng bãi với train (chỉ khác ngày, xem
>   `config.VAL_LOTS`) để có tập tinh chỉnh nhanh. Hệ quả: **một model chỉ học vị trí cũng đã
>   được ~0.52 trên val.** Vì vậy **0.5176 là SÀN của val**, không phải 0. Mọi điểm val phải
>   vượt rõ rệt con số này mới tính là model học được gì thật.
> - Ngưỡng "✅ Tốt mAP > 0.50" ở §5 chỉ có nghĩa khi đo trên **test**, không phải val.

### Baseline B — "Decision Tree"

Model đơn giản nhất. Mọi model sau phải hơn nó.

---

## 5. BẢNG KẾT QUẢ — thứ duy nhất cần điền

Tất cả các dòng đo trên **val** (UFPR04, khác ngày). Test chỉ mở đúng 1 lần ở Ngày 12.

| Model | mAP (val) | Sai số số chỗ trống | Kết luận |
|---|---|---|---|
| **Baseline A (đoán vị trí) — SÀN** | **0.5176** | **7.17** | 🔴 Mọi model phải vượt |
| Baseline B (Decision Tree) | | | Điểm xuất phát |
| Random Forest | | | Phải hơn B |
| RF + hard negative mining | | | Phải hơn RF |
| *LightGBM + Optuna (nếu kịp)* | | | |

**Đọc bảng:** nếu điểm tăng dần từ trên xuống → dự án thành công.

> ⚠️ Baseline A **không** phải dòng "phải THẤP" như dự kiến ban đầu — trên val nó là **sàn 0.5176**
> (vì val cùng bãi với train, xem §4). Dòng "phải THẤP" là **test = 0.00**, đã kiểm và PASS.
> Nguồn số: `results.csv`, dòng cuối cùng (`SỬA LỚN: … camera UFPR04 xê dịch 2 lần`) — **không**
> dùng các số cũ 0.1529 / 0.2122 ở những dòng phía trên, chúng sai do lỗi xê dịch camera đã sửa.

### Tiêu chí thành công

| Mức | Ngưỡng (val) | Ngưỡng (test — Ngày 12) |
|---|---|---|
| ❌ Thất bại | Không vượt sàn Baseline A **0.5176** | — |
| ⚠️ Tối thiểu | mAP > 0.60 (≈ sàn + 0.08) | mAP > 0.35 |
| ✅ Tốt | mAP > 0.70, sai số chỗ trống < 3 ô | mAP > 0.50, sai số chỗ trống < 3 ô |

---

## 6. PHÂN VAI

| | Vai trò | Sở hữu | File |
|---|---|---|---|
| **P1** | Tech Lead | Harness đánh giá, config, tích hợp, báo cáo | `evaluate_pklot.py`, `config.py` |
| **P2** | Data Engineer | Parse XML, cắt vùng, lấy mẫu thời gian, split, canh test set | `pklot_data.py`, `build_dataset.py` |
| **P3** | Pipeline Engineer | Cửa sổ trượt, feature, tốc độ, ablation | `windows.py`, `features.py` |
| **P4** | Model Engineer | DT → RF → mining → LightGBM | `train_model.py` |

> ⚠️ **P4 là điểm nghẽn** — gánh chuỗi model liên tục từ Ngày 5 đến Ngày 12. **P2 chuyển sang hỗ trợ P4 từ Ngày 6** (dữ liệu cho mining). P1 nên đọc hiểu code của P4 từ Ngày 7 để tiếp quản được nếu cần.

---

## 7. LỊCH 16 NGÀY

### Ngày 1–4 (23–26/08) — Nền tảng · **cả nhóm làm chung**

| Ngày | Việc |
|---|---|
| **1** | P2 tải PKLot, parse XML, **đo kích thước ô + độ méo phối cảnh**. P3 đo tốc độ trích feature |
| **2** | 🔴 **CỔNG KIỂM TRA**: harness PASS self-test. P2 cắt vùng, lấy mẫu thời gian, chia split theo bãi, **khóa**. Chạy Baseline A |
| **3** | Mỗi người trích feature 1 shard (~1 phút) → 8 file parquet trên Drive |
| **4** | P1 review, **KHÓA `config.py`** |

**🔴 Hai quyết định chốt Ngày 1 (P2 báo cáo, P1 duyệt):**
- `WINDOW_SIZE` — theo kích thước ô trung vị thật
- `SCALES` — nếu độ méo `p90/p10 > 2.0` thì phải dùng multi-scale

**Cổng Ngày 2:** nộp ground truth vào harness phải ra mAP = 1.0. FAIL → cả nhóm dừng sửa.

### Ngày 5 (27/08) — Baseline

P4 train Decision Tree, **in luật ra đọc**. P1 chạy hết đường → có con số đầu tiên (mAP ~0.10, đó là đúng).

**Buổi "nhìn ảnh" 30 phút, cả nhóm:** vẽ 5 ảnh có box, xem false positive rơi vào đâu (lối đi? vỉa hè?). Buổi này định hướng cả tuần sau.

### Ngày 6–9 (28–31/08) — Random Forest + mining · **4 người tách ra**

| Người | Việc |
|---|---|
| P1 | Gom kết quả, quét ngưỡng + NMS, viết báo cáo phần đầu |
| P2 | Hỗ trợ P4 (dữ liệu mining), canh test set |
| P3 | Tối ưu tốc độ (N6–7) → ablation feature (N8–9) |
| P4 | Random Forest (N6–7) → hard negative mining 3 vòng (N8–9) |

**Hard negative mining** (trục quan trọng nhất): train → chạy trên ảnh train → thu false positive → thêm vào tập nền → train lại. Lặp 3 vòng. Vẽ biểu đồ mAP qua từng vòng.

Kỳ vọng cuối Ngày 9: **mAP ≥ 0.42**

### Ngày 10–11 (01–02/09) — LightGBM + Optuna *(nếu kịp học)*

🔴 **CODE FREEZE cuối Ngày 11.** Sau mốc này chỉ sửa bug và viết báo cáo. Không thêm ý tưởng mới.

### Ngày 12–13 (03–04/09) — SHAP + mở test set

- SHAP: feature nhóm nào quan trọng nhất? Đối chiếu với bảng ablation của P3
- **Mở test set — một lần duy nhất.** Điểm sẽ tụt (val 0.55 → test 0.35). Đó là domain shift thật, không phải thất bại
- Phân loại lỗi: FP trên lối đi? FN ở ô xa camera?

### Ngày 14–16 (05–07/09) — Báo cáo

Mỗi người viết phần mình sở hữu, P1 ghép. Ngày 16 chỉ luyện thuyết trình, **không làm thêm tính năng**.

---

## 8. BỐN QUY TẮC KHÔNG ĐƯỢC VI PHẠM

1. **Harness viết xong và PASS trước khi ai train** (Ngày 2)
2. **Chia split theo bãi đỗ, khóa vĩnh viễn** — không bao giờ chia ngẫu nhiên
   *(ngoại lệ có chủ ý: **val** = UFPR04 cùng bãi với train nhưng tách theo NGÀY, để tinh chỉnh
   nhanh. Đổi lại val có sàn Baseline A = 0.5176 — xem §4. **test = PUCPR** vẫn khác bãi hoàn toàn,
   đây mới là phép đo thật.)*
3. **Test set chỉ mở Ngày 12, một lần**
4. **CODE FREEZE Ngày 11**

---

## 9. BỐN ĐẶC THÙ CỦA PKLOT PHẢI XỬ LÝ

| Vấn đề | Giải pháp | Ai |
|---|---|---|
| **Nhãn không đầy đủ** — PUCPR chỉ gán ~100/300 ô | Cắt ảnh về vùng có nhãn → nhãn trở nên đầy đủ | P2 |
| **Ảnh gần trùng** — chụp 5 phút/lần, camera cố định | Lấy mẫu thời gian: giữ 1 ảnh/2 giờ (156 → 7 ảnh/ngày) | P2 |
| **Méo phối cảnh** — ô gần to hơn ô xa | Cắt vùng (đã giảm), nếu vẫn `p90/p10 > 2.0` thì multi-scale | P2 |
| **Hai lớp cùng hình dạng** — ô có xe và ô trống đều là chữ nhật cùng cỡ | Tín hiệu nằm ở **màu và kết cấu**, không phải hình dạng. Dự đoán `color_*` quan trọng hơn HOG | P3 |

---

## 10. RỦI RO CHÍNH

| Rủi ro | Xử lý |
|---|---|
| **P4 quá tải hoặc vắng** | P2 hỗ trợ từ N6, P1 đọc code P4 từ N7 để tiếp quản |
| **Leakage do ảnh trùng** | Lấy mẫu thời gian + chia theo bãi. Kiểm bằng Baseline A ở N2 |
| **Méo phối cảnh quá lớn** | Bật multi-scale, hoặc chỉ dùng PUCPR (tầng 10, nhìn thẳng nhất) |
| **mAP < 0.30 ở Ngày 9** | Tăng tỉ lệ negative, giảm bước trượt, thêm vòng mining |
| **Chưa kịp học LightGBM** | Bỏ, giữ Random Forest làm model cuối |
| **Không kịp deadline** | Bỏ ablation và SHAP, giữ bảng kết quả 4 dòng |

---

## 11. BÁO CÁO — 8 mục

| Mục | Người viết |
|---|---|
| 1. Bài toán & output | P1 |
| 2. PKLot & 4 đặc thù | P2 |
| 3. Chống leakage + Baseline A | P2 |
| 4. Pipeline & feature | P3 |
| 5. Ablation feature | P3 |
| 6. Tiến hóa model DT → RF → mining | P4 |
| 7. SHAP & phân tích lỗi | P4 |
| 8. Test set, domain shift, hướng Module 4 | P1 |

---

## 12. BẢNG HẰNG SỐ

| Hằng số | Giá trị |
|---|---|
| `N_IMAGES` | **`None`** — đã bỏ cap 150, dùng TOÀN BỘ ảnh sau `temporal_subsample()`: **340 train / 29 val / 210 test** (579 ảnh) |
| `WINDOW_SIZE` | **96** — p50 kích thước ô (97, 64.5)px, đo trên UFPR04+UFPR05 (`pklot_data.slot_size_report()`) |
| `SCALES` | **`[0.5, 0.75, 1.0, 1.5, 2.0]`** — p90/p10 area = 3.3–3.45x (> 2.0) → multi-scale (`pklot_data.perspective_report()`) |
| `STRIDE` | 16 |
| `IOU_POSITIVE` / `IOU_IGNORE` | 0.5 / 0.3 |
| `NEG_SAMPLE_RATE` | 0.10 |
| `NMS_IOU` | 0.45 |
| `TEMPORAL_MINUTES` | 120 |
| `RANDOM_SEED` | 42 |

**Ngân sách thời gian:** tải PKLot 15–40 phút · trích feature 150 ảnh ~4 phút (chia 4 người → 1 phút/người) · Decision Tree 3 giây · Random Forest 2–5 phút · 1 vòng mining ~12 phút · Optuna 30 trials 30–60 phút.

---

## 13. NẾU CÒN THỜI GIAN (không bắt buộc)

| Mở rộng | Giá trị |
|---|---|
| **Nhánh B** — vị trí ô đã biết, chỉ phân loại từng ô | Lưới an toàn, ~1 giờ. Có baseline công bố để so (~89% khi đổi bãi) |
| Thí nghiệm domain shift theo thời tiết | Train ngày nắng → test ngày mưa |
| Chi phí lỗi bất đối xứng | Báo nhầm "còn chỗ" tệ hơn báo nhầm "hết chỗ" |
| AdaBoost cascade (Viola-Jones) | Tăng tốc 3–5× |

Những mục này viết vào phần "hướng phát triển" của báo cáo nếu không kịp làm.
