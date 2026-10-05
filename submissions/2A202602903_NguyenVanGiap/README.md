# DeepWeeds Lab Day 2 — Nguyễn Văn Giáp · 2A202602903

Bài làm hoàn thiện theo **21 ô của `starter/lab_day2.ipynb`**, chạy trên **Kaggle**. Hai file [`lab_day2.ipynb`](lab_day2.ipynb) và [`code/lab_day2.ipynb`](code/lab_day2.ipynb) có cùng nội dung. Notebook đã chứa mã nguồn; chỉ cần import một file notebook, không cần upload ZIP riêng hoặc clone repo. `starter/` và `eval.py` gốc được giữ nguyên.

## Chạy trên Kaggle

1. Mở [Kaggle Code](https://www.kaggle.com/code), tạo notebook và chọn **File → Import Notebook** để import `lab_day2.ipynb`.
2. Trong **Settings / Session options**, chọn **Accelerator → GPU** và **Internet → On**. Pipeline dùng GPU 0, kể cả khi máy có hai GPU.
3. Chạy từ trên xuống. Giữ `MODE="full"`, `BATCH_SIZE=32` ở ô tải dữ liệu để chạy đầy đủ. `MODE="smoke"` dùng 2 epoch, vẫn đi qua các bước sàng lọc nhưng **bỏ chung kết/test**; không dùng số liệu smoke để nộp bài.
4. Chọn **Save Version → Save & Run All** cho phiên chạy đầy đủ. Kết quả được ghi vào `/kaggle/working/day2_full_batch32/`. Ảnh và mã setup nằm ở `/kaggle/temp` để tránh đưa dataset vào output.
5. Ô cuối tạo `/kaggle/working/day2_full_batch32_results.zip`. Tải ZIP qua link trong ô hoặc tab Output của phiên đã lưu. ZIP gồm Excel, report, notebook và code, predictions, curves, nhãn CSV, `eval.py` và log cấu hình; checkpoint nằm riêng trong `runs/` của output.

Cách cấu hình GPU và output theo [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks). Link notebook Kaggle công khai cần bổ sung sau khi bạn lưu phiên trên tài khoản Kaggle; repository chưa có phiên Kaggle đã chạy.

## Dữ liệu và cấu hình

- Dùng DeepWeeds, CSV **fold 0 nguyên bản** từ GitHub tác giả; không chia lại, không gộp val vào train. Ảnh tải từ Zenodo và kiểm MD5 `b7b30f96d466fba86016aa5a26606e0f`.
- Nếu đã gắn dataset chứa **`images.zip` nguyên bản**, đặt `IMAGES_ARCHIVE="/kaggle/input/<dataset>/images.zip"`. Notebook đọc archive mà không sửa Input, kiểm checksum rồi giải nén vào `/kaggle/temp/deepweeds-data`. CSV và trọng số pretrained vẫn cần Internet.
- Full: 12 epoch, ảnh 224, AdamW, warmup 1 epoch, AMP. Batch 32 giảm từ batch 64 tham khảo để vừa VRAM; mọi backbone dùng cùng batch và cấu hình được lưu. Nếu OOM, dùng batch 16 và tên phiên mới cho toàn bộ phép so sánh.
- Seed sàng lọc: 0. Chung kết và mốc: **0/1/2**. Checkpoint chọn theo macro-F1 val; test chỉ dùng sau khi đã lưu quyết định trong `selection.json`.
- Cài `requirements-kaggle.txt`, giữ cặp CUDA torch/torchvision có sẵn trên Kaggle. Phiên bản thực tế ghi trong `environment-pip.txt` và `runs/*/seed*/environment.json`. `pyproject.toml`/`uv.lock` là môi trường local, notebook không sử dụng.

## Tiếp tục phiên

- Chạy lại trong cùng phiên với cùng cấu hình: tự dùng `latest.pt` hoặc `summary.json`, không huấn luyện lại các run đã xong.
- Sang phiên mới: lưu output phiên trước, gắn output đó qua **Add Input**, đặt `RESUME_DIR` đến thư mục có `runs/`, ví dụ `/kaggle/input/<previous-notebook>/day2_full_batch32`. Giữ nguyên `MODE`, `BATCH_SIZE` và `SESSION_NAME`. Notebook chép output vào thư mục ghi được rồi khôi phục.
- `latest.pt` lưu model, optimizer, scheduler, scaler, EMA, lịch sử và RNG ở cuối epoch. Epoch bị ngắt giữa chừng sẽ chạy lại. ZIP nộp bài không chứa checkpoint; muốn tiếp tục huấn luyện cần **output đầy đủ**, gồm `.pt`.
- Test CSV/logit đã lưu được dùng lại để hoàn tất summary; không forward test lại nếu đã có dự đoán. Nếu ngắt trước khi ghi dự đoán thành công, phần chưa lưu phải thực hiện tiếp.
- Config khác run cũ sẽ báo lỗi. Dùng tên phiên mới để thử cấu hình mới. `/kaggle/working` chỉ được dùng để tiếp tục sau khi output đã được lưu và gắn vào phiên mới; không coi đó là Drive tự đồng bộ.

## Chuyển phiên Colab sang Kaggle khi hết GPU

1. Trên Google Drive, tải **toàn bộ `day2_full_batch32/`**, gồm `runs/` (giữ `.pt`, `.npz`, `.json`, `.csv`), `predictions/`, `curves/`, `selection.json`, `inference_results.json` và file môi trường nếu có. ZIP sản phẩm ở ô cuối đã loại checkpoint nên không đủ để tiếp tục train. Chỉ ảnh thư mục `runs/` chưa cho biết seed nào đã hoàn tất.
2. Tạo dataset riêng trên Kaggle từ thư mục/ZIP này, gắn vào notebook bằng **Add Input**. Import lại notebook Kaggle mới nhất đã hỗ trợ chuyển đường dẫn Colab.
3. Giữ `MODE="full"`, `BATCH_SIZE=32`, `SESSION_NAME="day2_full_batch32"` và đặt `RESUME_DIR` đến thư mục chứa `runs/` hoặc file ZIP. Dùng đường dẫn thực tế hiển thị trong Input, ví dụ:

   ```python
   RESUME_DIR = "/kaggle/input/<dataset>/day2_full_batch32"
   # Hoặc, nếu Input giữ file ZIP:
   # RESUME_DIR = "/kaggle/input/<dataset>/day2_full_batch32.zip"
   ```

4. Chạy setup/tải dữ liệu. Notebook chép output sang `/kaggle/working`, đổi **chỉ đường dẫn** trong config/summary/selection/cache suy luận, giữ nguyên siêu tham số, seed và quyết định đã chốt. Bản metadata ban đầu lưu ở `migration_originals/`; phần cứng/phiên bản Colab được ghi trong `migration.json`. Không sửa Input.
5. Chạy các ô **Bước 1 → Bước 2 → Bước 3** để khôi phục biến và đọc kết quả đã hoàn tất. Nếu các run đó có `summary.json`, chúng được đọc lại, không train lại. Cache suy luận và quyết định đã chốt được giữ nguyên. Nếu file chưa hoàn tất/mất, ô tương ứng phải chạy phần còn thiếu.
6. Chạy **Bước 4** để hoàn tất T00/F01 với seed 0/1/2, rồi chạy score, grade và **Bước 5**. Có `latest.pt` thì khôi phục cuối epoch; có test CSV/logit đã lưu thì dùng lại. Bảng trạng thái sau setup ghi `Train hoàn tất`, `Test hoàn tất` hoặc `Chưa hoàn tất` cho từng seed.

Độ trễ cũ vẫn gắn với GPU Colab đã đo; độ trễ các run thực hiện trên Kaggle gắn với GPU Kaggle. Khi báo cáo, giữ thông tin phần cứng theo từng lần đo. Việc đổi GPU có thể làm kết quả sau khôi phục khác nhỏ so với tiếp tục trên cùng GPU.

## Các bước theo starter

| Ô trong template | Nội dung đã hoàn thiện |
|---|---|
| Setup và tải dữ liệu | Mã nguồn nhúng, cài thư viện, GPU, checksum, fold 0, cấu hình |
| Bước 0 | Đếm tập/lớp, giao rỗng, đủ 17.509 ảnh, kiểm file ảnh, biểu đồ và 3 ảnh/lớp; loss ban đầu, overfit một batch, ảnh augmentation |
| Bước 1 | ResNet-50, ResNeXt-50, ConvNeXt-Tiny, DeiT-Small, MobileNetV3-Large; params, GMAC, metric val, thời gian và latency |
| Bước 2 | Ba trục finetune/frozen, basic/color, CE/label smoothing/focal; T05 kết hợp các yếu tố tốt trên val |
| Bước 3 | I00 một view; TTA flip gộp probability/logit, temperature scaling, gộp BN, AMP; đo p50/p95/p99 |
| Bước 4 | T00 và F01, 3 seed; suy luận chung kết hỗ trợ identity/hflip_prob/hflip_logit/temperature; `eval.py score` và `grade` |
| Bước 5 | 7 sheet Excel: Backbones, Training, Inference, Final, PerClass, Latency, Summary; biểu đồ, report và ZIP |

Độ trễ đo batch 1, 10 warmup, 100 lần, CUDA synchronize; không gồm đọc/tiền xử lý ảnh. GMAC dùng fvcore và có thể bỏ qua operation chưa hỗ trợ; xem cảnh báo trong output. Báo cáo tự động lấy số đo thật; đọc ảnh lỗi và bổ sung phân tích của bạn trước khi nộp.

Notebook chạy test bài làm và test gốc trong hai process riêng. Test gốc yêu cầu `starter/` còn khung, nên không sửa template. Test bài làm kiểm tra focal, label smoothing, CutMix, BN fusion, EMA, optimizer, calibration, scheduler và setup Kaggle.

## Cập nhật mã nguồn

Sau khi sửa `code/*.py`, tài liệu hoặc test, chạy lệnh nhẹ sau để nhúng mã mới vào cả hai notebook (không train, không cần torch):

```bash
python submissions/2A202602903_NguyenVanGiap/build_kaggle_notebook.py
```

[`HUONG_DAN_CODE.md`](HUONG_DAN_CODE.md) giải thích từng module. Các file `requirements-colab.txt` và `code/colab_setup.py` thuộc bản Colab trước đây; luồng Kaggle dùng `build_kaggle_notebook.py`, `requirements-kaggle.txt` và `code/kaggle_setup.py`.
