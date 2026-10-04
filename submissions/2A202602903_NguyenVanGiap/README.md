# DeepWeeds Lab Day 2 — chạy trên Google Colab

Bài làm nằm trong `code/`; `starter/` là khung gốc của giảng viên. Không sửa `eval.py`.

## Cách chạy

1. Mở [Google Colab](https://colab.research.google.com/), chọn **Upload notebook**, upload `code/lab_day2.ipynb`.
2. Chọn **Runtime → Change runtime type → T4 GPU** (hoặc GPU khác).
3. Chạy từ trên xuống. Ở ô đầu, upload **`colab_bundle.zip` trong thư mục gốc repo**. ZIP chứa mã mới nhất, không chứa dataset, checkpoint hay môi trường local. Nếu đã push mã lên GitHub, có thể dùng `SOURCE="github"`.
4. Cho phép Colab mount Google Drive. Kết quả lưu ở `MyDrive/DeepWeedsLab/day2_full_batch32/`.
5. Giữ `MODE="full"` để chạy lab đầy đủ. `MODE="smoke"` dùng 2 epoch kiểm tra luồng và bỏ qua test; không dùng kết quả smoke để nộp.
6. Sau khi hoàn tất, notebook tải `day2_results.zip`, gồm Excel, report, code, predictions, curves, thông tin cấu hình/môi trường và kết quả `eval.py`.

[Link Colab từ GitHub](https://colab.research.google.com/github/Giappp/K4-DAY02-NguyenVanGiap-2A202602903/blob/main/submissions/2A202602903_NguyenVanGiap/code/lab_day2.ipynb) chỉ sử dụng được **sau khi đẩy notebook/mã lên đúng branch**. Cách upload ở trên chạy ngay với file đã chuẩn bị, không cần push.

## Cấu hình và khôi phục

- Toàn bộ compute của lab chạy trên Colab. Không cần chạy `uv sync`, `train.py` hay Jupyter trên máy cá nhân.
- Cài thư viện bằng `requirements-colab.txt`, giữ nguyên cặp **CUDA torch/torchvision của Colab**. `pyproject.toml`/`uv.lock` là môi trường local có sẵn trước đó; notebook không sử dụng chúng.
- Full: 12 epoch, ảnh 224, batch 32, AdamW, warmup 1 epoch, AMP. Batch 32 giảm từ 64 của công thức gốc để phù hợp VRAM; mọi backbone dùng cùng công thức và thay đổi được ghi trong config.
- Nếu OOM: chọn batch 16, dùng `SESSION_NAME` mới và chạy lại các thí nghiệm với cùng batch. Không đổi batch giữa các backbone của cùng phiên so sánh.
- Colab ngắt phiên: mở notebook, chạy lại các ô với cùng `MODE`, `BATCH_SIZE`, `SESSION_NAME`. Dữ liệu archive được cache trên Drive; ảnh giải nén đọc từ `/content`.
- `runs/<exp_id>/seed<k>/latest.pt` khôi phục model, optimizer, scheduler, scaler, EMA, lịch sử và trạng thái RNG từ cuối epoch. Epoch đang chạy khi bị ngắt sẽ chạy lại.
- Lần chạy hoàn tất dùng lại `summary.json`. Test CSV/logit đã lưu được dùng để khôi phục summary mà không đánh giá test lại. Ngắt giữa forward test trước khi lưu thành công có thể cần thực hiện lại phần chưa lưu; không được dùng lần chạy lỗi để chọn cấu hình.
- Config khác với run đã có sẽ báo lỗi; tạo tên phiên mới để không trộn kết quả.

## Thực nghiệm và sản phẩm

- 5 backbone: ResNet-50, ResNeXt-50, ConvNeXt-Tiny, DeiT-Small, MobileNetV3-Large.
- 3 trục: khởi tạo finetune/frozen, augmentation basic/color, loss CE/label smoothing/focal; thêm cấu hình kết hợp.
- Suy luận: một view, TTA flip trung bình probability/logit, calibration, gộp BN, AMP. Chỉ chọn trên validation. Phương pháp chung kết hỗ trợ identity, hflip_prob, hflip_logit, temperature.
- Mốc T00 và chung kết F01 dùng 3 seed (0/1/2); T00 dùng backbone đã chọn với công thức nền. Temperature khớp trên validation của từng seed.
- Độ trễ: warmup ≥10, 100 lần đo, CUDA synchronize, batch 1; không tính đọc/tiền xử lý ảnh. GMAC dùng fvcore, có thể thiếu operation chưa hỗ trợ; xem cảnh báo trong output.
- `results.xlsx`: Backbones, Training, Inference, Final, PerClass, Latency, Summary.
- Report tự động dùng số đo thật; xem ảnh lỗi/confusion matrix để bổ sung giả thuyết và nhận xét của bạn. Không có kết quả huấn luyện được điền trước.
- `environment-pip.txt`, `config.json`, `environment.json`, `history.csv` ghi phiên bản thực tế, tag trọng số và dấu vết mỗi lần chạy. Checkpoint giữ trên Drive, loại khỏi ZIP nộp bài.

## Kiểm tra trên Colab

Notebook chạy test bài làm và test gốc trong hai process riêng. Test gốc cố ý yêu cầu `starter/` còn stub, nên không thay khung đó.

Test bài làm kiểm tra focal gamma=0, label smoothing=0, CutMix theo diện tích thực, BN fusion giữ output và không gộp nhánh sai, EMA, tham số optimizer, calibration, scheduler và parsing cấu hình.

Tài liệu API: [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html), [timm](https://huggingface.co/docs/timm/en/models).
