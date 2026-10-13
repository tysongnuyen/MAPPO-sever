# MAPPO server training

Mặc định: **710 global episode**. Episode 1–10 chỉ warm-up allocation;
episode 11–710 train cả placement và allocation (700 episode), không có pha
freeze allocation cuối. Placement PPO vẫn tích lũy 10 global episode trước
mỗi lần update; allocation vẫn tối ưu trên các placement cố định trong vòng lặp bên trong.

Seed huấn luyện: `2027`. Bộ seed bản đồ train: `3000–4619` (1620 seed).
Lịch mặc định dùng 1420 seed khác nhau sau khi xáo trộn, 2 bản đồ/global.
Validation: `214–223`; diagnostic: `224–273`; test: `434–483`.

## Cài đặt và chạy

Dùng Python 3.12. Trong thư mục này, tạo môi trường riêng và cài các phiên bản
thư viện trong `requirements.txt` (đã kiểm tra bằng môi trường CPU cục bộ):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python test_training_schedule.py
```

Với server GPU, PyTorch cần là bản hỗ trợ CUDA tương thích với driver của server.
Kiểm tra trước khi train:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA is not available'; print(torch.cuda.get_device_name(0))"
```

Chạy trong phiên terminal bền vững của server (ví dụ `tmux`):

```bash
python -u train_strict_local_mappo.py --device cuda --output-dir training_outputs_joint700_seed2027
```

Nếu chạy CPU, thay `--device cuda` bằng `--device cpu`. Thư mục output phải
rỗng hoặc chưa tồn tại khi bắt đầu run mới. Lịch và seed mới dành cho run mới.

Nếu muốn 600 episode train cả hai, dùng `--placement-episodes 610` (gồm 10 warm-up).
`--placement-only-episodes` mặc định là `0`.

`test_training_schedule.py` kiểm tra lịch mặc định, seed tách biệt và chạy thử CPU
3 global episode (1 warm-up + 2 joint), so sánh trọng số và optimizer trong
checkpoint để xác nhận cả hai vẫn cập nhật ở cuối. Đây không phải run đủ 710
episode hay kiểm tra CUDA trên server.
