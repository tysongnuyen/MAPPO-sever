# Greedy placement + MAPPO allocation (attention only)

Chạy trong thư mục repo `MAPPO-sever`; toàn bộ dependency là file nằm cạnh launcher,
không dùng đường dẫn tuyệt đối của máy local. `env.py`, `ver2.py` và thuật toán
PPO/GAE dùng chung với các bản đã có. Các file gốc của repo không bị sửa.

## Chạy


python -u train_greedy_mappo.py \
  --device cuda \
  --output-dir training_outputs_greedy_mappo_attention_700_seed2027


```bash
conda activate myrl
bash run_greedy_mappo.sh --device cuda
```

CPU: đổi thành `--device cpu`. Interpreter tùy chọn:

```bash
BASELINE_PYTHON=/home/honganh/miniconda3/envs/myrl/bin/python bash run_greedy_mappo.sh --device cpu
```

Trên server dùng Python của môi trường server, không chép đường dẫn miniconda local.
Xem cấu hình trước khi train: `python train_greedy_mappo.py --print-config`.
Chạy kiểm tra: `python -m unittest -v test_greedy_mappo`.

```bash
python -u train_greedy_mappo.py \
  --resume training_outputs_greedy_mappo_attention_700_seed2027/checkpoints/latest_resume.pt
```

Resume tự đọc cấu hình đã lưu, kiểm tra source/config, khôi phục optimizer và RNG,
cắt phần log phát sinh sau checkpoint. SHA-256 kiểm tra cả nội dung log trước mốc
checkpoint; chỉ phần hậu tố sau mốc được cắt. Khi quay về checkpoint cũ, `latest_resume.pt`
được đưa về cùng mốc trước khi cắt log, các checkpoint/kết quả tương lai được lưu vào
thư mục archive. Không ghép checkpoint của run này vào output của run khác.
Không resume training từ best checkpoint. Kiểm tra resume chính xác đã thực hiện trên CPU;
không suy ra tính đồng nhất bit khi đổi thiết bị hoặc phiên bản thư viện.
`--stop-after-global N` dừng sạch ở mốc N mà giữ nguyên tổng ngân sách để kiểm tra resume.
Run mới yêu cầu output rỗng. Không tự bắt đầu train đầy đủ khi chạy unit tests.

## Cấu hình mặc định

- Theo lịch server: **710 global = 10 global đầu + 700 global tiếp theo**, seed train **2027**.
  Baseline này chỉ học allocation trong cả 710 global; greedy không cần warm-up riêng.
  `--placement-episodes` là tên CLI kế thừa cho **tổng số global**, không phải số map.
- 2 map/global từ pool `3000..4619` được shuffle bằng seed train: tổng **1420 map khác nhau**.
- Mỗi map có 5 hoán vị cyclic của UAV ID. Hình học greedy giống nhau; gán vị trí cho ID khác nhau.
  Một inner update có 10 episode đầy đủ = **500 transition** khi train 50 UE.
- Tối đa 20 inner update/global, patience 10 lần liên tiếp không tăng mean reward
  deterministic trên 10 bài toán cố định của global. Luôn giữ model ở update cuối;
  best inner chỉ dùng đếm patience, **không rollback**.
- 5 PPO epochs, actor minibatch 500; critic dùng chunk/minibatch 64 của implementation gốc.
  LR actor `2e-4`, critic `3e-4`; gamma `1`, GAE lambda `0.95`; entropy `0.003` cố định;
  clip `0.2`, gradient norm tối đa `0.5`, target KL `0.02` riêng từng actor trong một PPO update.
- Attention embed 64, 4 heads; input local và hierarchical participation/PRB/power giữ nguyên.
  Không COMA, không local residual bypass bổ sung, không PopArt, không bảng chi phí cho critic.
  Các residual/LayerNorm bên trong attention encoder vẫn là cấu trúc encoder thông thường.
- Giữ quy ước entropy của project: entropy có giá trị bằng joint entropy, nhưng detach
  trọng số nhánh khi truyền gradient về head cha. Đây không phải gradient joint-entropy
  nguyên bản. "Attention-only" ở đây mô tả các phần bổ sung bị bỏ, không có nghĩa là
  tái hiện mọi chi tiết implementation MAPPO chính thức. Reward vẫn chia cho số UE lúc train
  khi tính GAE/critic, advantage vẫn chuẩn hóa theo rollout; các bước này không phải PopArt.
- Reward `1 - 0.2*P_winner/2 - n_winner/135` nếu có winner; còn lại 0.
  135 PRB chung, 2 W/UAV; reject + 100 cặp (1..10 PRB × 0.1..1.0 W).
  Actor mask chỉ xét coverage. Env vẫn kiểm tra QoS/tài nguyên và phân xử bid như cũ.
- Validation `214..223` mỗi 20 global; diagnostic `224..273`, test `434..483` sau training,
  sử dụng best validation. Không có K-means + greedy allocation.
- Device auto, 4 torch threads theo run tham chiếu; có thể chọn CPU/CUDA theo máy.
  `--num-ues` mặc định 50. Nếu test checkpoint ở số UE khác, giữ reward_scale lúc train.

Các hệ số lấy từ run `training_outputs_none_costcritic_2maps_5orders_10_400_200_seed2026_ent003_pat10_11dim_actor`;
lịch 710 và seed/splits lấy từ bản MAPPO-sever mới. GAE lambda kế thừa baseline attention (`0.95`).
Không còn giai đoạn placement-only/freeze allocation. `placement-batch-episodes` chỉ giữ
tương thích kiểm tra lịch server; placement không có rollout, loss hoặc optimizer update.

## Greedy tính gì?

1. Xét mọi candidate `(x,y,h,theta)` của env, không K-means rồi ánh xạ grid.
2. Với mỗi candidate/UE, tính cost nhỏ nhất trong lưới PRB/power đạt QoS và coverage.
   Gain dùng `env.gain` (khoảng cách 3D, antenna gain); rate dùng công thức Shannon của env.
3. Chọn tuần tự UAV: tối đa số UE **mới** có nghiệm QoS; khi hòa, tối thiểu tổng chi phí
   tốt nhất cho hợp UE đang phủ (bao gồm cải thiện cost của UE đã phủ); cuối cùng chọn index nhỏ nhất.
4. Không trùng xyz, kể cả khác góc; có thể trùng xy nếu khác h, đúng action space của env.

Đây là ước lượng khả năng phục vụ **từng UE riêng lẻ**, không phải số UE chắc chắn phục vụ
đồng thời dưới quỹ PRB/power. Bảng greedy chỉ dùng tại placement, không đưa vào actor/critic,
không dùng để lọc allocation action. Chưa phải tái hiện nguyên bản một paper cụ thể.

## Output và đánh giá

`allocation_metrics.csv`: mọi episode allocation theo đúng thứ tự lấy mẫu.
`learning_diagnostics.jsonl`: mọi PPO update; `inner_evaluations.jsonl`: cả inner 0 và mọi lần đánh giá.
`training_metrics.csv`: mỗi global, placement thực dùng và tài nguyên/return tổng hợp.
`validation_metrics.csv`, `test_metrics.csv`, `diagnostic_metrics.csv`: từng map.
`allocation_training.png`: toàn bộ reward và moving mean 1000 mẫu; `validation.png`: từng map và mean.
`training_config.json`: hyperparameter, hash source, runtime. `training_budget_summary.json`: transition và optimizer thực tế.

Checkpoint có framework riêng để tránh vô tình dùng learned-placement checkpoint.
Do tái sử dụng serializer, checkpoint vẫn chứa module placement khởi tạo và đóng băng;
chúng không được gọi để ra quyết định và không được train. `active_allocation_parameters`
chỉ đếm các tham số thực sự học. Khi so sánh công bằng, đối chiếu **transition thực**;
cùng 710 global không đảm bảo số update giống nhau do early stopping.

Để đánh giá bằng code: tạo env, `GreedyAttentionMAPPO.from_env(env, ...)`, load checkpoint,
rồi gọi `evaluate_one_episode(env, model, ue_seed=...)` từ `strict_local_independent_mappo`.
Phải dùng chính env đã bind vào model; các tham số embed/heads/GAE/reward_scale phải khớp lúc train.

Baseline này so sánh **greedy placement + allocator MAPPO** với phương pháp toàn hệ.
Nó không tự tách được đóng góp riêng của COMA nếu phương pháp chính đồng thời khác
placement, nhánh residual và đầu vào critic. Muốn tách placement, đối chiếu với bản
learned-placement + cùng attention-only allocator. Greedy có mô hình vật lý để chấm
candidate lúc triển khai; cần tính cả thời gian planner khi báo cáo inference end-to-end.
