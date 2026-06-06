# pi0.5 UMI 微调指南

## 概述

在 UMI 数据上微调 pi0.5，采用「冻结 VLM + 训练 ViT + 6D 旋转 + geodesic loss」策略。

| 项目 | 说明 |
|------|------|
| 模型 | pi0.5 (3.6B params) |
| 框架 | PyTorch (JAX 版本在 JIT 编译阶段会 OOM) |
| 单卡显存 | >24GB（冻结 VLM 后 ~14-16GB with batch=8） |
| 多卡 | DDP 即可，不需要 FSDP |

---

## 1. Action 格式

每个机器人的 action 为 10 维，在 LeRobot 数据集中按如下顺序存储：

```
[pos_x, pos_y, pos_z,  rot6d_0..rot6d_5,  gripper]
 └─── 3 dims ────┘  └──── 6 dims ──────┘  └─ 1 dim ─┘
```

| 分量 | 维度 | 含义 | 类型 |
|------|------|------|------|
| position | `[0:3]` | 末端位姿的 x, y, z | **相对位移**（delta） |
| rotation | `[3:9]` | 6D 连续旋转 | 旋转矩阵前两列（6 个数），**相对旋转** |
| gripper | `[9:10]` | 夹爪开合 | 绝对位置，归一化到 [0, 1] 或 [-1, 1] |

**多机器人**：如果是双臂 UMI（2 robots），action = 20 dims，排列为 `[robot0(10), robot1(10)]`。

**为什么用 6D 旋转**：参考 Zhou et al., "On the Continuity of Rotation Representations in Neural Networks" (CVPR 2019)。相比四元数和欧拉角，6D 表示在神经网络中是连续的，没有万向锁问题，比 3×3 矩阵省维度。

### 6D ↔ 旋转矩阵 转换

```
6D 定义：[a1, a2] = 旋转矩阵的前两列（展开为 6 个标量）
恢复算法（Gram-Schmidt）：
    b1 = normalize(a1)
    b2 = a2 - (b1·a2)·b1     # 正交化
    b2 = normalize(b2)
    b3 = b1 × b2              # 叉积得第三列
    R = [b1, b2, b3]          # 3×3 标准正交矩阵
```

实现位置：`src/openpi/shared/rotation_utils.py`

---

## 2. Loss 计算

### 整体流程

```
模型输出 v_t (velocity prediction)
    ↓
恢复干净 action: action_pred = x_t - t · v_t
    ↓
拆分为 pos / rot6d / grip
    ↓
分别计算 loss → 加权求和
```

### 各分项

```
Loss = w_pos · MSE(pos_pred, pos_gt)      ← L2 距离
     + w_rot · geodesic(rot_pred, rot_gt)  ← 角度距离（弧度）
     + w_grip · MSE(grip_pred, grip_gt)    ← L2 距离
```

### Geodesic Loss（旋转）

```python
R_err = R_pred^T @ R_gt
cos_θ = (trace(R_err) - 1) / 2
loss = arccos(clamp(cos_θ, -1, 1))    # 弧度, 值域 [0, π]
```

- 单位是**弧度**，值域 `[0, π]`
- 45° 误差 ≈ 0.79 rad，90° 误差 ≈ 1.57 rad
- 训练时用弧度，**评估时转成度**（`loss / π * 180`）

### 权重默认值（需根据数据调）

在 `Pi0Config` 中：

```python
Pi0Config(
    use_geodesic_loss=True,
    pos_loss_weight=1.0,    # MSE 权重
    rot_loss_weight=1.0,    # geodesic 权重（弧度）
    grip_loss_weight=1.0,   # MSE 权重
)
```

**重要**：MSE 的尺度和 geodesic（弧度）的尺度通常不在同一量级。归一化后 position MSE 通常在 0.01~0.1 范围，而 geodesic 在 0.1~0.5 弧度范围。建议：
- 先用 `rot_loss_weight=1.0` 跑几步，观测各分项的数值
- 如果 geodesic loss 主导了总 loss，适当降低 `rot_loss_weight`（如 0.1~0.5）
- 如果 position 拟合不好，增大 `pos_loss_weight`

---

## 3. 数据归一化

pi0.5 使用 **Quantile Normalization**（代码默认），将数据映射到 `[-1, 1]`：

```
x_norm = (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
```

其中 `q01` 和 `q99` 是第 1 和第 99 百分位数。

### 计算归一化参数

```bash
uv run scripts/compute_norm_stats.py \
    --config-name pi05_umi \
    --data.repo-id your_hf_username/my_umi_dataset
```

这会生成 `norm_stats.json` 存入 checkpoint 目录。

### 旋转需要归一化吗？

**不需要对旋转单独归一化。** 6D 表示的每个分量来自旋转矩阵的列向量，天然在 `[-1, 1]` 范围内（因为旋转矩阵的列是单位向量）。归一化参数对整个 10-dim action 统一计算即可，位置和夹爪受益更多。

### 位置归一化的注意事项

如果 UMI 数据中末端位姿的位移范围很小（如 ±2cm），归一化后 MSE 会很小。Geodesic loss 不受影响。如果发现 pos loss 太小（如 1e-5），可以考虑增大 `pos_loss_weight` 或取消位置的归一化。

---

## 4. 冻结策略

| 组件 | 参数量 | 策略 | 原因 |
|------|--------|------|------|
| VLM (Gemma 2B / language_model) | 2.5B | **🔒 冻结** | 语言理解已充分预训练，无需再学 |
| ViT (SigLIP / vision_tower) | 412M | **✅ 训练** | 适配 UMI 鱼眼相机 |
| Multi-Modal Projector | 2.4M | **✅ 训练** | 桥接变化的 ViT 特征到冻结的 VLM |
| Action Expert (Gemma 300M) | 691M | **✅ 训练** | 学习 UMI 特定的操控策略 |
| Time MLP + 投影层 | ~4M | **✅ 训练** | 参数极少，与 action 直接相关 |

冻结代码（`test_pi05_latency.py` 中的 `freeze_vlm_keep_vit`）：

```python
for name, param in model.named_parameters():
    if "language_model" in name:
        param.requires_grad = False   # 冻结 VLM
    else:
        param.requires_grad = True    # 训练 ViT + Action Expert
```

**显存节省**：冻结 VLM 后，优化器状态从 ~14GB 降到 ~4GB，总显存 ~14-16GB（batch=8）。

---

## 5. 学习率建议

UMI 文档建议解冻 ViT 后用更小的学习率：

| 参数组 | LR | 原因 |
|--------|-----|------|
| 全局 (peak_lr) | 1e-5 | 防止鱼眼相机的视觉梯度冲击 |
| VLM | 0（冻结） | — |

默认 CosineDecaySchedule：`peak_lr=1e-5`，warmup 1000 steps，decay 到 `peak_lr`（即不衰减）。

训练后期如果需要，可以手动调低到 `5e-6`。

---

## 6. 配置文件一览

### Pi0Config（模型层）

```python
Pi0Config(
    pi05=True,
    action_horizon=10,        # 预测未来 10 步 action
    use_geodesic_loss=True,   # 启用 geodesic loss
    num_robots=1,             # 单臂=1，双臂=2
    pos_dim=3,
    rot_dim=6,
    grip_dim=1,
    pos_loss_weight=1.0,
    rot_loss_weight=1.0,      # ⚠️ 需要根据数据 tune
    grip_loss_weight=1.0,
)
```

### TrainConfig（训练层）

```python
TrainConfig(
    name="pi05_umi",
    model=Pi0Config(...),
    data=LeRobotUMIDataConfig(
        repo_id="your_hf_username/my_umi_dataset",
        action_dim=10,
        num_robots=1,
    ),
    weight_loader=CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    batch_size=32,                   # 全局 batch
    num_train_steps=30_000,
    save_interval=5_000,
    lr_schedule=CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=1e-5,                # 小 LR
        decay_steps=1_000_000,
        decay_lr=1e-5,
    ),
)
```

---

## 7. 数据准备

### LeRobot v2.0 格式

```python
features = {
    "observation.state":  {"dtype": "float32", "shape": (10,)},
    "action":             {"dtype": "float32", "shape": (10,)},
    "observation.images.cam_high":        {"dtype": "image", "shape": (480, 640, 3)},
    "observation.images.cam_left_wrist":  {"dtype": "image", "shape": (480, 640, 3)},
    "observation.images.cam_right_wrist": {"dtype": "image", "shape": (480, 640, 3)},
}

# 每帧：
dataset.add_frame({
    "observation.state":  eef_state_10d,   # 当前末端位姿 (10,)
    "action":             eef_action_10d,  # 下一步的 delta 位姿 (10,)
    "observation.images.cam_high":        img_high,
    "observation.images.cam_left_wrist":  img_lwrist,
    "observation.images.cam_right_wrist": img_rwrist,
    "task": "pick the red block",          # 语言指令
})
```

**关键点**：
- `action` 是 **相对位移**（delta），不是绝对位姿
- `observation.state` 是 **当前绝对位姿**（同样是 10-dim 格式）
- 旋转部分用 **6D 表示**，不是 axis-angle 或四元数
- 如果原始数据是 axis-angle，需要先转成旋转矩阵再取前两列

### 从 axis-angle 转换到 6D

```python
# axis_angle: (3,) = (rx, ry, rz)
# → 旋转矩阵 R (3×3)
# → 取前两列 → 6D
from scipy.spatial.transform import Rotation
R = Rotation.from_rotvec(axis_angle).as_matrix()
rot6d = np.concatenate([R[:, 0], R[:, 1]])  # (6,)
```

---

## 8. 训练流程

### Step 1: 准备数据

将 UMI 数据转换为 LeRobot v2.0 格式，上传到 HuggingFace Hub（或本地路径）。

### Step 2: 计算归一化参数

```bash
uv run scripts/compute_norm_stats.py \
    --config-name pi05_umi \
    --data.repo-id your_hf_username/my_umi_dataset
```

### Step 3: 下载 base checkpoint

```bash
# 从 GCS 下载 pi05_base（需要 gcloud 认证）
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base ~/.cache/openpi/openpi-assets/checkpoints/
```

如果没有 GCS 访问权限，需要先将 JAX checkpoint 转为 PyTorch 格式，或联系 PI 团队获取 PyTorch checkpoint。

### Step 4: 启动训练

**单卡**：
```bash
python scripts/train_pytorch.py pi05_umi \
    --exp-name my_umi_v1 \
    --batch-size 8 \
    --pytorch-training-precision bfloat16 \
    --save-interval 5000
```

**多卡（DDP）**：
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py pi05_umi \
    --exp-name my_umi_v1 \
    --batch-size 32 \
    --pytorch-training-precision bfloat16 \
    --save-interval 5000
```

### Step 5: 监控 loss

训练日志中注意观察 loss 值：
- **刚开始**：pos_loss ~0.01-0.1, rot_loss ~0.3-1.0 rad, grip_loss ~0.01
- **收敛后**：pos_loss ~0.001, rot_loss ~0.05-0.2 rad (~3-10°), grip_loss ~0.001

如果 rot_loss 一直很大（>0.5 rad ≈ 30°），检查旋转数据的 6D 表示是否正确。

### Step 6: 评估

```python
from openpi.shared import rotation_utils

# 推理
action_pred = policy.infer(obs)["actions"]  # (AH, 10)

# 旋转误差（度）
rot_error_deg = rotation_utils.compute_rotation_error_degrees(
    torch.tensor(action_pred[..., 3:9]),
    torch.tensor(action_gt[..., 3:9]),
)
print(f"Rotation error: {rot_error_deg:.1f}°")
```

---

## 9. 常见问题排查

| 问题 | 可能原因 | 检查方法 |
|------|---------|---------|
| rot_loss 不下降 | 数据中 6D 表示有误 | 检查 6D→matrix 的 orthogonality |
| pos_loss 异常大/小 | 归一化参数不对 | 检查 `norm_stats.json` 中的 q01/q99 |
| OOM | batch size 太大 | 降到 4，或确认 VLM 已冻结 |
| Loss 中出现 NaN | 6D 输入导致 Gram-Schmidt 异常 | 检查 6D 值是否在合理范围（-1 到 1） |
| action_pred 和 actions 差距很大 | flow recovery 公式有误 | `action_pred` 应接近 `actions`（尤其在低噪阶段） |

---

## 10. 关键文件索引

| 文件 | 用途 |
|------|------|
| `src/openpi/shared/rotation_utils.py` | 6D↔matrix, geodesic_loss |
| `src/openpi/models/pi0_config.py` | Pi0Config（use_geodesic_loss 等配置） |
| `src/openpi/models_pytorch/pi0_pytorch.py` | PyTorch 模型 forward（geodesic loss 分支） |
| `src/openpi/policies/umi_policy.py` | UMI 输入输出 transform |
| `src/openpi/training/config.py` | pi05_umi TrainConfig + LeRobotUMIDataConfig |
| `test_pi05_latency.py` | UMI 微调显存测试 |
| `test_umi_verify.py` | 6D 旋转 + geodesic loss 正确性验证 |
| `docs/umi_finetune.md` | 本文档 |
| `docs/umi_finetune_config.md` | 原始 UMI 策略建议 |
