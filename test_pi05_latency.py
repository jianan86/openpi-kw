"""Test pi0.5 UMI fine-tuning: geodesic loss + freeze VLM memory usage.

UMI action format: pos(3) + rot6d(6) + grip(1) = 10 dims per robot.
Strategy: Freeze VLM (Gemma 2B), train ViT + Action Expert + Projections.
Loss: MSE(pos+grip) + geodesic(rotation in radians).
"""

import gc
import types
import time

import torch

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.shared import rotation_utils


def log_step(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mem_gb(device):
    return torch.cuda.max_memory_allocated(device) / 1024**3


def freeze_vlm_keep_vit(model: PI0Pytorch):
    """UMI strategy: freeze language_model (VLM/Gemma 2B), train ViT + Action Expert."""
    frozen, trainable = 0, 0
    for name, param in model.named_parameters():
        if "language_model" in name:
            param.requires_grad = False
            frozen += param.numel()
        else:
            param.requires_grad = True
            trainable += param.numel()
    log_step(f"Freeze: {frozen/1e6:.0f}M params (VLM), {trainable/1e6:.0f}M trainable (ViT+Action)")
    return model


def create_umi_batch(batch_size, config, device):
    """Create random UMI training batch.

    Per-robot action: pos(3) + rot6d(6) + grip(1) = 10 dims.
    Total: num_robots * 10 dims.
    """
    action_dim = config.action_dim  # auto-set to num_robots * 10

    images = {}
    image_masks = {}
    for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
        images[key] = torch.rand(batch_size, 3, 224, 224, dtype=torch.float32, device=device) * 2 - 1
        image_masks[key] = torch.ones(batch_size, dtype=torch.bool, device=device)

    state = torch.rand(batch_size, action_dim, dtype=torch.float32, device=device) * 2 - 1
    tokenized_prompt = torch.randint(0, 256000, (batch_size, config.max_token_len), dtype=torch.int32, device=device)
    tokenized_prompt_mask = torch.ones(batch_size, config.max_token_len, dtype=torch.bool, device=device)
    tokenized_prompt_mask[:, -10:] = False

    obs = types.SimpleNamespace(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        token_ar_mask=None,
        token_loss_mask=None,
    )

    # Generate valid rot6d: first two columns of a random rotation matrix
    # Random rotation via QR decomposition of random matrix
    rand_mat = torch.randn(batch_size, config.action_horizon, config.num_robots, 3, 3, device=device)
    Q, _ = torch.linalg.qr(rand_mat)  # (B, AH, NR, 3, 3) — orthonormal
    # For pi0.5 num_steps with random time, Q might not be a proper rotation
    # (det could be -1), but this is close enough for a memory test.

    # Extract 6D from the rotation matrix
    rot6d = torch.cat([Q[..., 0], Q[..., 1]], dim=-1)  # (B, AH, NR, 6)

    # Build full action
    pos = torch.randn(batch_size, config.action_horizon, config.num_robots, 3, device=device) * 0.01
    grip = torch.rand(batch_size, config.action_horizon, config.num_robots, 1, device=device)

    actions = torch.cat([pos, rot6d, grip], dim=-1)  # (B, AH, NR, 10)
    actions = actions.reshape(batch_size, config.action_horizon, -1)  # (B, AH, action_dim)

    return obs, actions


def test_umi(batch_size, config, device):
    """Run one training step and return peak memory."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    model = PI0Pytorch(config).to(device)
    model.gradient_checkpointing_enable()
    model = freeze_vlm_keep_vit(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)

    obs, actions = create_umi_batch(batch_size, config, device)

    # Dry run (CUDA compile)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model(obs, actions).mean()
        loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Real measurement
    torch.cuda.reset_peak_memory_stats(device)
    obs, actions = create_umi_batch(batch_size, config, device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model(obs, actions).mean()
        loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    peak = mem_gb(device)

    # Quick sanity: verify 6D->matrix conversion works
    dummy_rot6d = torch.randn(batch_size, config.action_horizon, 6, device=device)
    R = rotation_utils.rot6d_to_mat_torch(dummy_rot6d)
    assert R.shape == (batch_size, config.action_horizon, 3, 3), f"Bad rot mat shape: {R.shape}"

    del model, optimizer, trainable_params, obs, actions
    torch.cuda.empty_cache()
    gc.collect()
    return peak, loss.item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
    log_step(f"GPU: {gpu_name} ({gpu_mem:.0f}GB)")

    # UMI config: 1 robot, 10-dim action, geodesic loss
    config = Pi0Config(
        pi05=True,
        action_horizon=10,
        use_geodesic_loss=True,
        num_robots=1,
        pos_dim=3,
        rot_dim=6,
        grip_dim=1,
        pos_loss_weight=1.0,
        rot_loss_weight=1.0,
        grip_loss_weight=1.0,
    )
    log_step(f"Config: action_dim={config.action_dim} (auto-set), geodesic_loss=True")

    # Test batch sizes
    print(f"\n{'Batch':>8}  {'Peak Mem':>10}  {'Status':>14}  {'Loss':>10}  Note")
    print("-" * 72)

    max_ok = 0
    for bs in [1, 2, 4, 6, 8, 10, 12]:
        try:
            peak, loss_val = test_umi(bs, config, device)
            status = "FITS 24GB ✅" if peak <= 24 else "OOM ❌"
            note = f"{24 - peak:.1f}GB margin" if peak <= 24 else ""
            print(f"{bs:>8}  {peak:>8.1f} GB  {status:>14}  {loss_val:>8.4f}  {note}")
            if peak <= 24:
                max_ok = bs
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>8}  {'OOM':>10}  {'OOM ❌':>14}")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"{bs:>8}  {'ERR':>10}  {'ERROR':>14}  {e}")
            break

    print("-" * 72)

    print(f"\n{'='*65}")
    print(f"pi0.5 UMI Fine-Tuning — Summary")
    print(f"{'='*65}")
    print(f"  Action format:   pos(3) + rot6d(6) + grip(1) = 10 dims")
    print(f"  Loss:            MSE(pos+grip) + geodesic(rotation, rad)")
    print(f"  Freeze:          VLM (Gemma 2B)")
    print(f"  Train:           ViT (SigLIP) + Action Expert (300M)")
    print(f"  GPU:             {gpu_name} ({gpu_mem:.0f}GB)")
    print(f"{'='*65}")

    if max_ok >= 4:
        print(f"\n  ✅ 4×24GB — easily fits (per-GPU batch={max_ok}, global={max_ok * 4})")
        print(f"     Launch: torchrun --nproc_per_node=4 scripts/train_pytorch.py pi05_umi \\")
        print(f"             --exp-name my_umi --batch-size {max_ok * 4}")
    elif max_ok >= 1:
        print(f"\n  ⚠️  4×24GB fits but tight (per-GPU batch={max_ok}) — reduce batch or use FSDP")

    print(f"{'='*65}")


if __name__ == "__main__":
    main()
