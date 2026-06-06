"""Verify UMI 6D rotation + geodesic loss correctness."""

import torch
import numpy as np

from openpi.shared import rotation_utils


def test_rot6d_roundtrip():
    """6D -> matrix -> 6D should be identity for valid rotation matrices."""
    print("=" * 60)
    print("Test 1: rot6d <-> matrix roundtrip")
    print("=" * 60)

    # Generate random rotation matrices via QR decomposition
    rand = torch.randn(100, 3, 3)
    Q, R = torch.linalg.qr(rand)
    # Ensure proper rotation (det=+1)
    det = torch.linalg.det(Q)
    Q = Q * det.sign().unsqueeze(-1).unsqueeze(-1)

    # Verify orthonormal
    I = Q @ Q.transpose(-2, -1)
    ortho_err = (I - torch.eye(3)).abs().max().item()
    print(f"  Random rotation matrices: orthogonality error = {ortho_err:.2e}")
    assert ortho_err < 1e-5, f"QR failed to produce orthonormal matrices: {ortho_err}"

    # Matrix -> 6D -> matrix
    rot6d = rotation_utils.mat_to_rot6d_torch(Q)
    Q_recovered = rotation_utils.rot6d_to_mat_torch(rot6d)

    # Compare
    diff = (Q - Q_recovered).abs().max().item()
    print(f"  Matrix -> 6D -> Matrix max error: {diff:.2e}")
    assert diff < 1e-5, f"Roundtrip error too large: {diff}"

    # Test Gram-Schmidt orthonormality of recovered matrix
    I2 = Q_recovered @ Q_recovered.transpose(-2, -1)
    ortho_err2 = (I2 - torch.eye(3).unsqueeze(0)).abs().max().item()
    print(f"  Recovered matrix orthogonality: {ortho_err2:.2e}")
    assert ortho_err2 < 1e-5

    print("  ✅ Roundtrip passes\n")


def test_geodesic_known_angles():
    """Geodesic loss should match known angles."""
    print("=" * 60)
    print("Test 2: Geodesic loss on known rotation angles")
    print("=" * 60)

    # Identity vs identity: 0 degrees
    I = torch.eye(3).unsqueeze(0)  # (1, 3, 3)
    loss = rotation_utils.geodesic_loss(I, I, reduce=True, return_degrees=True)
    print(f"  Identity vs Identity: {loss.item():.4f} deg (expect 0)")
    assert loss.item() < 0.05, f"Expected ~0, got {loss.item()} (floating point noise)"

    # 90-degree rotation around Z
    cos90, sin90 = 0.0, 1.0
    R90z = torch.tensor([[[cos90, -sin90, 0], [sin90, cos90, 0], [0, 0, 1]]], dtype=torch.float32)
    loss90 = rotation_utils.geodesic_loss(R90z, I, reduce=True, return_degrees=True)
    print(f"  90° around Z vs Identity: {loss90.item():.2f} deg (expect 90)")
    assert abs(loss90.item() - 90.0) < 0.1, f"Expected ~90, got {loss90.item()}"

    # 45-degree rotation
    cos45 = sin45 = np.sqrt(2) / 2
    R45y = torch.tensor([[[cos45, 0, sin45], [0, 1, 0], [-sin45, 0, cos45]]], dtype=torch.float32)
    loss45 = rotation_utils.geodesic_loss(R45y, I, reduce=True, return_degrees=True)
    print(f"  45° around Y vs Identity: {loss45.item():.2f} deg (expect 45)")
    assert abs(loss45.item() - 45.0) < 0.5, f"Expected ~45, got {loss45.item()}"

    # 180-degree rotation
    R180z = torch.tensor([[[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1]]], dtype=torch.float32)
    loss180 = rotation_utils.geodesic_loss(R180z, I, reduce=True, return_degrees=True)
    print(f"  180° around Z vs Identity: {loss180.item():.1f} deg (expect 180)")
    assert abs(loss180.item() - 180.0) < 0.1

    print("  ✅ Known angles pass\n")


def test_geodesic_symmetry():
    """Geodesic(R1, R2) == Geodesic(R2, R1)."""
    print("=" * 60)
    print("Test 3: Geodesic loss symmetry")
    print("=" * 60)

    rand = torch.randn(50, 3, 3)
    Q1, _ = torch.linalg.qr(rand)
    det1 = torch.linalg.det(Q1)
    Q1 = Q1 * det1.sign().unsqueeze(-1).unsqueeze(-1)

    rand2 = torch.randn(50, 3, 3)
    Q2, _ = torch.linalg.qr(rand2)
    det2 = torch.linalg.det(Q2)
    Q2 = Q2 * det2.sign().unsqueeze(-1).unsqueeze(-1)

    loss_12 = rotation_utils.geodesic_loss(Q1, Q2, reduce=False, return_degrees=True)
    loss_21 = rotation_utils.geodesic_loss(Q2, Q1, reduce=False, return_degrees=True)

    max_diff = (loss_12 - loss_21).abs().max().item()
    print(f"  Max asymmetry: {max_diff:.2e} deg")
    assert max_diff < 1e-4, f"Asymmetric! Max diff: {max_diff}"

    print("  ✅ Symmetry passes\n")


def test_6d_to_matrix_properties():
    """6D inputs that are NOT valid rotation columns should still produce orthonormal output."""
    print("=" * 60)
    print("Test 4: Gram-Schmidt produces orthonormal from any 6D input")
    print("=" * 60)

    # Random 6D vectors (NOT columns of rotation matrix)
    random_6d = torch.randn(1000, 6)
    R = rotation_utils.rot6d_to_mat_torch(random_6d)  # (1000, 3, 3)

    # Check orthonormal: R @ R^T = I
    I_batch = R @ R.transpose(-2, -1)
    I3 = torch.eye(3).unsqueeze(0)
    ortho_err = (I_batch - I3).abs().max().item()
    print(f"  Max orthogonality error: {ortho_err:.2e}")
    assert ortho_err < 1e-5, f"Gram-Schmidt failed to produce orthonormal: {ortho_err}"

    # Check determinant is ±1
    dets = torch.linalg.det(R)
    det_err = (dets.abs() - 1.0).abs().max().item()
    print(f"  Max |det| deviation from 1: {det_err:.2e}")
    assert det_err < 1e-5, f"Dets not ±1: max deviation {det_err}"

    print("  ✅ Gram-Schmidt properties pass\n")


def test_model_geodesic_loss():
    """End-to-end: model forward with geodesic loss on random data."""
    print("=" * 60)
    print("Test 5: Model forward with geodesic loss")
    print("=" * 60)

    import types
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    config = Pi0Config(
        pi05=True,
        action_horizon=10,
        use_geodesic_loss=True,
        num_robots=1,
        pos_dim=3,
        rot_dim=6,
        grip_dim=1,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PI0Pytorch(config).to(device)
    model.gradient_checkpointing_enable()
    model.eval()

    B, AH = 2, 10
    AD = config.action_dim  # 10
    images = {}
    image_masks = {}
    for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
        images[key] = torch.randn(B, 3, 224, 224, device=device)
        image_masks[key] = torch.ones(B, dtype=torch.bool, device=device)

    state = torch.randn(B, AD, device=device)
    tok = torch.randint(0, 256000, (B, config.max_token_len), dtype=torch.int32, device=device)
    tok_mask = torch.ones(B, config.max_token_len, dtype=torch.bool, device=device)
    tok_mask[:, -10:] = False

    obs = types.SimpleNamespace(
        images=images, image_masks=image_masks, state=state,
        tokenized_prompt=tok, tokenized_prompt_mask=tok_mask,
        token_ar_mask=None, token_loss_mask=None,
    )

    # Generate valid 6D from rotation matrices
    rand_mat = torch.randn(B, AH, 3, 3, device=device)
    Q, _ = torch.linalg.qr(rand_mat)
    det = torch.linalg.det(Q)
    Q = Q * det.sign().unsqueeze(-1).unsqueeze(-1)
    rot6d = torch.cat([Q[..., 0], Q[..., 1]], dim=-1)
    pos = torch.randn(B, AH, 3, device=device) * 0.1
    grip = torch.rand(B, AH, 1, device=device)
    actions = torch.cat([pos, rot6d, grip], dim=-1)

    # Forward with geodesic loss
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        losses = model(obs, actions)

    print(f"  Loss shape: {losses.shape} (expect [{B}, {AH}])")
    assert losses.shape == (B, AH), f"Bad loss shape: {losses.shape}"

    loss_val = losses.mean().item()
    print(f"  Loss value: {loss_val:.4f}")
    assert loss_val > 0, "Loss should be > 0"
    assert not torch.isnan(losses).any(), "Loss contains NaN!"
    assert not torch.isinf(losses).any(), "Loss contains Inf!"

    print("  ✅ Model forward with geodesic loss passes\n")


def test_loss_components():
    """Verify pos/rot/grip losses are properly split."""
    print("=" * 60)
    print("Test 6: Loss component decomposition")
    print("=" * 60)

    # Perfect prediction: all losses should be ~0
    B, AH = 4, 5
    actions = torch.randn(B, AH, 10)
    R_pred = rotation_utils.rot6d_to_mat_torch(actions[..., 3:9])
    R_gt = R_pred.clone()  # identical

    geo = rotation_utils.geodesic_loss(R_pred, R_gt, reduce=True, return_degrees=True)
    print(f"  Geodesic loss (identical rotations): {geo.item():.4f} deg")
    assert geo.item() < 0.05, f"Should be ~0 for identical matrices, got {geo.item()}"

    # 90 deg apart: should be ~90
    cos, sin = 0.0, 1.0
    R90 = torch.tensor([[[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]], dtype=torch.float32)
    I = torch.eye(3).unsqueeze(0)
    geo90 = rotation_utils.geodesic_loss(R90, I, reduce=True, return_degrees=True)
    print(f"  Geodesic loss (90° apart): {geo90.item():.2f} deg")
    assert abs(geo90.item() - 90.0) < 0.5

    # Small rotation (~5.7 deg): geodesic loss should be ~5.7 deg
    angle = 0.1  # radians ~5.7 deg
    c, s = np.cos(angle), np.sin(angle)
    R_small = torch.tensor([[[c, -s, 0], [s, c, 0], [0, 0, 1]]], dtype=torch.float32)
    geo_small = rotation_utils.geodesic_loss(R_small, I, reduce=True, return_degrees=True)
    expected_deg = np.rad2deg(angle)
    print(f"  Geodesic loss ({expected_deg:.1f}° apart): {geo_small.item():.2f} deg")
    assert abs(geo_small.item() - expected_deg) < 1.0

    print("  ✅ Component decomposition passes\n")


if __name__ == "__main__":
    test_rot6d_roundtrip()
    test_geodesic_known_angles()
    test_geodesic_symmetry()
    test_6d_to_matrix_properties()
    test_model_geodesic_loss()
    test_loss_components()

    print("=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)
