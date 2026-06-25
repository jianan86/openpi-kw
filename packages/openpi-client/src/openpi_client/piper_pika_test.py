import numpy as np

from openpi_client import piper_pika


def test_ee_tcp_roundtrip():
    ee_pose = np.array([0.4, -0.1, 0.3, 0.2, -0.3, 0.4, 0.05], dtype=np.float32)
    tcp_pose = piper_pika.ee_pose7_to_tcp_pose7(ee_pose)
    np.testing.assert_allclose(piper_pika.tcp_pose7_to_ee_pose7(tcp_pose), ee_pose, atol=1e-5)


def test_relative_actions_to_absolute_tcp():
    right = np.array([0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 0.04], dtype=np.float32)
    left = np.array([0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.06], dtype=np.float32)
    state = np.concatenate([piper_pika.pose7_to_pose10(right), piper_pika.pose7_to_pose10(left)])
    identity = piper_pika.matrix_to_rot6d(np.eye(3, dtype=np.float32))
    action = np.zeros((2, 20), dtype=np.float32)
    for arm_start in (0, 10):
        action[:, arm_start + 3 : arm_start + 9] = identity
    action[0, :3] = [0.01, 0.02, -0.01]
    action[0, 9] = 0.03
    action[0, 10:13] = [-0.02, 0.0, 0.01]
    action[0, 19] = 0.05

    result = piper_pika.relative_actions_to_absolute_tcp(action, state)

    np.testing.assert_allclose(result[0, :3], right[:3] + action[0, :3], atol=1e-6)
    np.testing.assert_allclose(result[0, 7:10], left[:3] + action[0, 10:13], atol=1e-6)
    assert result[0, 6] == np.float32(0.03)
    assert result[0, 13] == np.float32(0.05)


def test_limit_pose_step_uses_shortest_rotation_delta():
    current = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi - 0.01, 0.05], dtype=np.float32)
    target = np.array([1.0, -1.0, 0.5, 1.0, -1.0, -np.pi + 0.01, 0.2], dtype=np.float32)

    limited = piper_pika.limit_pose_step(current, target, 0.1, 0.2, 0.01, (0.0, 0.1))

    np.testing.assert_allclose(limited[:3], [0.1, -0.1, 0.1], atol=1e-6)
    np.testing.assert_allclose(limited[3:5], [0.2, -0.2], atol=1e-6)
    np.testing.assert_allclose(limited[5] - current[5], 0.02, atol=1e-5)
    np.testing.assert_allclose(limited[6], 0.06, atol=1e-6)


def test_accelerate_gripper_closure():
    actions = np.zeros((3, 14), dtype=np.float32)
    actions[:, 6] = [0.08, 0.075, 0.07]
    actions[:, 13] = [0.04, 0.05, 0.06]
    original = actions.copy()

    result, offsets = piper_pika.accelerate_gripper_closure(actions)

    np.testing.assert_allclose(result[:, 6], original[:, 6] - 0.01)
    np.testing.assert_allclose(result[:, 13], original[:, 13])
    np.testing.assert_allclose(actions, original)
    np.testing.assert_allclose(offsets, [0.01, 0.0])


def test_accelerate_gripper_closure_uses_proportional_offset():
    actions = np.zeros((2, 14), dtype=np.float32)
    actions[:, 6] = [0.09, 0.01]
    actions[:, 13] = [0.08, 0.02]

    result, offsets = piper_pika.accelerate_gripper_closure(actions)

    np.testing.assert_allclose(offsets, [0.016, 0.012])
    np.testing.assert_allclose(result[:, 6], actions[:, 6] - 0.016)
    np.testing.assert_allclose(result[:, 13], actions[:, 13] - 0.012)


def test_accelerate_gripper_closure_rejects_wrong_shape():
    try:
        piper_pika.accelerate_gripper_closure(np.zeros((2, 20), dtype=np.float32))
    except ValueError as error:
        assert "(T, 14)" in str(error)
    else:
        raise AssertionError("expected ValueError")
