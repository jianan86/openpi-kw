# Piper + Pika remote inference

Start the policy server on the GPU host:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_umi_bimanual_pika \
  --policy.dir=/path/to/checkpoint \
  --port=8000
```

Only SSH port `50210` needs to be externally reachable. The robot client creates
an SSH local-forward to the server's loopback WebSocket port:

```bash
uv run python -m examples.piper_pika.main \
  --server-ssh jianan@183.230.224.121 \
  --server-ssh-port 50210 \
  --instruction "move"
```

Use `--dry-run --no-piper --no-pika` for an end-to-end test without hardware.
Use `--arm-mode single` to duplicate the right-arm observation for the bimanual
model while controlling only the right Piper and Pika.

The remote protocol, observation request timing, and action chunk consumption are
the existing OpenPI WebSocket, `WebsocketClientPolicy`, and `ActionChunkBroker`
implementations. Hardware collection, EE/TCP conversion, and control conventions
follow the RDT2 `async_inference` Piper/Pika client.


## End-to-end flow

1. A background sensor worker reads both Pika fisheye cameras, Piper end-effector poses, and Pika gripper widths. Images are converted from BGR to RGB, padded without distortion, and resized directly to the model input size, `224x224`.
2. The client converts Piper EE poses to the Pika TCP frame. Each arm becomes `[xyz, rot6d, gripper]`; right and left arms form the 20-dimensional state. `cam_high` is a black image. In single-arm mode, the right state and image are duplicated for the bimanual model.
3. `Runtime` asks for an observation every control step. `ActionChunkBroker` only calls the remote policy when its current 10-step chunk is exhausted.
4. `WebsocketClientPolicy` serializes the observation with the OpenPI msgpack NumPy codec and sends it through the SSH local-forward.
5. The existing server pipeline repacks UMI keys, normalizes state, preprocesses images and prompt, runs model inference, unnormalizes output, and returns the first 20 active dimensions of the `10x32` model output.
6. `RelativeActionPolicy` uses the exact state sent with the request to convert each relative arm action into an absolute TCP target, producing a `10x14` chunk.
7. The existing broker returns one row per control step. The environment limits translation, shortest-path Euler rotation, and gripper changes, converts TCP targets back to Piper EE targets, and commands Piper and Pika.
8. Piper fatal status terms stop execution. `dry-run`, disabled-hardware modes, and optional joint-space home movement support staged validation.

## Implementation sources

| Behavior | Source |
| --- | --- |
| WebSocket protocol, msgpack payload, synchronous request/response | Existing OpenPI |
| Observation request timing and 10-step chunk consumption | Existing OpenPI `Runtime` and `ActionChunkBroker` |
| UMI mapping, normalization, inference, and output truncation | Existing OpenPI policy/config pipeline |
| SSH forwarding through port 50210 | RDT2 `async_inference/real_piper_client.py` |
| Pika camera/gripper and Piper SDK interaction | RDT2 `async_inference/real_piper_client.py` |
| EE/TCP calibration and relative-to-absolute pose math | RDT2 `async_inference/pose_utils.py` |
| 20D ordering and black `cam_high` | Existing OpenPI Pika dataset/config |

RDT2 gRPC, asynchronous observation queues, action streaming, waterline scheduling, and chunk merge strategies are intentionally not used.

## LeRobot VISTA checkpoint

The VISTA action-expert checkpoint uses the LeRobot gRPC protocol, 16D quaternion
state/actions, and left-arm-first ordering. Start its server on the GPU host:

```bash
conda activate rdt2
cd /home/jianan/code/umi-vista/post_training/lerobot
CUDA_VISIBLE_DEVICES=4 python -m lerobot.async_inference.policy_server \
  --host=127.0.0.1 --port=8080 --fps=30 --inference_latency=0 --obs_queue_timeout=1
```

Run sensor collection and one prediction without enabling robot motion:

```bash
conda activate lerobot
cd /home/kw/workspace/openpi-kw
export PYTHONPATH=$PWD/packages/openpi-client/src:$PWD
python -m examples.piper_pika.vista_grpc \
  --checkpoint=/data/jianan/outputs/vista_action_expert_train/26-07-01_18-07-02_vista_action_expert_gpu4567/checkpoints/060000/pretrained_model \
  --prediction-only --max-predictions=1
```

Motion is disabled by default. Only with an operator at the emergency stop, manually
place both arms at the task start pose and replace `--prediction-only` with
`--enable-motion --max-control-steps=30`. The motion path enforces 2 mm / 0.01 rad /
1 mm per-step limits and a 2 cm total translation guard for each arm. There is no total rotation guard by default. It never moves
to `home_pos` automatically.
