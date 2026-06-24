from __future__ import annotations

import argparse
import logging
from pathlib import Path

from openpi_client import action_chunk_broker
from openpi_client import piper_pika
from openpi_client import websocket_client_policy
from openpi_client.runtime import runtime
from openpi_client.runtime.agents import policy_agent
import yaml

from examples.piper_pika import env
from examples.piper_pika import hardware


def run(args: argparse.Namespace) -> None:
    with Path(args.hardware_config).expanduser().open() as file:
        config = yaml.safe_load(file)

    tunnel = None
    robot = None
    environment = None
    try:
        if args.no_tunnel:
            host, port = args.host, args.port
        else:
            tunnel = piper_pika.SshTunnel(
                server=args.server_ssh,
                ssh_port=args.server_ssh_port,
                remote_host=args.remote_host,
                remote_port=args.port,
                local_port=args.local_port,
                connect_timeout=args.connect_timeout,
            )
            port = tunnel.start()
            host = "127.0.0.1"

        robot = hardware.PiperPikaHardware(
            config=config,
            arm_mode=args.arm_mode,
            dry_run=args.dry_run,
            no_piper=args.no_piper,
            no_pika=args.no_pika,
        )
        environment = env.PiperPikaEnvironment(
            hardware=robot,
            prompt=args.instruction,
            home=config.get("home_pos") if args.use_home_pos else None,
            max_pos_step=args.max_pos_step,
            max_rot_step=args.max_rot_step,
            max_gripper_step=args.max_gripper_step,
            min_gripper=args.min_gripper,
            max_gripper=args.max_gripper,
            dry_run=args.dry_run,
        )
        remote_policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        absolute_policy = piper_pika.RelativeActionPolicy(remote_policy, action_horizon=args.action_horizon)
        agent = policy_agent.PolicyAgent(
            action_chunk_broker.ActionChunkBroker(absolute_policy, action_horizon=args.action_horizon)
        )
        runtime.Runtime(
            environment=environment,
            agent=agent,
            subscribers=[],
            max_hz=args.control_hz,
            num_episodes=1,
            max_episode_steps=args.max_episode_steps,
        ).run()
    finally:
        if environment is not None:
            environment.close()
        elif robot is not None:
            robot.close()
        if tunnel is not None:
            tunnel.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenPI remote inference client for Piper + Pika")
    parser.add_argument("--server-ssh", default="jianan@183.230.224.121")
    parser.add_argument("--server-ssh-port", default=50210, type=int)
    parser.add_argument("--remote-host", default="127.0.0.1")
    parser.add_argument("--host", default="127.0.0.1", help="direct WebSocket host used with --no-tunnel")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--local-port", default=0, type=int)
    parser.add_argument("--connect-timeout", default=10.0, type=float)
    parser.add_argument("--no-tunnel", action="store_true")
    parser.add_argument("--hardware-config", default="examples/piper_pika/hardware_config.yaml")
    parser.add_argument("--arm-mode", choices=("dual", "single"), default="dual")
    parser.add_argument("--instruction", default="move")
    parser.add_argument("--action-horizon", default=10, type=int)
    parser.add_argument("--control-hz", default=30.0, type=float)
    parser.add_argument("--max-episode-steps", default=0, type=int)
    parser.add_argument("--use-home-pos", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-piper", action="store_true")
    parser.add_argument("--no-pika", action="store_true")
    parser.add_argument("--max-pos-step", default=0.01, type=float)
    parser.add_argument("--max-rot-step", default=0.05, type=float)
    parser.add_argument("--max-gripper-step", default=0.005, type=float)
    parser.add_argument("--min-gripper", default=0.0, type=float)
    parser.add_argument("--max-gripper", default=0.10, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    run(parse_args())
