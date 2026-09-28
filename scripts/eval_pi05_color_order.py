#!/usr/bin/env python3
"""Evaluate each IID order equally and RGB as OOD using the existing scheduler."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
IID_ORDERS = ("RBG", "GRB", "GBR", "BRG", "BGR")


def build_commands(args):
    orders = IID_ORDERS if args.split == "iid" else ("RGB",) if args.split == "ood" else (*IID_ORDERS, "RGB")
    commands = []
    for order in orders:
        command = [
            sys.executable, str(ROOT / "scripts/eval_policy_multitask.py"),
            "--config", str(args.scheduler_config), "--policy-name", "Pi_05",
            "--ckpt-name", str(args.checkpoint), "--env-cfg-type", "aloha_agilex",
            "--policy-conda-env", "uv", "--eval-env-conda-env", args.eval_env,
            "--task-config", "color_order", "--target-order", order,
            "--test-num", str(args.test_num), "--seed", str(args.seed),
            "--instruction-type", args.instruction_type, "--frequency", "15",
            "--no-expert-check",
        ]
        if args.dry_run:
            command.append("--dry-run")
        commands.append((order, command))
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("language", "icl"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("iid", "ood", "all"), default="all")
    parser.add_argument("--test-num", type=int, default=100, help="Episodes PER order.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instruction-type", choices=("seen", "unseen"), default="seen")
    parser.add_argument("--eval-env", default="RoboTwin")
    parser.add_argument("--scheduler-config", type=Path, default=ROOT / "env_cfg/eval/color_order.yml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.test_num <= 0:
        parser.error("--test-num must be positive")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    deploy = ROOT / f"XPolicyLab/policy/Pi_05/deploy_color_order_{args.mode}.yml"
    env = dict(os.environ, OPENPI_DEPLOY_CONFIG=str(deploy))
    for order, command in build_commands(args):
        split = "OOD" if order == "RGB" else "IID"
        print(f"{split} {order}: {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
