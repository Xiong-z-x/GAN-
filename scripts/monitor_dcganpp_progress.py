from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


EXPERIMENTS = [
    ("E0 基线 DCGAN", "outputs/dcgan_e0_baseline"),
    ("E1 稳定化 DCGAN", "outputs/dcgan_e1_stable"),
    ("E2 结构增强 DCGAN", "outputs/dcgan_e2_residual"),
    ("E3 128 分辨率 DCGAN++", "outputs/dcgan_e3_128"),
    ("E4 R1 细化 DCGAN++", "outputs/dcgan_e4_r1"),
]


@dataclass(frozen=True)
class ExperimentState:
    name: str
    path: Path
    exists: bool
    epochs: int = 0
    batch_size: int = 0
    dataset_size: int = 0
    total_steps: int = 0
    current_step: int = 0
    current_epoch: int = 0
    latest_loss_d: float | None = None
    latest_loss_g: float | None = None
    checkpoint_epochs: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时监控 DCGAN++ 实验进度")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="项目根目录")
    parser.add_argument("--interval", type=int, default=10, help="刷新间隔，单位秒")
    parser.add_argument("--once", action="store_true", help="只输出一次")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_last_jsonl(path: Path) -> dict:
    if not path.exists():
        return {}
    last_line = ""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        end = handle.tell()
        block_size = 4096
        data = b""
        while end > 0 and b"\n" not in data.rstrip(b"\n"):
            read_size = min(block_size, end)
            end -= read_size
            handle.seek(end)
            data = handle.read(read_size) + data
        lines = [line for line in data.decode("utf-8", errors="ignore").splitlines() if line.strip()]
        if lines:
            last_line = lines[-1]
    if not last_line:
        return {}
    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        return {}


def count_checkpoint_epochs(path: Path) -> int:
    checkpoints_dir = path / "checkpoints"
    if not checkpoints_dir.exists():
        return 0
    return len(list(checkpoints_dir.glob("generator_epoch_*.pth")))


def collect_state(project_root: Path, name: str, relative_path: str) -> ExperimentState:
    path = project_root / relative_path
    if not path.exists():
        return ExperimentState(name=name, path=path, exists=False)

    config = read_json(path / "config.json")
    last_log = read_last_jsonl(path / "loss_history.jsonl")
    epochs = int(config.get("epochs", 0) or 0)
    batch_size = int(config.get("batch_size", 0) or 0)
    dataset_size = int(config.get("dataset_size", 0) or 0)
    steps_per_epoch = dataset_size // batch_size if batch_size > 0 else 0
    total_steps = steps_per_epoch * epochs
    current_step = int(last_log.get("step", 0) or 0)
    current_epoch = int(last_log.get("epoch", 0) or 0)

    loss_d = last_log.get("loss_d")
    loss_g = last_log.get("loss_g")
    return ExperimentState(
        name=name,
        path=path,
        exists=True,
        epochs=epochs,
        batch_size=batch_size,
        dataset_size=dataset_size,
        total_steps=total_steps,
        current_step=current_step,
        current_epoch=current_epoch,
        latest_loss_d=float(loss_d) if loss_d is not None else None,
        latest_loss_g=float(loss_g) if loss_g is not None else None,
        checkpoint_epochs=count_checkpoint_epochs(path),
    )


def get_gpu_summary() -> str:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return "GPU 信息暂不可用"
    line = output.strip().splitlines()[0] if output.strip() else ""
    if not line:
        return "GPU 信息暂不可用"
    parts = [item.strip() for item in line.split(",")]
    if len(parts) < 6:
        return line
    return f"{parts[0]} | 利用率 {parts[1]}% | 显存 {parts[2]}/{parts[3]} MiB | 温度 {parts[4]}C | 功耗 {parts[5]}W"


def list_processes(project_root: Path) -> list[str]:
    keywords = [
        "src.dcgan.train",
        "run_dcganpp_suite.sh",
        "run_sota_enhanced_pipeline.sh",
        "run_stylegan3_generate.sh",
        "run_stylegan3_video.sh",
        "run_animegan2_infer.sh",
        "run_cyclegan_pretrained_style.sh",
        "download_ffhq.py",
        "python -m src.dcgan.train",
    ]
    try:
        output = subprocess.check_output(
            ["ps", "-eo", "pid,etimes,pcpu,pmem,args"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ["进程信息暂不可用"]

    lines: list[str] = []
    for raw_line in output.splitlines()[1:]:
        line = raw_line.strip()
        if not line:
            continue
        if str(project_root) not in line and not any(keyword in line for keyword in keywords):
            continue
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        pid, elapsed, cpu, mem, cmd = parts
        lines.append(f"PID {pid} | 运行 {elapsed}s | CPU {cpu}% | 内存 {mem}% | {cmd}")
    return lines or ["未找到相关进程"]


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "暂不可估计"
    seconds = int(seconds)
    return str(timedelta(seconds=seconds))


def progress_bar(ratio: float, width: int = 30) -> str:
    ratio = min(max(ratio, 0.0), 1.0)
    filled = int(round(width * ratio))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def print_report(states: list[ExperimentState], started_at: float, project_root: Path) -> None:
    now = time.time()
    elapsed = now - started_at
    total_steps = sum(state.total_steps for state in states if state.exists)
    done_steps = sum(min(state.current_step, state.total_steps) for state in states if state.exists)
    active = next((state for state in states if state.exists and state.total_steps > 0 and state.current_step < state.total_steps), None)
    overall_ratio = done_steps / total_steps if total_steps > 0 else 0.0
    speed = done_steps / elapsed if elapsed > 0 and done_steps > 0 else None
    remaining_steps = max(total_steps - done_steps, 0)
    eta_seconds = remaining_steps / speed if speed else None

    print("\033[2J\033[H", end="")
    print(f"刷新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU：{get_gpu_summary()}")
    print()
    print(f"总进度：{progress_bar(overall_ratio)} {overall_ratio * 100:6.2f}%")
    print(f"已完成步数：{done_steps}/{total_steps}")
    print(f"平均速度：{speed:.2f} step/s" if speed else "平均速度：暂不可估计")
    print(f"预计剩余：{format_duration(eta_seconds)}")
    if eta_seconds is not None:
        finish_time = datetime.now() + timedelta(seconds=int(eta_seconds))
        print(f"预计完成时间：{finish_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("预计完成时间：暂不可估计")
    print()

    print("相关进程：")
    for item in list_processes(project_root):
        print(f"  {item}")
    print()

    for state in states:
        if not state.exists:
            print(f"{state.name}：未开始")
            continue
        ratio = state.current_step / state.total_steps if state.total_steps > 0 else 0.0
        loss_text = ""
        if state.latest_loss_d is not None and state.latest_loss_g is not None:
            loss_text = f" | D {state.latest_loss_d:.4f} | G {state.latest_loss_g:.4f}"
        active_mark = " <- 当前阶段" if active and active.name == state.name else ""
        print(
            f"{state.name}：{progress_bar(ratio, 20)} {ratio * 100:6.2f}% "
            f"| epoch {state.current_epoch}/{state.epochs} "
            f"| step {state.current_step}/{state.total_steps} "
            f"| checkpoint {state.checkpoint_epochs}{loss_text}{active_mark}"
        )


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    started_at = time.time()
    while True:
        states = [collect_state(project_root, name, relative_path) for name, relative_path in EXPERIMENTS]
        print_report(states, started_at, project_root)
        if args.once:
            break
        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    main()
