#!/usr/bin/env python3
"""Evaluate ActiveScope answer files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parent
STAGE2_DIR = RELEASE_ROOT / "stage2_zoomeye"


def infer_benchmark(answers_file: Path) -> str | None:
    parts = answers_file.parts
    for name in ("Vstar", "hr-bench_4k", "hr-bench_8k"):
        if name in parts:
            return name
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--benchmark", choices=["Vstar", "hr-bench_4k", "hr-bench_8k"], default=None)
    parser.add_argument("--python", default=sys.executable)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    answers_file = Path(args.answers_file).expanduser()
    if not answers_file.is_absolute():
        answers_file = Path.cwd() / answers_file

    benchmark = args.benchmark or infer_benchmark(answers_file)
    if benchmark is None:
        raise ValueError("--benchmark is required when it cannot be inferred from --answers-file")

    if benchmark == "Vstar":
        script = "ZoomEye/eval/eval_results_vstar.py"
    else:
        script = "ZoomEye/eval/eval_results_hr-bench.py"

    cmd = [args.python, script, "--answers-file", str(answers_file)]
    print("[cmd] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(STAGE2_DIR), check=True)


if __name__ == "__main__":
    main()
