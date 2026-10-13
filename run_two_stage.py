#!/usr/bin/env python3
"""Run ActiveScope: actively seeking and correcting perception for MLLMs."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional


RELEASE_ROOT = Path(__file__).resolve().parent
STAGE1_DIR = RELEASE_ROOT / "stage1_mllms_know"
STAGE2_DIR = RELEASE_ROOT / "stage2_zoomeye"


def resolve_release_path(value: Optional[str]) -> Optional[Path]:
    """Resolve local release paths while still allowing Hugging Face repo IDs."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path.parts[:1] == (RELEASE_ROOT.name,):
        return RELEASE_ROOT.parent / path
    if value.startswith(("./", "../")) or path.parts[:1] in {("data",), ("models",), ("outputs",)}:
        return RELEASE_ROOT / path
    if path.suffix in {".json", ".jsonl"}:
        return Path.cwd() / path
    return path


def load_config(path: Optional[str]) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cfg_get(config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return config.get(section, {}).get(key, default)


def run_command(cmd: list[str], cwd: Path, gpu: Optional[str], dry_run: bool) -> None:
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    print(f"[cwd] {cwd}")
    if gpu is not None:
        print(f"[env] CUDA_VISIBLE_DEVICES={gpu}")
    print("[cmd] " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def list_stage1_dirs(save_root: Path, model: str, task: str, method: str) -> set[Path]:
    return {p for p in save_root.glob(f"{model}_{task}_{method}_*") if p.is_dir()}


def find_stage1_bbox_file(
    save_root: Path,
    model: str,
    task: str,
    method: str,
    before_dirs: set[Path],
) -> Path:
    after_dirs = list_stage1_dirs(save_root, model, task, method)
    new_dirs = sorted(after_dirs - before_dirs, key=lambda p: p.stat().st_mtime)
    candidates = new_dirs if new_dirs else sorted(after_dirs, key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No stage1 output directory found under {save_root}")
    bbox_file = candidates[-1] / f"{model}-{task}-{method}.json"
    if not bbox_file.exists():
        raise FileNotFoundError(f"Expected bbox file not found: {bbox_file}")
    return bbox_file


def run_stage1(args: argparse.Namespace) -> Path:
    save_root = resolve_release_path(args.stage1_save_path) or (RELEASE_ROOT / "outputs" / "localization")
    model_path = resolve_release_path(args.model_path)
    question_path = resolve_release_path(args.stage1_question_path)
    image_path = resolve_release_path(args.stage1_image_path)
    before_dirs = list_stage1_dirs(save_root, args.stage1_model, args.stage1_task, args.stage1_method)

    cmd = [
        args.python,
        "run_localization.py",
        "--model",
        args.stage1_model,
        "--task",
        args.stage1_task,
        "--method",
        args.stage1_method,
        "--model-id",
        str(model_path),
        "--question-path",
        str(question_path),
        "--image-path",
        str(image_path),
        "--save-path",
        str(save_root),
        "--total-chunks",
        str(args.total_chunks),
        "--chunk-id",
        str(args.chunk_id),
        "--max-iters",
        str(args.max_iters),
        "--hr-question-mode",
        args.hr_question_mode,
        "--seed",
        str(args.seed),
    ]
    run_command(cmd, cwd=STAGE1_DIR, gpu=args.gpu, dry_run=args.dry_run)

    if args.dry_run:
        return Path("<stage1_bbox_file_from_new_output_dir>")
    return find_stage1_bbox_file(
        save_root,
        args.stage1_model,
        args.stage1_task,
        args.stage1_method,
        before_dirs,
    )


def run_stage1_inprocess(args: argparse.Namespace):
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    save_root = resolve_release_path(args.stage1_save_path) or (RELEASE_ROOT / "outputs" / "localization")
    model_path = resolve_release_path(args.model_path)
    question_path = resolve_release_path(args.stage1_question_path)
    image_path = resolve_release_path(args.stage1_image_path)

    sys.path.insert(0, str(STAGE1_DIR))
    run_localization = importlib.import_module("run_localization")
    stage1_args = argparse.Namespace(
        model=args.stage1_model,
        task=args.stage1_task,
        method=args.stage1_method,
        model_id=str(model_path),
        question_path=str(question_path),
        image_path=str(image_path),
        save_path=str(save_root),
        total_chunks=args.total_chunks,
        chunk_id=args.chunk_id,
        max_iters=args.max_iters,
        flush_every=args.flush_every,
        seed=args.seed,
        hr_question_mode=args.hr_question_mode,
    )
    print("[stage1] running in-process")
    return run_localization.run(stage1_args, return_model=True)


def default_answers_file(model_path: str, benchmark: str, bbox_file: Path) -> Path:
    stage1_name = bbox_file.parent.name or "stage1-output"
    return (
        RELEASE_ROOT
        / "outputs"
        / "answers"
        / benchmark
        / f"{Path(model_path).name}-{stage1_name}_bbox"
        / "zoom_eye.jsonl"
    )


def run_stage2(args: argparse.Namespace, bbox_file: Path) -> Path:
    model_path = resolve_release_path(args.model_path)
    annotation_path = resolve_release_path(args.annotation_path)
    answers_file = (
        resolve_release_path(args.answers_file)
        if args.answers_file
        else default_answers_file(str(model_path), args.benchmark, bbox_file)
    )
    bbox_file = resolve_release_path(str(bbox_file)) or bbox_file
    answers_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python,
        "ZoomEye/eval/perform_zoom_eye.py",
        "--model-path",
        str(model_path),
        "--annotation_path",
        str(annotation_path),
        "--benchmark",
        args.benchmark,
        "--bbox-file",
        str(bbox_file),
        "--answers-file",
        str(answers_file),
        "--model-device",
        args.model_device,
        "--num-chunks",
        str(args.num_chunks),
        "--chunk-idx",
        str(args.chunk_idx),
    ]
    if args.device_map:
        cmd.extend(["--device-map", args.device_map])
    if args.load_in_8bit:
        cmd.append("--load-in-8bit")

    run_command(cmd, cwd=STAGE2_DIR, gpu=args.gpu, dry_run=args.dry_run)
    print(f"[answers] {answers_file}")
    return answers_file


def run_stage2_inprocess(args: argparse.Namespace, bbox_file: Path, model_bundle=None) -> Path:
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    model_path = resolve_release_path(args.model_path)
    annotation_path = resolve_release_path(args.annotation_path)
    answers_file = (
        resolve_release_path(args.answers_file)
        if args.answers_file
        else default_answers_file(str(model_path), args.benchmark, bbox_file)
    )
    bbox_file = resolve_release_path(str(bbox_file)) or bbox_file
    answers_file.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(STAGE2_DIR))
    perform_zoom_eye = importlib.import_module("ZoomEye.eval.perform_zoom_eye")
    stage2_args = argparse.Namespace(
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx,
        answers_file=str(answers_file),
        split_num=args.split_num,
        model_path=str(model_path),
        annotation_path=str(annotation_path),
        benchmark=args.benchmark,
        direct_answer=False,
        bbox_file=str(bbox_file),
        allow_zoomeye_search=False,
        model_device=args.model_device,
        device_map=args.device_map,
        load_in_8bit=args.load_in_8bit,
        coco_root=str(resolve_release_path(args.coco_root) or args.coco_root),
    )
    print("[stage2] running in-process with the stage1 model")
    result = perform_zoom_eye.run(stage2_args, model_bundle=model_bundle)
    print(f"[answers] {result}")
    return result


def build_parser(config: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run", choices=["both", "stage1", "stage2"], default="both")
    parser.add_argument("--python", default=cfg_get(config, "runtime", "python", "python"))
    parser.add_argument("--gpu", default=cfg_get(config, "runtime", "gpu", "0"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval", action="store_true", help="Deprecated no-op. Use evaluate_activescope.py to compute metrics.")

    parser.add_argument("--model-path", default=cfg_get(config, "model", "path", "models/qwen3vl"))
    parser.add_argument("--benchmark", default=cfg_get(config, "stage2", "benchmark", "Vstar"))
    parser.add_argument("--annotation-path", default=cfg_get(config, "stage2", "annotation_path", "data/zoom_eye/zoom_eye_data"))
    parser.add_argument("--bbox-file", default=cfg_get(config, "stage2", "bbox_file", None))
    parser.add_argument("--answers-file", default=cfg_get(config, "stage2", "answers_file", None))
    parser.add_argument("--model-device", default=cfg_get(config, "stage2", "model_device", "cuda:0"))
    parser.add_argument("--device-map", default=cfg_get(config, "stage2", "device_map", None))
    parser.add_argument("--load-in-8bit", action="store_true", default=cfg_get(config, "stage2", "load_in_8bit", False))
    parser.add_argument("--split-num", type=int, default=cfg_get(config, "stage2", "split_num", 4))
    parser.add_argument("--coco-root", default=cfg_get(config, "stage2", "coco_root", "data/ms_coco"))

    parser.add_argument("--stage1-question-path", default=cfg_get(config, "stage1", "question_path", "data/zoom_eye/zoom_eye_data/Vstar/annotation_Vstar.json"))
    parser.add_argument("--stage1-image-path", default=cfg_get(config, "stage1", "image_path", "data/zoom_eye/zoom_eye_data/Vstar"))
    parser.add_argument("--stage1-save-path", default=cfg_get(config, "stage1", "save_path", "outputs/localization"))
    parser.add_argument("--stage1-model", default=cfg_get(config, "stage1", "model", "qwen3"))
    parser.add_argument("--stage1-task", default=cfg_get(config, "stage1", "task", "vstar"))
    parser.add_argument("--stage1-method", default=cfg_get(config, "stage1", "method", "rel_att"))
    parser.add_argument("--max-iters", type=int, default=cfg_get(config, "stage1", "max_iters", 5))
    parser.add_argument("--flush-every", type=int, default=cfg_get(config, "stage1", "flush_every", 10))
    parser.add_argument("--hr-question-mode", choices=["random-option", "question"], default=cfg_get(config, "stage1", "hr_question_mode", "random-option"))
    parser.add_argument("--seed", type=int, default=cfg_get(config, "stage1", "seed", 0))

    parser.add_argument("--total-chunks", type=int, default=cfg_get(config, "chunks", "total_chunks", 1))
    parser.add_argument("--chunk-id", type=int, default=cfg_get(config, "chunks", "chunk_id", 0))
    parser.add_argument("--num-chunks", type=int, default=cfg_get(config, "chunks", "num_chunks", 1))
    parser.add_argument("--chunk-idx", type=int, default=cfg_get(config, "chunks", "chunk_idx", 0))
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args(argv)
    config = load_config(config_args.config)

    args = build_parser(config).parse_args(argv)
    bbox_file = resolve_release_path(args.bbox_file) if args.bbox_file else None

    if args.eval:
        print("[eval] --eval is deprecated and ignored. Run ActiveScope/evaluate_activescope.py on the answer file.")

    if args.run == "both" and not args.dry_run:
        bbox_file, model, processor = run_stage1_inprocess(args)
        run_stage2_inprocess(args, bbox_file, model_bundle=(model, processor))
        return

    if args.run in ["both", "stage1"]:
        bbox_file = run_stage1(args)

    if args.run in ["both", "stage2"]:
        if bbox_file is None:
            raise ValueError("--bbox-file is required when running stage2 without stage1")
        run_stage2(args, bbox_file)


if __name__ == "__main__":
    main()
