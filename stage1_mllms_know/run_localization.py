#!/usr/bin/env python3
"""Stage 1 localization for the release two-stage pipeline.

Supported paths:
- qwen3 + vstar
- internvl + hr_4k/hr_8k

The output schema matches the original mllms_know JSON: each sample keeps its
annotation fields and receives a `bboxes` dict keyed by extracted target.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

from qwen2_5_methods import target_iterate_attention_qwen2_5
from qwen3_new import apply_qwen3vl_attention_hooks
from utils import load_image


RELEASE_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_IDS = {
    "qwen3": str(RELEASE_ROOT / "models" / "qwen3vl"),
    "internvl": str(RELEASE_ROOT / "models" / "internvl2_5_4b"),
}

DEFAULT_QUESTION_PATHS = {
    "vstar": str(RELEASE_ROOT / "data" / "zoom_eye" / "zoom_eye_data" / "Vstar" / "annotation_Vstar.json"),
    "hr_4k": str(RELEASE_ROOT / "data" / "zoom_eye" / "zoom_eye_data" / "hr-bench_4k" / "annotation_hr-bench_4k.json"),
    "hr_8k": str(RELEASE_ROOT / "data" / "zoom_eye" / "zoom_eye_data" / "hr-bench_8k" / "annotation_hr-bench_8k.json"),
}

DEFAULT_IMAGE_PATHS = {
    "vstar": str(RELEASE_ROOT / "data" / "zoom_eye" / "zoom_eye_data" / "Vstar"),
    "hr_4k": str(RELEASE_ROOT / "data" / "zoom_eye" / "zoom_eye_data" / "hr-bench_4k"),
    "hr_8k": str(RELEASE_ROOT / "data" / "zoom_eye" / "zoom_eye_data" / "hr-bench_8k"),
}

GENERAL_QUESTION = "Write a general description of the image."


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_annotations(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            return list(json.load(f))
    return list(load_dataset(path)["test"])


def resolve_image_path(item: dict[str, Any], image_root: str, task: str) -> str:
    if task == "vstar":
        image_name = item.get("image") or item.get("input_image") or item.get("image_path")
    else:
        image_name = item.get("input_image") or item.get("image") or item.get("image_path")
    if image_name is None:
        raise KeyError("annotation has no image/input_image/image_path field")

    image_path = Path(str(image_name))
    if image_path.is_absolute():
        return str(image_path)
    return str(Path(image_root) / image_path)


def sample_key(item: dict[str, Any], task: str) -> Any:
    if task == "vstar" and "question_id" in item:
        try:
            return int(item["question_id"])
        except Exception:
            return item["question_id"]
    if "index" in item:
        try:
            return int(item["index"])
        except Exception:
            return item["index"]
    return item.get("image_path")


def build_question(item: dict[str, Any], task: str, hr_question_mode: str) -> str:
    if task == "vstar":
        return item.get("text") or item.get("question")

    if task in {"hr_4k", "hr_8k"}:
        question = item["question"]
        if hr_question_mode == "random-option":
            options = item.get("options") or []
            if options:
                option = random.choice(options)
                question = question + "\n" + option + "Answer with the option's letter from the given choices directly."
        return question

    raise ValueError(f"Unsupported task: {task}")


def load_qwen3(model_id: str):
    if "32B" in model_id or "30B" in model_id:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            load_in_4bit=True,
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    processor = AutoProcessor.from_pretrained(model_id)
    apply_qwen3vl_attention_hooks(model)
    return model, processor


def load_internvl(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()
    return model, tokenizer


def load_model(model_name: str, model_id: str):
    if model_name == "qwen3":
        return load_qwen3(model_id)
    if model_name == "internvl":
        return load_internvl(model_id)
    raise ValueError(f"Unsupported model: {model_name}")


def localize(
    model_name: str,
    image: Image.Image,
    question: str,
    image_path: str,
    model,
    processor,
    max_iters: int,
) -> dict[str, Any]:
    prompt = f"{question} Answer the question using a single word or phrase."
    general_prompt = f"{GENERAL_QUESTION} Answer the question using a single word or phrase."

    if model_name == "qwen3":
        return target_iterate_attention_qwen2_5(
            image,
            prompt,
            general_prompt,
            model,
            processor,
            max_iters=max_iters,
            image_path=image_path,
        )

    if model_name == "internvl":
        from internvl2_5_new import target_iterate_attention_internvl

        pixel_values = load_image(image, max_num=12, use_anyres=True).to(torch.bfloat16).cuda()
        return target_iterate_attention_internvl(
            image,
            pixel_values,
            prompt,
            general_prompt,
            model,
            processor,
            image_size=image.size,
            max_iters=max_iters,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def run(
    args: argparse.Namespace,
    model_bundle: tuple[Any, Any] | None = None,
    return_model: bool = False,
):
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    save_root = Path(args.save_path)
    if not save_root.is_absolute():
        save_root = Path.cwd() / save_root

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = save_root / f"{args.model}_{args.task}_{args.method}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.model}-{args.task}-{args.method}.json"

    model_id = args.model_id or DEFAULT_MODEL_IDS[args.model]
    question_path = args.question_path or DEFAULT_QUESTION_PATHS[args.task]
    image_root = args.image_path or DEFAULT_IMAGE_PATHS[args.task]

    data = load_annotations(question_path)
    chunks = np.array_split(data, args.total_chunks)
    current = list(chunks[args.chunk_id])
    if model_bundle is None:
        model, processor = load_model(args.model, model_id)
    else:
        model, processor = model_bundle

    results: list[dict[str, Any]] = []

    for index, raw_item in enumerate(tqdm(current, desc=f"{args.model} {args.task}", ncols=100), start=1):
        item = dict(raw_item)
        try:
            image_path = resolve_image_path(item, image_root, args.task)
            question = build_question(item, args.task, args.hr_question_mode)
            if not question:
                print(f"skip item {index}: missing question")
                continue

            image = Image.open(image_path).convert("RGB")
            bboxes = localize(args.model, image, question, image_path, model, processor, args.max_iters)

            item["image_path"] = image_path
            item["stage1_question"] = question
            item["stage1_key"] = sample_key(item, args.task)
            item["bboxes"] = bboxes
            item["bbox"] = None
            item.pop("image", None)
            results.append(item)
        except Exception as exc:
            print(f"err item {index}: {exc}")

        if index % args.flush_every == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, default=json_default)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, default=json_default)

    print("[stage1 bbox]", output_path)
    if return_model:
        return output_path, model, processor
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "internvl"], default="qwen3")
    parser.add_argument("--task", choices=["vstar", "hr_4k", "hr_8k"], default="vstar")
    parser.add_argument("--method", default="rel_att")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--question-path", default=None)
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--save-path", default="./playground/data/results")
    parser.add_argument("--total-chunks", type=int, default=1)
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hr-question-mode", choices=["random-option", "question"], default="random-option")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
