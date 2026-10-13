import sys
import os
from pathlib import Path
# os.environ['CUDA_VISIBLE_DEVICES']='7'
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)
release_root = Path(__file__).resolve().parents[3]
from PIL import Image
import argparse
import json
import warnings
warnings.filterwarnings("ignore")
import torch
from tqdm import tqdm
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          Qwen2ForCausalLM)
# from ZoomEye.zoom_model import ZoomModelGlobalLocal, ZoomModelLocal
from ZoomEye.zoom_model_internvl import ZoomModelInternvl
from ZoomEye.zoom_model_qwenvl import ZoomModelQwenVL
from ZoomEye.zoom_eye import get_zoom_eye_response, get_direct_response, format_question,format_question_multichoice
from datetime import datetime
from PIL import Image, ImageDraw
import re


def judge_answer(text, choices):
    ALPHA_MAP = ["A", "B", "C", "D", "E", "F"]


    pattern = re.compile(r'\(([A-Za-z])\)')
    res = pattern.findall(text)
    if len(res) >= 1:
        pred = ord(res[-1].upper()) - ord('A')
        # pred = res[-1].upper()  # 'A', 'B', ...
    else:
        res = []
        for i, choice in enumerate(choices):
            if choice.lower() in text.lower():
                # res.append(ALPHA_MAP[i])
                res.append(i)
        if len(res) >= 1:
            pred = res[-1]
        else:
            for i, choice in enumerate(choices):
                text = re.sub(r'[\n.,!?]', ' ', text)
                if ALPHA_MAP[i] in text.split(" "):
                    # res.append(ALPHA_MAP[i])
                    res.append(i)
            if len(res) >= 1:
                pred = res[-1]
            else:
                for i, choice in enumerate(choices):
                    text = re.sub(r'[\n.,!?]', ' ', text)
                    if ALPHA_MAP[i].lower() in text.split(" "):
                        # res.append(ALPHA_MAP[i])
                        res.append(i)
                if len(res) >= 1:
                    pred = res[-1]
                else:
                    pred = 1   #"FAILED"
    return pred
                    
def build_lookup(lst):
    mp = {}
    for item in lst:
        key = None

        if 'index' in item:
            key = item['index']

        if key == None and 'Question_id' in item:
            key =  item['Question_id']
        if key == None and 'image_path' in item:
            key = item['image_path'].split('Vstar/')[1]

        if key == None and 'input_image' in item:
            try:
                # key = int(item['index'])
                key = item['input_image']
            except:
                key = item['index']
        if key == None and 'question_id' in item:
            try:
                key = int(item['question_id'])
            except:
                key = item['question_id']
        if key == None and 'question' in item and 'image_link' in item:
            key = item['question']+item['image_link']

        if key is not None:
            mp[key] = item
    return mp

def get_chunk(lst, n, k):
    subarrays = [[] for _ in range(n)]
    for i in range(n):
        subarrays[i] = lst[i::n]
    return subarrays[k]

def get_basename(path):
    return os.path.basename(path)

def get_annotation_key(annotation, benchmark):
    if benchmark == "Vstar":
        return annotation["input_image"]
    if benchmark in ["hr-bench_4k", "hr-bench_8k"]:
        return annotation["index"]
    if benchmark == "mme-realworld":
        return annotation["Question_id"]
    if benchmark == "docvqa":
        return annotation["questionId"]
    if benchmark == "vsr":
        return annotation["question"] + annotation["image_link"]
    raise ValueError(f"Unsupported benchmark for bbox lookup: {benchmark}")

def get_predicted_bboxes(annotation, bbox_map, benchmark):
    item = bbox_map.get(get_annotation_key(annotation, benchmark))
    if item is None:
        return []
    if "bboxes" not in item:
        bbox = item.get("bbox")
        return [bbox] if bbox else []

    bboxes = []
    for _, value in item["bboxes"].items():
        if value:
            bboxes.append(value[0])
    return bboxes

def answer_with_predicted_bboxes(zoom_model, annotation, bbox_map, benchmark, annotation_path, coco_root):
    bbox = get_predicted_bboxes(annotation, bbox_map, benchmark)
    annotation["active_scope_bbox_count"] = len(bbox)
    annotation["active_scope_bbox"] = bbox

    if benchmark == "vsr":
        question = annotation["question"].split(" \n")[0]
        options = ["Yes.", "No."]
        image_path = annotation["image_link"].split(".org/")[-1]
        image_path = os.path.join(coco_root, image_path)
        image = Image.open(image_path).convert("RGB")
        answer_type = "logits_match"
        annotation["input_image"] = image_path
        annotation["answer_type"] = answer_type
        annotation["options"] = options
        annotation["question"] = question
    else:
        question = annotation["question"]
        options = annotation.get("options", None)
        image_path = os.path.join(annotation_path, benchmark, annotation["input_image"])
        image = Image.open(image_path).convert("RGB")
        answer_type = annotation.get("answer_type", "free_form")

    if answer_type == "logits_match":
        with torch.no_grad():
            return zoom_model.multiple_choices_inference_bbox(image, question, options, bbox)
    if answer_type == "option_list":
        answers = []
        for option_str in options:
            question_input = format_question(question, option_str)
            with torch.no_grad():
                answers.append(zoom_model.free_form_using_bbox(image, question_input, bbox))
        return answers
    if answer_type == "Multiple Choice":
        question_input = format_question_multichoice(question, options)
        return zoom_model.free_form_using_bbox(image, question_input, bbox)

    return zoom_model.free_form_using_bbox(image, question, bbox)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--split-num", type=int, default=4)
    parser.add_argument("--model-path", type=str, default=str(release_root / "models" / "qwen3vl"))
    parser.add_argument("--annotation_path", type=str, default=str(release_root / "data" / "zoom_eye" / "zoom_eye_data"))
    parser.add_argument("--benchmark",  type=str, choices=["Vstar", "hr-bench_4k", "hr-bench_8k", "mme-realworld", "docvqa","vsr"], default="Vstar")
    parser.add_argument("--direct-answer", default=False, action="store_true")
    parser.add_argument("--bbox-file", type=str, default=None)
    parser.add_argument("--allow-zoomeye-search", default=False, action="store_true")
    parser.add_argument("--model-device", type=str, default="cuda:0")
    parser.add_argument("--device-map", type=str, default=None)
    parser.add_argument("--load-in-8bit", default=False, action="store_true")
    parser.add_argument("--coco-root", type=str, default=str(release_root / "data" / "ms_coco"))
    return parser


def build_zoom_model(args, model_bundle=None):
    model_path = args.model_path
    config_path = os.path.join(model_path, "config.json")
    config = json.load(open(config_path, "r"))

    if "internvl" in model_path.lower():
        kwargs = {}
        if model_bundle is not None:
            kwargs["model"], kwargs["tokenizer"] = model_bundle
        zoom_model = ZoomModelInternvl(model_path=model_path, device="cuda:0", torch_dtype=torch.bfloat16, patch_scale=1.2, **kwargs)
        def pop_limit_func(max_depth):
            return max_depth * 3
        search_kwargs = {
            "pop_limit": pop_limit_func,
            "threshold_descrease": [0.1, 0.1, 0.2],
            "answering_confidence_threshold_lower": -0.2,
            "answering_confidence_threshold_upper": 0.2,
            "visual_cue_threshold": 0.6,
        }
    elif "qwen" in model_path.lower():
        if args.load_in_8bit:
            kwargs = {"load_in_8bit": True}
        elif "32b" in model_path.lower() or "30b" in model_path.lower():
            kwargs = {"load_in_8bit": True}
        else:
            kwargs = {}
        if args.device_map is not None:
            kwargs["device_map"] = args.device_map
        if model_bundle is not None:
            kwargs["model"], kwargs["processor"] = model_bundle
        zoom_model = ZoomModelQwenVL(
            model_path=model_path,
            device=args.model_device,
            torch_dtype=torch.bfloat16,
            patch_scale=1.2,
            **kwargs,
        )
        def pop_limit_func(max_depth):
            return max_depth * 3
        search_kwargs = {
            "pop_limit": pop_limit_func,
            "threshold_descrease": [0.05, 0.1, 0.2],
            "answering_confidence_threshold_lower": 0,
            "answering_confidence_threshold_upper": 0.95,
            "visual_cue_threshold": 0.6,
        }
    else:
        raise ValueError(f"Model {model_path} not supported")
    return zoom_model, search_kwargs


def run(args, model_bundle=None):

    model_path = args.model_path
    annoataion_path = args.annotation_path
    benchmark = args.benchmark
    split_num = args.split_num

    if args.answers_file is None:
        answers_dir = f"ZoomEye/eval/answers/{benchmark}"

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        answers_dir = os.path.join(answers_dir, os.path.basename(args.model_path)+"-"+current_time)
        
        os.makedirs(answers_dir, exist_ok=True)
        answer_tag = "zoom_eye" if not args.direct_answer else "direct_answer"
        args.answers_file = os.path.join(answers_dir, f"{answer_tag}.jsonl")
        print(args.answers_file)

    if not args.direct_answer and args.bbox_file is None and not args.allow_zoomeye_search:
        raise ValueError(
            "ActiveScope stage2 requires --bbox-file so it can answer with SAL/ISR predicted boxes. "
            "Pass --allow-zoomeye-search only when you intentionally want the original ZoomEye search baseline."
        )

    zoom_model, search_kwargs = build_zoom_model(args, model_bundle=model_bundle)
    
    print(search_kwargs)

    decomposed_question_template = "What is the appearance of the {}?"  
    ic_examples_path = os.path.join(root_dir, "ZoomEye", "ic_examples", f"{benchmark}.json")
    m = json.load(open(os.path.join(annoataion_path, f"{benchmark}/annotation_{benchmark}.json"), "r"))
    m = get_chunk(m, args.num_chunks, args.chunk_idx)

    answers_dir = os.path.dirname(args.answers_file)
    if answers_dir:
        os.makedirs(answers_dir, exist_ok=True)

    bbox_map = None
    if args.bbox_file:
        datas_bbox = json.load(open(args.bbox_file, "r"))
        bbox_map = build_lookup(datas_bbox)
        print(f"[ActiveScope] loaded predicted bbox file: {args.bbox_file}")
        print(f"[ActiveScope] bbox lookup entries: {len(bbox_map)}")

    results_file = open(args.answers_file, "w")

    for annotation in tqdm(m):
        try:
            if args.direct_answer:
                if args.benchmark == "vsr":
                    question = annotation["question"].split(" \n")[0]
                    options = ["Yes.", "No."]
                    image_path = annotation["image_link"].split(".org/")[-1]
                    image_path = os.path.join(args.coco_root, image_path)
                    annotation["input_image"] = image_path
                    annotation["answer_type"] = "logits_match"
                    annotation["options"] = options
                    annotation["question"] = question
                    response = get_direct_response(
                        zoom_model=zoom_model,
                        annotation=annotation,
                        image_folder=None,
                    )
                else:
                    response = get_direct_response(
                        zoom_model=zoom_model,
                        annotation=annotation,
                        image_folder=os.path.join(annoataion_path, benchmark),
                    )
            elif bbox_map is not None:
                response = answer_with_predicted_bboxes(
                    zoom_model=zoom_model,
                    annotation=annotation,
                    bbox_map=bbox_map,
                    benchmark=benchmark,
                    annotation_path=annoataion_path,
                    coco_root=args.coco_root,
                )
            else:
                annotation["image"] = annotation["input_image"]
                response = get_zoom_eye_response(
                    zoom_model=zoom_model,
                    annotation=annotation,
                    ic_examples=json.load(open(ic_examples_path, "r")),
                    decomposed_question_template=decomposed_question_template,
                    image_folder=os.path.join(annoataion_path, benchmark),
                    split_num=split_num,
                    **search_kwargs,
                )
        except Exception as exc:
            print(f"err: {exc}")
            continue

        annotation["output"] = response
        results_file.write(json.dumps(annotation) + "\n")
        results_file.flush()

    results_file.close()
    return Path(args.answers_file)


if __name__ == "__main__":
    run(build_parser().parse_args())
