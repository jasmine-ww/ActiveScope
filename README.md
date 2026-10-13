# ActiveScope

**ActiveScope: Actively Seeking and Correcting Perception for MLLMs**

Official implementation of the ICML 2026 paper **[ActiveScope: Actively Seeking and Correcting Perception for MLLMs](https://proceedings.mlr.press/v306/wang26h.html)**. See also the [arXiv version](https://arxiv.org/abs/2606.24292).

ActiveScope is a training-free framework for improving fine-grained perception in high-resolution image understanding. It actively seeks task-relevant visual evidence and corrects failed localization before answering.


## Method Overview

ActiveScope is implemented as a two-stage inference pipeline:

- **Semantic Anchor Localization (SAL)** extracts fine-grained semantic anchors from the question and localizes each key target independently using MLLM attention.
- **Interference-Suppressed Refinement (ISR)** iteratively verifies each localized crop with `Does the target exist?` and suppresses attention on failed or distracting regions to refine localization.
- **Region-aware answering** feeds the original image plus the predicted target regions into the MLLM answer module.

The current release includes the settings used for V* Bench and HR-Bench experiments.


## Quick Start

Install dependencies:

```bash
cd ActiveScope
pip install -r requirements.txt
```

Download models and datasets, then place or symlink them under the relative paths below.

```text
ActiveScope/
  models/
    qwen3vl/
    internvl2_5_4b/
  data/
    zoom_eye/
      zoom_eye_data/
        Vstar/
          annotation_Vstar.json
          direct_attributes/
          relative_position/
        hr-bench_4k/
          annotation_hr-bench_4k.json
          image/
        hr-bench_8k/
          annotation_hr-bench_8k.json
          image/
```

The release does not include benchmark images, benchmark annotations, or model weights. Users should download them separately. The config files use these relative paths by default, and the runner resolves them relative to `ActiveScope/`.

For V* Bench, `annotation_Vstar.json` is read directly from `data/zoom_eye/zoom_eye_data/Vstar/` by both stages. For HR-Bench, each split keeps its own annotation file directly under `hr-bench_4k/` or `hr-bench_8k/`, with images under the corresponding `image/` directory.

Run ActiveScope on V* Bench with an existing bbox file:

```bash
cd <repo-root>

python ActiveScope/run_activescope.py \
  --config ActiveScope/configs/activescope_vstar_qwen3.json \
  --run stage2 \
  --bbox-file ActiveScope/outputs/localization/<stage1-run>/qwen3-vstar-rel_att.json
```

Run the full ActiveScope pipeline on V* Bench:

```bash
python ActiveScope/run_activescope.py \
  --config ActiveScope/configs/activescope_vstar_qwen3.json \
  --run both
```

Evaluate an answer file:

```bash
python ActiveScope/evaluate_activescope.py \
  --benchmark Vstar \
  --answers-file ActiveScope/outputs/answers/Vstar/<run>/zoom_eye.jsonl
```

## HR-Bench Settings

HR-Bench uses `hr_4k` and `hr_8k` for SAL/ISR localization, and `hr-bench_4k` and `hr-bench_8k` for answer evaluation. The two stages are aligned by the annotation `index` field.

```bash
python ActiveScope/run_activescope.py \
  --config ActiveScope/configs/activescope_hr_bench_4k_internvl25_4b.json \
  --run both
```

```bash
python ActiveScope/run_activescope.py \
  --config ActiveScope/configs/activescope_hr_bench_8k_internvl25_4b.json \
  --run both
```

To answer with an existing HR-Bench bbox file, set `stage2.bbox_file` in the config or pass `--bbox-file`, then run `--run stage2`.

## Configured Experiments

- `configs/activescope_vstar_qwen3.json`: Qwen3-VL on V* Bench.
- `configs/activescope_hr_bench_4k_internvl25_4b.json`: InternVL2.5-4B on HR-Bench 4K.
- `configs/activescope_hr_bench_8k_internvl25_4b.json`: InternVL2.5-4B on HR-Bench 8K.

The model paths in these configs are local relative paths. If a user stores models elsewhere, pass `--model-path /path/to/model` or edit `model.path`.

## ZoomEye Integration

The answer stage is built on [ZoomEye](https://github.com/om-ai-lab/ZoomEye). In ActiveScope, stage 2 uses the bbox file produced by SAL/ISR and then asks the MLLM to answer with the original image plus the localized regions.

When `--bbox-file` is provided, the script skips the original ZoomEye search baseline and directly uses ActiveScope's predicted boxes. For debugging and analysis, each answer row stores:

- `active_scope_bbox_count`: how many ActiveScope boxes were used for this sample.
- `active_scope_bbox`: the box coordinates actually passed to the answer model.


## Outputs

Stage 1 writes localization results under:

```text
ActiveScope/outputs/localization/<model>_<task>_<method>_<timestamp>/<model>-<task>-<method>.json
```

If `stage2.answers_file` is `null`, stage 2 writes answers under:

```text
ActiveScope/outputs/answers/<benchmark>/<model>-<stage1-output-dir>_bbox/zoom_eye.jsonl
```

## Citation

If you find this repository helpful to your research, welcome to cite our paper:

```bibtex
@inproceedings{wang2026activescope,
  title = {ActiveScope: Actively Seeking and Correcting Perception for MLLMs},
  author = {Wang, Yajing and Bi, Chao and Sun, Junshu and Shen, Shufan and Qi, Zhaobo and Wang, Shuhui and Huang, Qingming},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year = {2026}
}
```
