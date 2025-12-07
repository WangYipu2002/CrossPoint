# Towards Cross-View Point Correspondence in Vision-Language Models

This repository provides a comprehensive framework for evaluating VLMs on the CrossPoint-Bench benchmark and training CroPond.

[![arXiv](https://img.shields.io/badge/arXiv-2512.04686-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.04686)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-CrossPoint--Bench-yellow.svg)](https://huggingface.co/datasets/WangYipu2002/CrossPoint-Bench)

<p align="center">
  <img src="assets/teaser.png" width="100%">
</p>

## 📋 TODO
- [x] Release CrossPoint-Bench
- [ ] Release CrossPoint-378K
- [x] Release CroPond model

## 🤖 Models

We provide two versions of CroPond:

| Model | Base Model | Parameters | HuggingFace |
|-------|------------|------------|-------------|
| CroPond-3B | Qwen2.5-VL-2B | 3B | [🤗 WangYipu2002/CroPond-3B](https://huggingface.co/WangYipu2002/CroPond-3B) |
| CroPond-7B | Qwen2.5-VL-7B | 7B | [🤗 WangYipu2002/CroPond-7B](https://huggingface.co/WangYipu2002/CroPond-7B) |


## 🚀 Setup

Clone the repository:
```bash
git clone https://github.com/WangYipu2002/CrossPoint.git
cd CrossPoint
```

### 📦 Install Dependencies

**For evaluation:**
```bash
conda create -n crosspoint_eval python=3.10
conda activate crosspoint_eval
pip install -r requirements_eval.txt
```

**For training:**
```bash
conda create -n crosspoint_train python=3.10
conda activate crosspoint_train
pip install -r requirements_train.txt
```


## 📊 Evaluation on CrossPoint-Bench

The evaluation process consists of three steps:

### Step 1: Download CrossPoint-Bench

Download CrossPoint-Bench from [Hugging Face](https://huggingface.co/datasets/WangYipu2002/CrossPoint-Bench).

After downloading, the directory structure should look like:
```
CrossPoint-Bench/
├── image/                     # Contains all benchmark images
│   ├── origin_image/          # Original scene images
│   └── visual_image/          # Annotated visualization images
└── CrossPoint-Bench.jsonl     # Benchmark annotations
```

### Step 2: Run Inference

Choose one of the following methods based on your model type:

#### Option A: API-based Models (GPT, Claude, Gemini)

Edit `scripts/eval/eval_api.sh` to configure your paths, model names, and API credentials.

Run:
```bash
bash scripts/eval/eval_api.sh
```

#### Option B: Open-source Models/Trained Models

Edit `scripts/eval/eval_opensource.sh` to configure your paths and model settings.

Run:
```bash
bash scripts/eval/eval_opensource.sh
```

**Output**: Inference results will be saved to `eval_results/inference/eval_<model_name>.jsonl`


### Step 3: Calculate Metrics

Edit `scripts/eval/cal_metric.sh` to configure paths and coordinate format settings.

Run:
```bash
bash scripts/eval/cal_metric.sh
```

**Output**: Metrics will be saved to `eval_results/scores/evaluation_summary.xlsx`


## 🏋️ Training

### Step 1: Download Training Datasets

In this repository, we primarily use [CrossPoint-378K](https://huggingface.co/datasets/WangYipu2002/CrossPoint-378K), which can be downloaded from Hugging Face. 


To enhance the model's spatial understanding capabilities while maintaining its general knowledge, we also incorporate other datasets including [RefSpatial](https://huggingface.co/datasets/JingkunAn/RefSpatial), [SAT](https://huggingface.co/datasets/array/SAT), [SPAR-7M](https://huggingface.co/datasets/jasonzhango/SPAR-7M), [MulSeT](https://huggingface.co/datasets/WanyueZhang/MulSeT) and [LLaVA-1.5](https://arxiv.org/abs/2310.03744). Please refer to the original papers and repositories for dataset preparation instructions.

### Step 2: Training

Edit `scripts/train/train.sh` to configure your actual paths and register your training data paths in `data/dataset_info.json`.

```bash
bash scripts/train/train.sh
```

**Output**: Checkpoints will be saved to `OUTPUT_PATH`

After training, you can evaluate the trained checkpoint on CrossPoint-Bench and other benchmarks using the evaluation scripts described above.



## 🙏 Acknowledgment
This repository is built upon the codebase of [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory).

We acknowledge [ScanNet](http://www.scan-net.org/), [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/), [ETH3D](https://www.eth3d.net/), and [ARKitScenes](https://github.com/apple/ARKitScenes) for their data.

## 📝 Citation
If you find CrossPoint-Bench, CrossPoint-378K, and CroPond useful for your research, please cite using this BibTeX:

```bibtex
@article{wang2025crosspoint,
  title={Towards Cross-View Point Correspondence in Vision-Language Models},
  author={Wang, Yipu and Ji, Yuheng and Liu, Yuyang and Zhou, Enshen and Yang, Ziqiang and Tian, Yuxuan and Qin, Ziheng and Liu, Yue and Tan, Huajie and Chi, Cheng and Ma, Zhiyuan and Zeng, Daniel Dajun and Zheng, Xiaolong},
  journal={arXiv preprint arXiv:2512.04686},
  year={2025}
}
```