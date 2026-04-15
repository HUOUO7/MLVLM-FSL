# MLVLM-FSL
The code repository for AAAI25 paper "**Making Large Vision Language Models to Be Good Few-Shot Learners**".

<img width="1039" height="542" alt="image" src="https://github.com/user-attachments/assets/b4eccd68-c62d-429d-a244-092eda1564ba" />

## 📋 Introduction
Few-shot classification (FSC) is challenging for Large Vision Language Models (LVLMs) due to insufficient learning from support data and severe **positional biases**. Our approach addresses these issues through:
* **Meta-learning Instruction Fine-tuning**: Teaching models "learn to learn" by constructing a rich set of meta-tasks for instruction fine-tuning.
* **Label Augmentation (LA)**: Implementing a character perturbation strategy during fine-tuning to ensure the model focuses on support information rather than relying on overconfident pre-training knowledge.
* **Candidate Selection (CS)**: An adaptive pipeline that generates and utilizes auxiliary attribute descriptions to filter unreliable candidates and simplify the task during inference.

## 📂 Data Preparation
The experiments utilize 13 datasets from ELEVATER for instruction tuning and 8 established benchmarks for testing.
* **Training Sets**: Covers scene recognition, general object recognition, remote sensing, and fine-grained classification.
* **Test Sets**: MiniImageNet (MINI), CIFAR-FS, Tiered-ImageNet, CUB, Stanford Dogs, FGVC-Aircraft, Oxford Flowers, and Stanford Cars.

## 🚀 Step-by-Step Guide

### Step 1: Build Meta-task Instructions
Generate the N-way K-shot instruction-following data for Qwen-VL.
* **Folder**: `build_data_instruction/`
* **Core Script**: `sample_base_novel_ele_qwen.py` (Converts datasets into the required format where query samples are aligned with candidate answers).

### Step 2: Instruction Fine-tuning
Perform QLoRA fine-tuning on the Qwen-VL-Chat-Int4 model.
* **Script**: `finetune_qlora_ds_cww.sh`
* **Key Components**: Uses **Label Augmentation (LA)** via character perturbation (e.g., Shuffle, Reversed, Random Insert) to break common token sequences.

### Step 3: Inference & Evaluation
Test the fine-tuned model's performance.
* **Folder**: `Test_Qwen_scripts/`
* **Action**: Execute `eva_ele_vanilia_3500_infer_fsl4_test_no_history.sh` for fine-tuned models.
* **Metric Calculation**: Use `cal_for_Qwen_using_CLIP.py` to calculate `Acc`, `Acc_occur`, and `Acc_clip`.

### Step 4: Attribute Description Generation
Generate high-quality visual attributes to assist the inference process.
* **Folder**: `attribute_text/`
* **Main Script**: `mllm_generate_text_v3.py` (Uses an adaptive framework to generate detailed attribute and global descriptions).

### Step 5: Candidate Selection (CS)
Simplify the classification task by filtering candidates based on semantic similarity.
* **Folder**: `attribute_filter/`
* **Workflow**: Execute scripts sequentially (1 -> 2 -> 3 -> 4).
* **Logic**: If the initial inference result is not within the top candidates, the task is reorganized for a final inference to boost reliability.

## 📊 Performance
Our method achieves State-of-the-Art performance on both general and fine-grained FSC benchmarks.

| Dataset | Ours (5-way 1-shot) | Gain over SOTA |
| :--- | :---: | :---: |
| MiniImageNet | **98.24%** | +2.02% |
| CIFAR-FS | **95.02%** | +5.08% |
| CUB | **96.40%** | +0.60% |
| Oxford Flowers | **99.58%** | +16.33% |
| Stanford Cars | **99.72%** | +11.02% |

## ✒️ Citation
```bibtex
@inproceedings{liu2025making,
  title={Making Large Vision Language Models to Be Good Few-Shot Learners},
  author={Liu, Fan and Cai, Wenwen and Huo, Jian and Zhang, Chuanyi and Chen, Delong and Zhou, Jun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
