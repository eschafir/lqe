# HIVE-to-V-SPLADE Distillation Framework

This repository implements a **Teacher-Student Distillation Framework** that compiles the multi-step online reasoning loop of a hypothesis-driven retriever (**HIVE**) into a static, query-encoding-free visual sparse index (**V-SPLADE**) offline. 

By supervising the student sparse encoder with reasoning-aware gating targets and listwise rank margins, we embed query-time reasoning directly into visual document term expansions, enabling complex search on a standard CPU-only inverted index in under 5ms.

---

## Key Features

1. **Teacher-Student Distillation:** Compiles multi-step LLM reasoning rationales (Stage 4) and compensatory query expansions (Stage 2) into a static student Query Lookup Table (LUT).
2. **Hybrid Sparse-Dense Channel:** Combines V-SPLADE lexical precision with dense retrieval (BGE / ColBERT) to resolve zero-shot domain mismatches.
3. **Pseudo-Relevance Feedback (PRF):** A two-pass CPU-only query expansion loop that leverages initial top-K document activations to bridge out-of-vocabulary (OOV) terms on unseen splits.

---

## Installation & Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   NVIDIA_API_KEY="your_nvidia_nim_api_key_here"
   ```

---

## Usage Guide

The pipeline is split into three phases:

### 1. Teacher Target Generation (Offline)
Generate reasoning rationales and relevance scores from the HIVE teacher:
```bash
python src/distillation/generate_teacher_data.py \
  --dataset mm-bright/MM-BRIGHT \
  --split apple,aviation,bioacoustics,bioinformatics,bitcoin \
  --num-queries 50 \
  --output-path results/mm_bright_train_teacher_targets_filtered.json
```

### 2. Student Distillation
Train the student Query Lookup Table (LUT) on the generated targets:
```bash
python src/distillation/train_student.py \
  --targets-path results/mm_bright_train_teacher_targets_filtered.json \
  --checkpoint-path results/vsplade_student_checkpoint.pt
```

### 3. Hybrid Evaluation
Evaluate the distilled student hybrid pipeline on held-out test splits (e.g. `law`):
```bash
python src/distillation/evaluate_hybrid.py \
  --checkpoint-path results/vsplade_student_checkpoint_32b.pt \
  --dense-model BAAI/bge-large-en-v1.5 \
  --dense-mode bi-encoder \
  --test-splits law \
  --use-prf \
  --prf-k 3 \
  --prf-alpha 0.3
```

---

## Repository Structure

* `src/`: Core Python source modules.
  * `src/models.py`: Helper functions for loading local/API models.
  * `src/distillation/generate_teacher_data.py`: HIVE online target generation.
  * `src/distillation/train_student.py`: Student Query LUT fine-tuning.
  * `src/distillation/evaluate_hybrid.py`: Hybrid sparse-dense evaluator with PRF.
* `results/`: Directory containing checkpoints, teacher targets, and evaluation logs.
* `docs/`: LaTeX papers, manuscripts, and slides outlining the theoretical framework.