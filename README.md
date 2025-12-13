# GRPO Training Demo for Gemma 2

This repository demonstrates training the **Gemma 2 2B-IT** model on the **GSM8K math reasoning benchmark** using **Group Relative Policy Optimization (GRPO)**.

## What is GRPO?

[GRPO (Group Relative Policy Optimization)](https://arxiv.org/pdf/2402.03300) is an RL algorithm designed to enhance the reasoning abilities of LLMs. It is a variant of PPO that reduces memory usage by eliminating the need for a separate value function model.

## Two Versions Available

| Version | File | Hardware | Framework |
|---------|------|----------|-----------|
| **TPU** | `grpo-demo-local.ipynb` | TPU v5e-8 | Google Tunix + JAX |
| **GPU** | `grpo-demo-gpu.ipynb` | NVIDIA GPU (24GB+) | HuggingFace TRL + PyTorch |

## Quick Start

### GPU Version (Recommended for most users)

1. **Requirements**
   - NVIDIA GPU with 24GB+ VRAM (RTX 3090/4090, A100, H100)
   - CUDA 11.8+
   - Python 3.10+

2. **Setup**
   ```bash
   pip install -r requirements_gpu.txt
   ```

3. **Get Hugging Face Token**
   - Go to https://huggingface.co/settings/tokens
   - Create a new token
   - Accept Gemma license at https://huggingface.co/google/gemma-2-2b-it

4. **Run**
   - Open `grpo-demo-gpu.ipynb`
   - Fill in `HF_TOKEN`
   - Run all cells

### TPU Version

1. **Requirements**
   - Google Cloud TPU v5e-8
   - Kaggle account with Gemma access

2. **Setup**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get Kaggle Credentials**
   - Go to https://www.kaggle.com/settings
   - Create New API Token
   - Accept Gemma license at https://www.kaggle.com/models/google/gemma-2

4. **Run**
   - Open `grpo-demo-local.ipynb`
   - Fill in `KAGGLE_USERNAME` and `KAGGLE_KEY`
   - Run all cells

## Training Details

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| LoRA Rank | 64 | Low-rank adaptation dimension |
| LoRA Alpha | 64 | LoRA scaling factor |
| Learning Rate | 3e-6 | AdamW learning rate |
| Num Generations | 4 | Responses per prompt (G in GRPO) |
| Beta | 0.08 | KL penalty coefficient |
| Epsilon | 0.2 | PPO clipping parameter |

### Reward Functions

The model is trained with 4 reward signals:
1. **Format Exact Match** (+3.0): Output matches `<reasoning>...</reasoning><answer>...</answer>`
2. **Format Approximate** (+/-0.5): Partial format matching
3. **Answer Correctness** (+3.0 to -1.0): Based on numerical accuracy
4. **Number Extraction** (+1.5): Correct number in answer tags

### Expected Results

After training on GSM8K:
- Answer Accuracy: ~50-60%
- Format Accuracy: ~80-90%

## File Structure

```
tpu/
├── grpo-demo-gemma2-2b.ipynb   # Original TPU version (Kaggle)
├── grpo-demo-local.ipynb       # TPU local version
├── grpo-demo-gpu.ipynb         # GPU version
├── requirements.txt            # TPU dependencies
├── requirements_gpu.txt        # GPU dependencies
└── README.md
```

## Hardware Requirements

### GPU Version
| GPU | VRAM | 4-bit | 8-bit | Full |
|-----|------|-------|-------|------|
| RTX 3090/4090 | 24GB | OK | Tight | OOM |
| A100 40GB | 40GB | OK | OK | Tight |
| A100 80GB | 80GB | OK | OK | OK |
| H100 | 80GB | OK | OK | OK |

### TPU Version
- TPU v5e-8: ~$1.20/hour on GCP
- Estimated training time: 30-50 hours for full dataset

## References

- [GRPO Paper](https://arxiv.org/pdf/2402.03300)
- [GSM8K Dataset](https://huggingface.co/datasets/openai/gsm8k)
- [Gemma 2 Model](https://huggingface.co/google/gemma-2-2b-it)
- [TRL Library](https://huggingface.co/docs/trl)
- [Google Tunix](https://github.com/google/tunix)

## License

This project is for educational purposes. Model weights are subject to Google's Gemma license.

