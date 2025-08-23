# LLM News Summarizer (RU)

Abstractive summarization of Russian news powered by open LLMs. The project explores inference-only (zero-/few-shot) and lightweight fine-tuning (QLoRA, DPO), with a simple Gradio demo.

## Highlights
- **Task:** Abstractive summarization of Russian news (neutral tone, ~3–5 sentences).
- **Models:** https://huggingface.co/Qwen/Qwen2-7B-Instruct (GPU) and https://huggingface.co/Qwen/Qwen2-0.5B-Instruct (CPU).
- **Training:** Supervised fine-tuning with **QLoRA** (https://github.com/huggingface/peft); preference alignment with **DPO** (https://github.com/huggingface/trl).
- **Baselines:** Extractive **Lead-3**.
- **Metrics:** ROUGE-1/2/L (https://github.com/google-research/google-research/tree/master/rouge) and BERTScore (https://github.com/Tiiiger/bert_score).
- **Demo:** Gradio UI (https://www.gradio.app/).

## Dataset
- **Gazeta** pairs (article → reference summary): https://huggingface.co/datasets/IlyaGusev/gazeta
- Reproducible `train/dev/test` splits (config in `config/dataset.ids.yml`).

## Results (validation)
| Setting | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|---|---:|---:|---:|---:|
| Lead-3 (extractive baseline) | 0.218 | 0.066 | 0.210 | 0.877 |
| Zero-shot (Qwen2-7B, CUDA) | 0.253 | 0.096 | 0.246 | 0.875 |
| Few-shot (Qwen2-7B, CUDA) | 0.240 | 0.087 | 0.235 | 0.875 |
| SFT (QLoRA on Qwen2-7B, CUDA) | 0.250 | 0.099 | 0.242 | 0.874 |
| SFT + DPO (Qwen2-7B, CUDA) | 0.235 | 0.088 | 0.226 | 0.873 |

- **vs. Lead-3**, Zero-shot (Qwen2-7B) improves ROUGE-L by **+0.035** (**16.8%**).
- **vs. Lead-3**, SFT (QLoRA) improves ROUGE-2 by **+0.033** (**50.1%**).

> Exact numbers and per-run CSVs are available in `metrics/` (aggregated in `metrics/rezult_scores.csv`).

## Project Structure
```
.
├── notebooks/
│   ├── 01_prepare_gazeta.ipynb           # Download & clean dataset; build splits
│   ├── 02_baseline_extractive.ipynb      # Lead-3 baseline
│   ├── 03_cpu_llm_zero_shot.ipynb        # Zero-shot on CPU (Qwen2-0.5B)
│   ├── 03_cuda_llm_zero_shot.ipynb       # Zero-shot on GPU (Qwen2-7B)
│   ├── 04_cpu_llm_few_shot.ipynb         # Few-shot on CPU
│   ├── 04_cuda_llm_few_shot.ipynb        # Few-shot on GPU
│   ├── 05_cuda_llm_qlora.ipynb           # SFT with QLoRA (PEFT)
│   ├── 06_cuda_llm_dpo.ipynb             # Preference optimization (DPO, TRL)
│   ├── 07_cuda_gradio.ipynb              # Gradio demo
│   └── 08_result_metrics.ipynb           # Metrics aggregation/report
├── src/
│   └── utils/                            # Data cleaning, metrics, simple baselines (Lead-3)
├── metrics/                              # Per-run metrics + `rezult_scores.csv`
├── docs/
│   ├── gradio_screen/                    # UI screenshots (not tracked in this archive)
│   └── samples/                          # Qualitative examples (TSV)
├── config/
│   ├── dataset.ids.yml                   # Paths to split ID files
│   └── models.params.yml                 # Model IDs and runtime settings
├── environment.yml                       # Conda env (PyTorch / Transformers / PEFT / TRL / Gradio)
├── .env.example                          # Paths: DATA_DIR, MODELS_DIR, etc.
└── README.md
```

## Quickstart

### 1) Environment
```bash
conda env create -f environment.yml
conda activate llm-news
```

### 2) Local paths
Copy `.env.example` → `.env` and set:
```dotenv
EXTERNAL_STORAGE_DIR=/abs/path/with/space
DATA_DIR=./data
MODELS_DIR=./models
```

### 3) Run in notebooks (recommended order)
1. `01_prepare_gazeta.ipynb`
2. `02_baseline_extractive.ipynb`
3. `03_*_llm_zero_shot.ipynb` → quick inference on CPU/GPU
4. `04_*_llm_few_shot.ipynb` → in-context examples
5. `05_cuda_llm_qlora.ipynb` → supervised fine-tuning (QLoRA)
6. `06_cuda_llm_dpo.ipynb` → preference alignment (DPO)
7. `07_cuda_gradio.ipynb` → launch the demo UI
8. `08_result_metrics.ipynb` → aggregate & export metrics

> GPU is recommended for 7B inference/fine-tuning. CPU runs use the 0.5B model.

## Gradio Demo
- Launch from `notebooks/07_cuda_gradio.ipynb`.
- Screenshots are in `docs/gradio_screen/`.

## License
MIT License. See `LICENSE`.

## Acknowledgments
- Qwen models: https://huggingface.co/Qwen/Qwen2-7B-Instruct, https://huggingface.co/Qwen/Qwen2-0.5B-Instruct
- Hugging Face Transformers: https://github.com/huggingface/transformers
- PEFT / QLoRA: https://github.com/huggingface/peft
- TRL / DPO: https://github.com/huggingface/trl
- Dataset: Gazeta (https://huggingface.co/datasets/IlyaGusev/gazeta)
