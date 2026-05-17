# CounterSeagas — Seq2Seq Dropout Value Experiment

A focused ablation study that trains an attentional LSTM Seq2Seq model for **English → Filipino machine translation** across three dropout values (0.3, 0.5, 0.7) and evaluates each with **SacreBLEU** and **METEOR**.

---

## Overview

This notebook implements and benchmarks a full sequence-to-sequence neural machine translation pipeline. The central experiment (Experiment 4) sweeps dropout rates to determine their effect on translation quality and generalisation, using a Tagalog/Filipino–English dataset from Hugging Face.

---

## Dataset

| Property | Value |
|---|---|
| Source | `rhyliieee/tagalog-filipino-english-translation` (Hugging Face) |
| Direction | English → Filipino |
| Split | 80% train / 10% validation / 10% test (auto-split if not pre-split) |
| Max train samples | 35,000 (experiment override) |
| Max val samples | 2,500 (experiment override) |
| Eval pairs used | Up to 200 test pairs |

Text is lowercased, HTML-unescaped, unicode-normalised, and filtered by token length and length ratio before training.

---

## Model Architecture

```text
Encoder  →  AdditiveAttention  →  Decoder  →  Linear projection
```

Both encoder and decoder are multi-layer LSTMs with additive (Bahdanau-style) attention connecting them.

| Hyperparameter | Value |
|---|---|
| Embedding dim | 384 |
| Hidden dim | 768 |
| Num LSTM layers | 3 (experiment override) |
| Attention | Additive (Bahdanau) |
| Decoding | Beam search (width 5, length penalty 0.7) |

---

## Experiment — Dropout Sweep

Three independent models are trained, one per dropout value, with all other hyperparameters held constant.

| Dropout | Applied to |
|---|---|
| 0.3 | Embedding dropout + inter-layer LSTM dropout |
| 0.5 | Embedding dropout + inter-layer LSTM dropout |
| 0.7 | Embedding dropout + inter-layer LSTM dropout |

**Training config (per run)**

| Setting | Value |
|---|---|
| Epochs | 15 |
| Batch size | 64 |
| Optimiser | Adam (lr=5e-4, weight_decay=1e-5) |
| Gradient clipping | 1.0 |
| Teacher forcing ratio | 0.5 |
| Mixed precision (AMP) | Yes (CUDA only) |
| `torch.compile` | Yes (default mode, if available) |

Each run saves its best checkpoint (lowest validation loss) to `/content/exp4_dropout_<value>.pt`.

---

## Evaluation Metrics

- **SacreBLEU** — corpus-level BLEU computed with the `sacrebleu` library.
- **METEOR** — average sentence-level METEOR via `nltk.translate.meteor_score`.

Both are computed on up to 200 test pairs using beam search decoding.

---

## Results (latest run)

| Dropout | Best Val Loss ↓ | SacreBLEU ↑ | METEOR ↑ | Total Time (min) |
|---:|---:|---:|---:|---:|
| 0.3 | **4.5099** | 9.5951 | 0.32430 | 52.0 |
| 0.5 | 4.5976 | **9.7267** | **0.33772** | 52.4 |
| 0.7 | 4.8004 | 7.1054 | 0.28490 | 51.6 |

Notes:
- Lowest validation loss occurs at dropout **0.3**.
- Best SacreBLEU and METEOR occur at dropout **0.5**.
- Highest dropout (**0.7**) performs worst across metrics, while runtime stays nearly constant.

---

## Outputs

| File | Contents |
|---|---|
| `data/exp4_dropout_summary.csv` | One row per dropout value: BLEU, METEOR, best val loss, timing |
| `data/exp4_dropout_history.csv` | Per-epoch train/val loss for every run |
| `/content/exp4_dropout_<value>.pt` | Best model checkpoint for each dropout value |

In this workspace, the latest exported copies are:
- `exp4_dropout_summary (4).csv`
- `exp4_dropout_history (1).csv`

The notebook also renders six matplotlib plots inline:
- Train loss vs. epoch (all dropout values overlaid)
- Validation loss vs. epoch
- Test BLEU by dropout (bar chart)
- Test METEOR by dropout (bar chart)
- Best validation loss by dropout (bar chart)
- Total training time by dropout (bar chart)

---

## Requirements

```text
torch
datasets
sacrebleu
nltk
tqdm
numpy
pandas
matplotlib
```

Install in Colab with:

```bash
pip -q install datasets nltk sacrebleu tqdm
```

NLTK resources (`wordnet`, `omw-1.4`) are downloaded automatically at runtime if missing.

---

## How to Run

1. Open the notebook in **Google Colab** (GPU runtime recommended).
2. Run all cells in order — data loading, preprocessing, model definition, and the dropout sweep are all self-contained.
3. Results are printed to the console and saved to `data/` automatically.

> **Note:** The sweep respects `max_train_time_sec` (default 3 hours). If the budget is exhausted mid-sweep, completed runs are still saved and exported.

---

## Project Structure

```text
CounterSeagas-Seq2seqDropoutValue.ipynb
│
├── Configuration
├── Preprocessing Utilities      # tokenisation, cleaning, filtering, vocabulary
├── Dataset Loader Utilities     # HuggingFace loading, split management
├── Model                        # Encoder, AdditiveAttention, Decoder, Seq2Seq
├── Training Utilities           # epoch loop, AMP, gradient accumulation, checkpointing
├── Prepare Data                 # instantiate vocabs and DataLoaders
├── Inference Utility            # translate_text() with greedy/beam decode
└── Experiment 4 — Dropout Sweep # main ablation + CSV export + plots
```
