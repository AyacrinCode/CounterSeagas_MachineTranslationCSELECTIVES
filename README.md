# An NLP-Based English-Filipino Machine Translation System

This project implements a clean and modular data-driven machine translation pipeline for English <-> Filipino using PyTorch Seq2Seq (LSTM Encoder-Decoder), Hugging Face datasets, and SacreBLEU evaluation.

## Key Artifacts

- **ACL-style LaTeX report (Overleaf-ready):** `report/english_filipino_translation_report.tex`
- **Rendered PDF (if present in your checkout):** `report/english_filipino_translation_report.pdf`
- **Long-form Markdown draft:** `report/RESEARCH_PAPER.md`
- **Primary experiment notebooks:** `notebooks/` (see `notebooks/Final/` for final versions)

## Project Structure

```text
mt-thesis-project/
|-- data/
|-- models/
|-- notebooks/
|-- src/
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- model.py
|   |-- train.py
|   |-- evaluate.py
|   |-- translate.py
|-- app/
|   |-- app.py
|-- requirements.txt
|-- README.md
```

## Features

- Loads dataset from Hugging Face: rhyliieee/tagalog-filipino-english-translation
- Preprocessing pipeline:
  - Lowercasing
  - Tokenization
  - Vocabulary building
  - Sequence padding
- Seq2Seq model with:
  - LSTM Encoder
  - LSTM Decoder
  - Optional attention mechanism
- Training and validation loops
- SacreBLEU evaluation on test split
- Translation inference mode (no retraining needed)
- Supports word and sentence translation
- Simple Streamlit app for text input/output

## Paper / Report (ACL Format)

The LaTeX report is already structured for the official ACL style.

### Option A: Build in Overleaf (recommended)

Upload these files into an Overleaf project:

- `report/english_filipino_translation_report.tex`
- `report/references.bib`
- `report/acl.sty`
- `report/acl_natbib.bst`

Then set the main file to `english_filipino_translation_report.tex`.

To switch between modes:
- Review (anonymous + line numbers): `\usepackage[review]{acl}`
- Camera-ready: `\usepackage{acl}` (and replace the author block)
- Preprint: `\usepackage[preprint]{acl}`

### Option B: Build locally (pdfLaTeX)

From the `report/` folder:

```bash
pdflatex english_filipino_translation_report.tex
bibtex english_filipino_translation_report
pdflatex english_filipino_translation_report.tex
pdflatex english_filipino_translation_report.tex
```

## Experiment Notes

### Dropout sweep (Experiment 4)

The dropout sweep results (dropout values 0.3/0.5/0.7) are summarized inside the paper/report and were computed using a capped dataset configuration (20k train, 1.5k val) with an evaluation subset of 200 pairs.

If you are tracking raw metrics during runs, store them as CSVs (e.g., training history per epoch and summary tables) and reference them in the report.

## 1. Environment Setup

Open terminal in project root folder:

```bash
cd mt-thesis-project
```

Create and activate virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Train the Model

### RTX 3050 (4GB) Recommended Settings

Use this command for better compatibility with low-VRAM GPUs:

```bash
python src/train.py --src_lang english --tgt_lang filipino --epochs 10 --batch_size 8 --embedding_dim 128 --hidden_dim 256 --use_attention --grad_accum_steps 4 --num_workers 0 --save_path models/seq2seq_en_to_fil_gpu.pt
```

### Capped Dataset (Faster 10-Epoch Runs)

If you want to reliably finish 10 epochs faster, cap the number of samples:

```bash
python src/train.py --src_lang english --tgt_lang filipino --epochs 10 --batch_size 8 --embedding_dim 128 --hidden_dim 256 --use_attention --grad_accum_steps 4 --num_workers 0 --max_train_samples 20000 --max_val_samples 2000 --save_path models/seq2seq_en_to_fil_gpu.pt
```

PowerShell launcher with cap:

```powershell
./scripts/start_training.ps1 -Epochs 10 -BatchSize 8 -EmbeddingDim 128 -HiddenDim 256 -GradAccumSteps 4 -MaxTrainSamples 20000 -MaxValSamples 2000
```

Why this works better on 4GB VRAM:
- Smaller batch size and model dimensions reduce memory usage.
- Gradient accumulation keeps an effectively larger batch without OOM.
- Automatic mixed precision (AMP) is enabled by default on CUDA.

### English -> Filipino

```bash
python src/train.py --src_lang english --tgt_lang filipino --epochs 10 --batch_size 32 --use_attention
```

### Filipino -> English (Bidirectional support)

```bash
python src/train.py --src_lang filipino --tgt_lang english --epochs 10 --batch_size 32 --use_attention --save_path models/seq2seq_fil_to_en.pt
```

Notes:
- The best validation model is saved to the path in --save_path.
- Default checkpoint path is models/seq2seq_latest.pt.

## 3. Evaluate with SacreBLEU

```bash
python src/evaluate.py --checkpoint models/seq2seq_latest.pt --max_samples 300
```

Output:
- SacreBLEU score in terminal
- Sample translations
- CSV file with predictions: data/evaluation_samples.csv

## 4. Translate Text (CLI)

Single text translation:

```bash
python src/translate.py --checkpoint models/seq2seq_latest.pt --text "How are you today?"
```

Interactive mode:

```bash
python src/translate.py --checkpoint models/seq2seq_latest.pt
```

Type quit to exit interactive mode.

## 5. Run the Simple App (Streamlit)

```bash
streamlit run app/app.py
```

In the app:
- Enter checkpoint path (default is models/seq2seq_latest.pt)
- Enter text
- Click Translate

## Module Guide

- src/data_loader.py
  - Loads dataset and returns train/validation/test source-target pairs.
- src/preprocessing.py
  - Handles normalization, tokenization, vocabulary, numericalization, and padding.
- src/model.py
  - Defines Encoder, Decoder, Attention, and Seq2Seq model.
- src/train.py
  - Trains model with CrossEntropyLoss and Adam optimizer.
- src/evaluate.py
  - Computes SacreBLEU and saves sample translations.
- src/translate.py
  - Loads saved model and performs inference.
- app/app.py
  - Streamlit UI for translation.

## Suggested Thesis Defense Flow

1. Explain dataset source and train/validation/test splitting.
2. Explain preprocessing choices and vocabulary handling.
3. Present Seq2Seq architecture (Encoder-Decoder and optional attention).
4. Show training/validation loss trends.
5. Report SacreBLEU score and qualitative examples.
6. Demo CLI and Streamlit app for word and sentence translation.

## Troubleshooting

- If NLTK tokenizer data is missing, the project auto-downloads punkt.
- If checkpoint not found, train first or provide correct --checkpoint path.
- If output quality is low, train for more epochs or increase training data size.
