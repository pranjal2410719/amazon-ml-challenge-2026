# amazon-ml-challenge-2026

A structured pipeline for the Amazon ML Challenge 2026, providing data preprocessing, blocking, feature engineering, model training, and inference in a clean, reproducible layout.

## Overview

This repository contains a complete, reproducible entity resolution pipeline for the **Amazon ML Challenge 2026 — Business Entity Resolution Challenge**. The task is to match business records from three independent sources (S1, S2, S3) that share no common identifiers, using only the provided training data (no external lookups).

## Challenge Details

- **Competition:** Amazon ML Challenge 2026
- **Problem Type:** Business Entity Resolution (ER)
- **Evaluation Metric:** F_0.5 (precision-weighted, macro-averaged per Source 1 entity)
- **Timeline:** 72-hour hackathon (25–27 September 2026)

## Project Structure

```
/home/dev/Desktop/AMAZON ML/
├── README.md                    ← run instructions
├── requirements.txt              ← pip install list
├── eda.py                        ← run this first, cell by cell
├── eda_completed.ipynb           ← original notebook (kept at root)
├── Documentation_template.md      ← filled-in methodology document
├── student_resource/             ← (your existing challenge folder, dataset inside)
│   └── dataset/...
└── src/                          ← source code package
    ├── __init__.py               (empty – just makes src a package)
    ├── normalize.py
    ├── blocking.py
    ├── features.py
    ├── train.py
    ├── predict.py
    ├── score.py
    └── model.py                  ← high‑level EntityResolutionModel wrapper
```

## How to Run

1️⃣ **Setup** – Install dependencies:
```bash
pip install -r requirements.txt
```

2️⃣ **Exploratory Data Analysis** – Run `eda.py` (it loads the Jupyter notebook cells as a script) or open `eda_completed.ipynb` in Jupyter.

3️⃣ **Blocking / Candidate Generation** – Use `src.blocking.generate_candidates`.

4️⃣ **Feature Engineering** – `src.features.pair_features`.

5️⃣ **Train Model** – `src.train.build_training_set` + `src.train.train_model`.

6️⃣ **Inference** – `src.predict.run_inference` – produces `output/candidate_pairs.tsv` and `output/matching_results.tsv`.

## Quick Example (Python)

```python
import pandas as pd
from src.blocking import generate_candidates
from src.train import build_training_set, train_model, save_model
from src.predict import run_inference

# Load data
s1 = pd.read_csv('student_resource/dataset/train_source1.tsv', sep='\t')
# ... similarly load s2, s3, ground truth

candidates = generate_candidates(s1, s2, s3)
X, y, name_tfidf, addr_tfidf = build_training_set(s1, s2, s3, ground_truth, candidates)
model = train_model(X, y)
save_model(model, 'model.txt')
run_inference(s1_test, s2_test, s3_test, model)
```

## High-Level Wrapper

```python
from src.model import EntityResolutionModel

model = EntityResolutionModel()
candidates = model.block(source1_df, source2_df, source3_df)
X, y, name_tfidf, addr_tfidf = model.build_training_set(
    source1_df, source2_df, source3_df, ground_truth, candidates
)
model.train(X, y)
model.save('model.txt', 'name_tfidf.pkl', 'addr_tfidf.pkl')
matches, candidates = model.infer(
    test_source1_df, test_source2_df, test_source3_df, threshold=0.5
)
```

## Validation

Validate your submission files before uploading:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Documentation

See **Documentation_template.md** for the full methodology write-up required by the challenge, including:
- Candidate generation / blocking strategy
- Model architecture and feature engineering
- Experiments and results
- Reproduction instructions

---

*This repository follows the Amazon ML Challenge 2026 submission guidelines. No external data lookup is used — the pipeline relies solely on the provided training data.*

## Project Layout
```
E:/6ab10eb3b23ba_student_resource/
├── README.md                    ← run instructions
├── requirements.txt              ← pip install list
├── eda.py                        ← run this first, cell by cell (or open the notebook)
├── student_resource/             ← (your existing challenge folder, dataset inside)
│   └── dataset/...
└── src/                          ← source code
    ├── __init__.py               (empty – just makes src a package)
    ├── normalize.py
    ├── blocking.py
    ├── features.py
    ├── train.py
    ├── predict.py
    └── score.py
```

## How to Run
1️⃣ **Setup** – Install dependencies:
```bash
pip install -r requirements.txt
```
2️⃣ **Exploratory Data Analysis** – Run `eda.py` (it loads the Jupyter notebook cells as a script) or open `eda_completed.ipynb` in Jupyter.
3️⃣ **Blocking / Candidate Generation** – Use `src.blocking.generate_candidates`.
4️⃣ **Feature Engineering** – `src.features.pair_features`.
5️⃣ **Train Model** – `src.train.build_training_set` + `src.train.train_model`.
6️⃣ **Inference** – `src.predict.run_inference` – produces `output/candidate_pairs.tsv` and `output/matching_results.tsv`.

## Quick Example (Python)
```python
import pandas as pd
from src.blocking import generate_candidates
from src.train import build_training_set, train_model, save_model
from src.predict import run_inference

# Load data
s1 = pd.read_csv('student_resource/dataset/train_source1.tsv', sep='\t')
# ... similarly load s2, s3, ground truth

candidates = generate_candidates(s1, s2, s3)
X, y, name_tfidf, addr_tfidf = build_training_set(s1, s2, s3, ground_truth, candidates)
model = train_model(X, y)
save_model(model, 'model.txt')
run_inference(s1_test, s2_test, s3_test, model)
```

---
*The repository already contains a robust pipeline; this README mirrors the required structure for the challenge.*
