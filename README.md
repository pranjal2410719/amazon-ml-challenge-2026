# amazon-ml-challenge-2026

A structured pipeline for the Amazon ML Challenge 2026 — Business Entity Resolution. Provides data preprocessing, blocking, feature engineering, model training, and inference in a clean, reproducible layout.

## Overview

This repository contains a complete, reproducible entity resolution pipeline for the **Amazon ML Challenge 2026**. The task is to match business records from three independent sources (S1, S2, S3) that share no common identifiers, using only the provided training data (no external lookups).

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
├── eda_completed.ipynb           ← original notebook
├── run_pipeline.py               ← FULL END-TO-END PIPELINE (blocking → train → tune → infer → validate)
├── Documentation_template.md      ← filled-in methodology document
├── student_resource/             ← challenge folder with dataset
│   └── dataset/
│       ├── train/
│       │   ├── train_source1.tsv
│       │   ├── train_source2.tsv
│       │   ├── train_source3.tsv
│       │   └── train_ground_truth.tsv
│       └── test/
│           ├── test_source1.tsv
│           ├── test_source2.tsv
│           └── test_source3.tsv
└── src/                          ← source code package
    ├── __init__.py               (empty — makes src a package)
    ├── normalize.py
    ├── blocking.py               ← Sorted Neighborhood Method with token precomputation fix
    ├── features.py
    ├── train.py
    ├── predict.py
    ├── score.py
    └── model.py                  ← high-level EntityResolutionModel wrapper
```

## Quick Start (Full Pipeline)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your dataset in student_resource/dataset/ (train/ and test/ subfolders)

# 3. Run the full pipeline end-to-end
python run_pipeline.py
```

This executes:
1. **Load data** → train + test splits
2. **Validation split** (15% of train_source1 held out)
3. **Blocking** on train/validation splits → quality report (recall ceiling, avg candidates)
4. **Train LightGBM matcher** on train split (hard negatives from blocking)
5. **Tune threshold** on validation split (sweeps 0.3–0.8, maximizes macro F_0.5)
6. **Full inference on test set** → writes `output/matching_results.tsv` & `output/candidate_pairs.tsv`
7. **Validate output format** using `utils/validate_submission.py`

## Manual Usage (Python)

```python
import pandas as pd
from src.blocking import generate_candidates
from src.train import build_training_set, train_model, save_model
from src.predict import run_inference

# Load data
s1 = pd.read_csv('student_resource/dataset/train/train_source1.tsv', sep='\t')
s2 = pd.read_csv('student_resource/dataset/train/train_source2.tsv', sep='\t')
s3 = pd.read_csv('student_resource/dataset/train/train_source3.tsv', sep='\t')
gt_df = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t')
ground_truth = {row.source1_entity_id: [x for x in row.matched_entity_ids.split(",") if x] for row in gt_df.itertuples()}

# Blocking
candidates = generate_candidates(s1, s2, s3, window=15, max_candidates=20)

# Training
X, y, name_tfidf, addr_tfidf = build_training_set(s1, s2, s3, ground_truth, candidates)
model = train_model(X, y)
save_model(model, 'output/model.txt')

# Inference on test
test_s1 = pd.read_csv('student_resource/dataset/test/test_source1.tsv', sep='\t')
test_s2 = pd.read_csv('student_resource/dataset/test/test_source2.tsv', sep='\t')
test_s3 = pd.read_csv('student_resource/dataset/test/test_source3.tsv', sep='\t')
run_inference(test_s1, test_s2, test_s3, model, threshold=0.5,
              candidate_out='output/candidate_pairs.tsv',
              matching_out='output/matching_results.tsv')
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

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
```

## Documentation

See **Documentation_template.md** for the full methodology write-up required by the challenge, including:
- Candidate generation / blocking strategy
- Model architecture and feature engineering
- Experiments and results (with real numbers once pipeline runs)
- Reproduction instructions

---

*This repository follows the Amazon ML Challenge 2026 submission guidelines. No external data lookup is used — the pipeline relies solely on the provided training data.*