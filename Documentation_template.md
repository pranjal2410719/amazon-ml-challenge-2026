# Amazon ML Challenge 2026 — Methodology Document

**Team Name:** Pranjal Yadav  
**Team Members:** Pranjal Yadav (pranjal2410719)  
**Submission Date:** [TO BE FILLED AFTER RUNNING PIPELINE]

---

## 1. Methodology Used

### Problem Formulation
Business Entity Resolution (ER) across three independent data sources (S1, S2, S3) with no shared identifiers. The goal is to find all matching records from S2 and S3 for each S1 entity.

### Approach Overview
Our pipeline follows a classic ER workflow:
1. **Blocking / Candidate Generation** — reduce the O(n²) search space to a manageable set of candidate pairs.
2. **Feature Engineering** — extract string-similarity and structural features from each candidate pair.
3. **Matching Model** — a LightGBM binary classifier that scores each candidate pair.
4. **Thresholding & Output** — convert probabilities to final matches using a precision-oriented threshold.

---

## 2. Candidate Generation / Blocking Strategy

### Algorithm
We use the **Sorted Neighborhood Method (SNM)**, a standard ER blocking technique:

1. **Country partitioning** — records are grouped by country (generic string equality, works for unseen countries like France).
2. **Multiple sort keys** — we sort by normalized name and address keys (token-sorted to be invariant to word order).
3. **Fixed window** — for each S1 record, we compare against a window of neighbors on each side (default: 15).
4. **Exact-name pass** — an additional pass that guarantees all records sharing an exact normalized name are captured as candidates, closing the recall gap on large duplicate-name groups (e.g., franchises).
5. **Re-ranking & truncation** — candidates are re-ranked by a combined name+address token Jaccard score and truncated to a hard cap (default: 20 per entity).

### Rationale
- SNM is O(n log n) for the sort plus O(n × window) for windowing — no pairwise blowup.
- Token-sorted keys handle word-order transpositions (e.g., "Gangaya Techinfra Limited" vs "Techinfra Gangaya Limited").
- The exact-name pass handles large duplicate-name groups that exceed the window size.
- Re-ranking by combined name+address Jaccard ensures we keep the strongest candidates, not just the first ones in the window.

### Blocking Quality Metrics
We report:
- **Recall ceiling**: fraction of true matches that survived blocking.
- **Average candidates per entity**: directly rewarded when small (mid-challenge ranking update).
- **Reduction ratio**: 1 - (avg candidates per entity / total S2+S3 records).

### Actual Results [TO BE FILLED AFTER RUNNING PIPELINE]
| Metric | Value |
|--------|-------|
| Recall Ceiling (Train) | [XX.XX%] |
| Recall Ceiling (Validation) | [XX.XX%] |
| Avg Candidates per Entity (Train) | [XX] |
| Avg Candidates per Entity (Validation) | [XX] |
| Max Candidates per Entity | [XX] |
| Reduction Ratio | [XX.XX%] |
| Blocking Window Used | 15 |
| Max Candidates Used | 20 |

---

## 3. Model Architecture and Feature Engineering

### Model
**LightGBM binary classifier** (gradient-boosted trees) trained from scratch on competition data — no pretrained weights, so no license/param-count concerns.

### Features
We extract 11 features per candidate pair:

| Feature | Description |
|---------|-------------|
| `name_ratio` | rapidfuzz fuzz.ratio on normalized names |
| `name_partial_ratio` | rapidfuzz fuzz.partial_ratio |
| `name_token_sort_ratio` | rapidfuzz fuzz.token_sort_ratio |
| `name_jaccard` | Token Jaccard similarity on name tokens |
| `name_tfidf_cosine` | Char n-gram TF-IDF cosine similarity |
| `addr_ratio` | rapidfuzz fuzz.ratio on normalized addresses |
| `addr_jaccard` | Token Jaccard similarity on address tokens |
| `addr_tfidf_cosine` | Char n-gram TF-IDF cosine similarity |
| `country_match` | 1.0 if countries match, else 0.0 |
| `name_len_diff` | Normalized absolute length difference |
| `name_exact_norm_match` | 1.0 if normalized names are identical |

All features are generic string-similarity / structural signals — none are tied to specific country values, so they generalize to unseen countries (e.g., France).

### Training Data
- **Positives**: every (S1, matched_id) pair from ground truth.
- **Negatives**: HARD negatives — candidates that survived blocking for that S1 entity but are NOT in ground truth. These are far more informative than random negatives because they are exactly the confusions the model will face at inference time.

### Model Hyperparameters
```python
{
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}
```

### Actual Results [TO BE FILLED AFTER RUNNING PIPELINE]
| Metric | Value |
|--------|-------|
| Training Set Size (pairs) | [XXXX] |
| Positive Pairs | [XXXX] |
| Negative Pairs | [XXXX] |
| Max Negatives per Entity | 10 |
| Feature Importances (top 5) | [TO BE FILLED] |

---

## 4. Experiments and Results

### Validation Setup
We hold out a validation split (15% of Source 1 training entities) from the training data and score it using the F_0.5 formula (per-entity macro average, as specified in the problem statement).

### Threshold Sweep Results [TO BE FILLED AFTER RUNNING PIPELINE]

| Threshold | Macro Precision | Macro Recall | Macro F_0.5 |
|-----------|----------------|--------------|-------------|
| 0.3 | [XX.XX%] | [XX.XX%] | [XX.XX%] |
| 0.4 | [XX.XX%] | [XX.XX%] | [XX.XX%] |
| 0.5 | [XX.XX%] | [XX.XX%] | [XX.XX%] |
| 0.6 | [XX.XX%] | [XX.XX%] | [XX.XX%] |
| 0.7 | [XX.XX%] | [XX.XX%] | [XX.XX%] |
| 0.8 | [XX.XX%] | [XX.XX%] | [XX.XX%] |

**Best Threshold:** [X.X] (F_0.5 = [XX.XX%])

### Key Findings
- **Blocking is critical**: recall ceiling is the upper bound on what the matcher can ever achieve. We tune `window` and `max_candidates` to balance recall and candidate set size.
- **Address is a strong disambiguator**: EDA found that name+country+address collisions are ZERO in Source 1 — address is what separates the correct branch from the wrong one once name ties.
- **Precision matters more than recall**: F_0.5 weights precision 2× over recall. We tune the prediction threshold accordingly.
- **Hard negatives are essential**: training on candidates that survived blocking but aren't true matches forces the model to learn fine-grained distinctions.

### Ablation Studies [OPTIONAL - TO BE FILLED IF PERFORMED]
| Experiment | Recall Ceiling | Avg Candidates | F_0.5 |
|------------|----------------|----------------|-------|
| Baseline (window=15, max=20) | [XX.XX%] | [XX] | [XX.XX%] |
| Reduced window (10, 15) | [XX.XX%] | [XX] | [XX.XX%] |
| Increased negatives (15) | [XX.XX%] | [XX] | [XX.XX%] |

---

## 5. Conclusion

Our pipeline is designed for scale, precision, and reproducibility:
- **Scalable blocking** (SNM) handles millions of records without pairwise blowup.
- **Generic normalization** handles unseen countries and noise patterns.
- **LightGBM** provides fast, accurate matching on candidate pairs.
- **Reproducible output** in the exact TSV format required by the spec.

### Final Competition Metrics [TO BE FILLED AFTER LEADERBOARD SUBMISSION]
| Metric | Value |
|--------|-------|
| Public Leaderboard F_0.5 | [XX.XX%] |
| Private Leaderboard F_0.5 | [XX.XX%] |
| Final Threshold Used | [X.X] |

---

## 6. Reproduction Instructions

### Setup
```bash
pip install -r requirements.txt
```

### Exploratory Data Analysis
```bash
python eda.py
```
(Or open `eda_completed.ipynb` in Jupyter.)

### Training
```python
from src.model import EntityResolutionModel
import pandas as pd

s1 = pd.read_csv('student_resource/dataset/train/train_source1.tsv', sep='\t')
s2 = pd.read_csv('student_resource/dataset/train/train_source2.tsv', sep='\t')
s3 = pd.read_csv('student_resource/dataset/train/train_source3.tsv', sep='\t')
gt = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t')

model = EntityResolutionModel()
candidates = model.block(s1, s2, s3, window=15, max_candidates=20)
X, y, name_tfidf, addr_tfidf = model.build_training_set(s1, s2, s3, gt, candidates)
model.train(X, y)
model.save('model.txt', 'name_tfidf.pkl', 'addr_tfidf.pkl')
```

### Inference
```python
test_s1 = pd.read_csv('student_resource/dataset/test/test_source1.tsv', sep='\t')
test_s2 = pd.read_csv('student_resource/dataset/test/test_source2.tsv', sep='\t')
test_s3 = pd.read_csv('student_resource/dataset/test/test_source3.tsv', sep='\t')

model = EntityResolutionModel()
model.load('model.txt', 'name_tfidf.pkl', 'addr_tfidf.pkl')
matches, candidates = model.infer(test_s1, test_s2, test_s3, threshold=0.5)
```

### Validation
```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
```

---

## 7. Submission Package Structure

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv      # Final matches (leaderboard file)
│   └── candidate_pairs.tsv       # Blocking candidates
├── code/
│   └── business_entity_resolution/
│       ├── src/                  # All source code
│       │   ├── __init__.py
│       │   ├── normalize.py
│       │   ├── blocking.py
│       │   ├── features.py
│       │   ├── train.py
│       │   ├── predict.py
│       │   ├── score.py
│       │   └── model.py
│       ├── requirements.txt      # Pinned dependencies
│       ├── validate_submission.py
│       └── README.md             # Run instructions
└── Documentation_template.md     # This file (filled in)
```

---

*This document will be finalized with actual numbers after the pipeline is executed on the real dataset.*