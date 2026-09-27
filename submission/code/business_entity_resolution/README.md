# Business Entity Resolution - Submission Package

## Overview
This package contains the complete solution for the Amazon ML Challenge 2026 Business Entity Resolution task.

## Structure
```
business_entity_resolution/
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── blocking.py         # Candidate generation via blocking
│   ├── features.py         # Feature engineering
│   ├── model.py            # Model training (XGBoost)
│   ├── normalize.py        # Text normalization
│   ├── predict.py          # Inference pipeline
│   ├── score.py            # Evaluation metrics
│   └── train.py            # Training pipeline
├── requirements.txt        # Python dependencies
├── run_pipeline.py         # Full pipeline runner (copied from parent)
├── validate_submission.py  # Submission validator
└── README.md               # This file
```

## Requirements
- Python 3.9+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Running the Full Pipeline

The `run_pipeline.py` script is included in this directory (copied from the parent directory). It runs the complete training and inference pipeline.

### Prerequisites
Ensure the following data directories exist:
- `student_resource/dataset/train/` - Training data (train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv)
- `student_resource/dataset/test/` - Test data (test_source1.tsv, test_source2.tsv, test_source3.tsv)

### Run Full Pipeline
```bash
python run_pipeline.py
```

**Outputs:**
- `output/matching_results.tsv` - Final matching predictions
- `output/candidate_pairs.tsv` - Candidate pairs for validation
- `output/model.txt` - Trained XGBoost model

## Running Individual Components

### 1. Training Only
Train the model and save to `output/model.txt`:
```bash
python -m src.train
```

Or programmatically:
```python
from src.train import train_model
import pandas as pd

train_s1 = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t")
train_s2 = pd.read_csv("student_resource/dataset/train/train_source2.tsv", sep="\t")
train_s3 = pd.read_csv("student_resource/dataset/train/train_source3.tsv", sep="\t")
ground_truth = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t")

booster = train_model(train_s1, train_s2, train_s3, ground_truth, output_path="output/model.txt")
```

### 2. Candidate Generation (Blocking)
Generate candidate pairs using blocking:
```python
from src.blocking import generate_candidates
import pandas as pd

test_s1 = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep="\t")
test_s2 = pd.read_csv("student_resource/dataset/test/test_source2.tsv", sep="\t")
test_s3 = pd.read_csv("student_resource/dataset/test/test_source3.tsv", sep="\t")

candidates = generate_candidates(test_s1, test_s2, test_s3, window=15, max_candidates=20)
candidates.to_csv("output/candidate_pairs.tsv", sep="\t", index=False)
```

### 3. Feature Engineering
Extract features for candidate pairs:
```python
from src.features import extract_features
import pandas as pd

candidates = pd.read_csv("output/candidate_pairs.tsv", sep="\t")
test_s1 = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep="\t")
test_s2 = pd.read_csv("student_resource/dataset/test/test_source2.tsv", sep="\t")
test_s3 = pd.read_csv("student_resource/dataset/test/test_source3.tsv", sep="\t")

features = extract_features(candidates, test_s1, test_s2, test_s3)
```

### 4. Inference Only
Run inference with a pre-trained model:
```python
from src.predict import run_inference
from src.train import load_model
import pandas as pd

test_s1 = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep="\t")
test_s2 = pd.read_csv("student_resource/dataset/test/test_source2.tsv", sep="\t")
test_s3 = pd.read_csv("student_resource/dataset/test/test_source3.tsv", sep="\t")

booster = load_model("output/model.txt")

run_inference(
    test_s1, test_s2, test_s3, booster,
    threshold=0.5,
    candidate_out="output/candidate_pairs.tsv",
    matching_out="output/matching_results.tsv"
)
```

### 5. Evaluation/Scoring
Evaluate predictions against ground truth:
```python
from src.score import evaluate
import pandas as pd

predictions = pd.read_csv("output/matching_results.tsv", sep="\t")
ground_truth = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t")

metrics = evaluate(predictions, ground_truth)
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1: {metrics['f1']:.4f}")
```

## Validating Output

### 1. Validate Submission Format
Run the submission validator to check output format and completeness:
```bash
python validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

**Expected output:** `PASS` (exit code 0)

### 2. Check Output Files Manually
Verify the output files have correct format:
```bash
# Check matching_results.tsv
head -5 output/matching_results.tsv
# Should have columns: source1_entity_id, matched_entity_ids

# Check candidate_pairs.tsv
head -5 output/candidate_pairs.tsv
# Should have columns: source1_entity_id, candidate_entity_ids
```

### 3. Validate Model Output
Ensure the model file exists and is loadable:
```bash
python -c "from src.train import load_model; booster = load_model('output/model.txt'); print('Model loaded successfully')"
```

## Key Parameters
- **Blocking window**: 15 (controls candidate generation scope)
- **Max candidates per entity**: 20
- **Validation split**: 15% of training data
- **Threshold range**: 0.3 - 0.8 (tuned on validation)

## Data Format
Input files (TSV):
- `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv` - Training entities
- `train_ground_truth.tsv` - Ground truth matches
- `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv` - Test entities

Output files (TSV):
- `matching_results.tsv` - Columns: `source1_entity_id`, `matched_entity_ids`
- `candidate_pairs.tsv` - Columns: `source1_entity_id`, `candidate_entity_ids`

## License
MIT/Apache 2.0 compatible dependencies only.