import os
import sys
import time
import subprocess
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.blocking import generate_candidates, blocking_quality_report
from src.train import build_training_set, train_model, save_model, load_model
from src.predict import run_inference
from src.score import macro_f05, score_entity

DATA_DIR = "student_resource/dataset"   
OUTPUT_DIR = "output"
VALIDATION_FRACTION = 0.15
RANDOM_SEED = 42
BLOCKING_WINDOW = 15
BLOCKING_MAX_CANDIDATES = 20
THRESHOLDS_TO_TRY = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
VALIDATE_SUBMISSION_SCRIPT = "utils/validate_submission.py"  

def log(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}")

def load_data():
    log("STEP 1: Loading data")
    t0 = time.time()
    s1 = pd.read_csv(f"{DATA_DIR}/train/train_source1.tsv", sep="\t")
    s2 = pd.read_csv(f"{DATA_DIR}/train/train_source2.tsv", sep="\t")
    s3 = pd.read_csv(f"{DATA_DIR}/train/train_source3.tsv", sep="\t")
    gt_df = pd.read_csv(f"{DATA_DIR}/train/train_ground_truth.tsv", sep="\t", keep_default_na=False)

    test_s1 = pd.read_csv(f"{DATA_DIR}/test/test_source1.tsv", sep="\t")
    test_s2 = pd.read_csv(f"{DATA_DIR}/test/test_source2.tsv", sep="\t")
    test_s3 = pd.read_csv(f"{DATA_DIR}/test/test_source3.tsv", sep="\t")

    ground_truth = {
        row.source1_entity_id: [x for x in row.matched_entity_ids.split(",") if x]
        for row in gt_df.itertuples()
    }
    print(f"Loaded in {time.time()-t0:.1f}s")
    print(f"Train: S1={len(s1):,} S2={len(s2):,} S3={len(s3):,}")
    print(f"Test:  S1={len(test_s1):,} S2={len(test_s2):,} S3={len(test_s3):,}")
    return s1, s2, s3, ground_truth, test_s1, test_s2, test_s3

def make_validation_split(s1, ground_truth):
    log("STEP 2: Creating validation split")
    rng = np.random.default_rng(RANDOM_SEED)
    ids = s1["entity_id"].to_numpy(dtype=object).copy()
    perm = rng.permutation(len(ids))
    ids = ids[perm]
    n_val = int(len(ids) * VALIDATION_FRACTION)
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])

    s1_train = s1[s1["entity_id"].isin(train_ids)].reset_index(drop=True)
    s1_val = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)
    gt_train = {k: v for k, v in ground_truth.items() if k in train_ids}
    gt_val = {k: v for k, v in ground_truth.items() if k in val_ids}

    print(f"Train split: {len(s1_train):,} entities | Validation split: {len(s1_val):,} entities")
    return s1_train, s1_val, gt_train, gt_val

def run_blocking_stage(s1_subset, s2, s3, ground_truth_subset, label, n_s2_s3):
    log(f"STEP 3: Blocking on {label} ({len(s1_subset):,} entities)")
    t0 = time.time()
    candidates = generate_candidates(
        s1_subset, s2, s3, window=BLOCKING_WINDOW, max_candidates=BLOCKING_MAX_CANDIDATES
    )
    elapsed = time.time() - t0
    print(f"Blocking took {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if ground_truth_subset:
        report = blocking_quality_report(candidates, ground_truth_subset, n_s2_s3=n_s2_s3)
        print("Blocking quality report:", report)
        if report["recall_ceiling"] < 0.85:
            print("WARNING: recall ceiling below 85% -- consider increasing "
                  "BLOCKING_WINDOW or BLOCKING_MAX_CANDIDATES before proceeding.")
    return candidates

def train_matcher(s1_train, s2, s3, gt_train, candidates_train):
    log("STEP 4: Training the matcher")
    t0 = time.time()
    X, y, name_tfidf, addr_tfidf = build_training_set(
        s1_train, s2, s3, gt_train, candidates_train, max_neg_per_entity=10
    )
    print(f"Training set: {X.shape[0]:,} pairs ({y.sum():,} positive, {(y==0).sum():,} negative)")
    model = train_model(X, y)
    save_model(model, f"{OUTPUT_DIR}/model.txt")
    print(f"Trained in {time.time()-t0:.1f}s, saved to {OUTPUT_DIR}/model.txt")
    return model

def tune_threshold(s1_val, s2, s3, gt_val, candidates_val, booster):
    log("STEP 5: Tuning decision threshold on validation split")
    from src.features import pair_features, TfidfSimilarity

    s1_lookup = s1_val.set_index("entity_id").to_dict("index")
    s2_lookup = s2.set_index("entity_id").to_dict("index")
    s3_lookup = s3.set_index("entity_id").to_dict("index")

    def resolve(eid):
        return s2_lookup.get(eid) if eid.startswith("S2-") else s3_lookup.get(eid)

    all_names = list(s1_val["business_name"]) + list(s2["business_name"]) + list(s3["business_name"])
    all_addrs = list(s1_val["business_address"]) + list(s2["business_address"]) + list(s3["business_address"])
    name_tfidf = TfidfSimilarity(all_names)
    addr_tfidf = TfidfSimilarity(all_addrs)

    scored_candidates = {}
    for eid in s1_val["entity_id"]:
        s1_rec = s1_lookup[eid]
        cand_ids = candidates_val.get(eid, [])
        pairs = []
        for cid in cand_ids:
            crec = resolve(cid)
            if crec is None:
                continue
            feats = pair_features(s1_rec, crec, name_tfidf, addr_tfidf)
            prob = booster.predict([feats])[0]
            pairs.append((cid, prob))
        scored_candidates[eid] = pairs

    best_threshold, best_score = None, -1
    for thresh in THRESHOLDS_TO_TRY:
        preds = {
            eid: [cid for cid, prob in pairs if prob >= thresh]
            for eid, pairs in scored_candidates.items()
        }
        result = macro_f05(preds, gt_val)
        print(f"  threshold={thresh}: macro F_0.5 = {result['macro_f0.5']:.4f}")
        if result["macro_f0.5"] > best_score:
            best_score = result["macro_f0.5"]
            best_threshold = thresh

    print(f"\nBest threshold: {best_threshold} (F_0.5 = {best_score:.4f})")
    return best_threshold, best_score

def run_final_test_inference(test_s1, test_s2, test_s3, booster, best_threshold):
    log("STEP 6: Running full inference on the REAL test set")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()
    run_inference(
        test_s1, test_s2, test_s3, booster, threshold=best_threshold,
        candidate_out=f"{OUTPUT_DIR}/candidate_pairs.tsv",
        matching_out=f"{OUTPUT_DIR}/matching_results.tsv",
        blocking_kwargs={"window": BLOCKING_WINDOW, "max_candidates": BLOCKING_MAX_CANDIDATES},
    )
    print(f"Test inference took {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")
    print(f"Wrote {OUTPUT_DIR}/matching_results.tsv and {OUTPUT_DIR}/candidate_pairs.tsv")

def validate_output():
    log("STEP 7: Validating output format")
    if not os.path.exists(VALIDATE_SUBMISSION_SCRIPT):
        print(f"'{VALIDATE_SUBMISSION_SCRIPT}' not found -- skipping automated format check. "
              "Locate the challenge's validator script and run it manually before submitting.")
        return
    result = subprocess.run(
        ["python3", VALIDATE_SUBMISSION_SCRIPT,
         "--matching", f"{OUTPUT_DIR}/matching_results.tsv",
         "--candidate", f"{OUTPUT_DIR}/candidate_pairs.tsv",
         "--test-dir", f"{DATA_DIR}/test"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("VALIDATION FAILED -- fix issues above before submitting:")
        print(result.stderr)
    else:
        print("VALIDATION PASSED -- safe to submit.")

def main():
    s1, s2, s3, ground_truth, test_s1, test_s2, test_s3 = load_data()
    n_s2_s3 = len(s2) + len(s3)

    s1_train, s1_val, gt_train, gt_val = make_validation_split(s1, ground_truth)

    candidates_train = run_blocking_stage(s1_train, s2, s3, gt_train, "TRAIN split", n_s2_s3)
    candidates_val = run_blocking_stage(s1_val, s2, s3, gt_val, "VALIDATION split", n_s2_s3)

    model = train_matcher(s1_train, s2, s3, gt_train, candidates_train)
    booster = load_model(f"{OUTPUT_DIR}/model.txt")

    best_threshold, best_score = tune_threshold(s1_val, s2, s3, gt_val, candidates_val, booster)

    run_final_test_inference(test_s1, test_s2, test_s3, booster, best_threshold)
    validate_output()

    log("PIPELINE COMPLETE")
    print(f"Final chosen threshold: {best_threshold}")
    print(f"Validation macro F_0.5: {best_score:.4f}")
    print(f"Outputs ready in: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()