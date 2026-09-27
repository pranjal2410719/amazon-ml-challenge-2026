"""
Scoring script -- implements the EXACT metric described in the problem
statement: per-Source-1-entity F_0.5, macro-averaged across all entities.

    F_0.5 = (1.25 * P * R) / (0.25 * P + R)

Singleton rule (explicit, per spec):
    - True matches empty AND predicted empty  -> score 1.0
    - True matches empty AND predicted non-empty -> score 0.0
    - True matches non-empty AND predicted empty -> score 0.0 (P and R both
      undefined/zero in the useful sense; treated as 0.0, matching "missed
      everything")

Use this as the SINGLE source of truth for local validation. Never approximate
with sklearn's flat/micro F-beta -- it does not match the per-entity macro
average the leaderboard uses.
"""
from typing import Dict, List


def _f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta ** 2
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def score_entity(predicted: List[str], truth: List[str]) -> float:
    pred_set, truth_set = set(predicted), set(truth)

    if not truth_set:
        # True singleton.
        return 1.0 if not pred_set else 0.0

    if not pred_set:
        # Missed everything for a non-singleton entity.
        return 0.0

    tp = len(pred_set & truth_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(truth_set) if truth_set else 0.0
    return _f_beta(precision, recall, beta=0.5)


def macro_f05(predictions: Dict[str, List[str]], ground_truth: Dict[str, List[str]]) -> dict:
    """
    predictions / ground_truth: {source1_entity_id: [matched_ids, ...]}
    Every key in ground_truth must be present in predictions (missing ->
    treated as empty prediction, scored accordingly, matching how the real
    leaderboard would likely penalize a missing row -- though a truly
    missing row is a hard validation failure per the spec, not just a
    scoring detail).
    """
    scores = []
    for s1_id, truth_list in ground_truth.items():
        pred_list = predictions.get(s1_id, [])
        scores.append(score_entity(pred_list, truth_list))

    return {
        "macro_f0.5": sum(scores) / len(scores) if scores else 0.0,
        "n_entities": len(scores),
        "n_singletons_true": sum(1 for v in ground_truth.values() if not v),
        "n_singletons_pred": sum(1 for v in predictions.values() if not v),
    }


def load_id_list_tsv(path: str) -> Dict[str, List[str]]:
    """Loads a TSV with columns [source1_entity_id, matched_entity_ids] (or
    candidate_entity_ids) where the second column is a comma-separated list,
    empty string for no matches."""
    import csv
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) < 2:
                s1_id, ids_str = row[0], ""
            else:
                s1_id, ids_str = row[0], row[1]
            ids = [x.strip() for x in ids_str.split(",") if x.strip()] if ids_str else []
            result[s1_id] = ids
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Score matching_results.tsv against ground truth")
    p.add_argument("--predictions", required=True)
    p.add_argument("--ground_truth", required=True)
    args = p.parse_args()

    preds = load_id_list_tsv(args.predictions)
    truth = load_id_list_tsv(args.ground_truth)
    result = macro_f05(preds, truth)
    print(result)
