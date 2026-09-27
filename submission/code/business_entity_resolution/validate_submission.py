"""
validate_submission.py — stdlib only, no dependencies.

Checks both output files against every rule in the Amazon ML Challenge 2026
spec so you can catch a rejection locally instead of spending a submission
on it.

Usage:
    python3 utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test

It prints PASS (exit 0) when the files are safe to submit, or a numbered
list of issues to fix (exit 1). It only reads your output files and the
test source files; it does not compute your score.
"""
import argparse
import csv
import os
import sys
from collections import Counter


def _load_tsv(path, id_col):
    """Load a TSV with an id column and a comma-separated list column."""
    if not os.path.isfile(path):
        return None, f"File not found: {path}"
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            return None, f"Empty file: {path}"
        if len(header) < 2:
            return None, f"Expected 2 columns, got {len(header)} in {path}"
        if header[0] != id_col:
            return None, f"Expected first column '{id_col}', got '{header[0]}' in {path}"
        for lineno, row in enumerate(reader, start=2):
            if len(row) < 2:
                return None, f"{path}:{lineno}: expected 2 columns, got {len(row)}"
            s1_id, ids_str = row[0], row[1]
            if s1_id in rows:
                return None, f"{path}:{lineno}: duplicate source1_entity_id '{s1_id}'"
            ids = [x.strip() for x in ids_str.split(",") if x.strip()] if ids_str else []
            if len(ids) != len(set(ids)):
                return None, f"{path}:{lineno}: duplicate entity IDs in list for '{s1_id}'"
            rows[s1_id] = ids
    return rows, None


def _load_test_ids(test_dir):
    """Collect all valid entity IDs from the three test source files."""
    ids = set()
    for src in ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv"):
        path = os.path.join(test_dir, src)
        if not os.path.isfile(path):
            return None, f"Missing test file: {path}"
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            if header is None or header[0] != "entity_id":
                return None, f"Expected 'entity_id' header in {path}"
            for row in reader:
                if row:
                    ids.add(row[0])
    return ids, None


def validate(matching_path, candidate_path, test_dir):
    issues = []

    # 1. Load files
    matching, err = _load_tsv(matching_path, "source1_entity_id")
    if err:
        issues.append(f"[1] {err}")
        matching = {}
    candidates, err = _load_tsv(candidate_path, "source1_entity_id")
    if err:
        issues.append(f"[2] {err}")
        candidates = {}

    # 2. Load test IDs
    test_ids, err = _load_test_ids(test_dir)
    if err:
        issues.append(f"[3] {err}")
        test_ids = set()

    # 3. Every S1 entity in test set must have exactly one row in matching_results
    s1_test_ids = set()
    s1_test_path = os.path.join(test_dir, "test_source1.tsv")
    if os.path.isfile(s1_test_path):
        with open(s1_test_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if row:
                    s1_test_ids.add(row[0])

    missing_s1 = s1_test_ids - set(matching.keys())
    if missing_s1:
        issues.append(f"[4] Missing Source 1 entities in matching_results.tsv: {sorted(missing_s1)[:10]}...")

    extra_s1 = set(matching.keys()) - s1_test_ids
    if extra_s1:
        issues.append(f"[5] Extra Source 1 entities in matching_results.tsv: {sorted(extra_s1)[:10]}...")

    # 4. matched_entity_ids must only reference S2/S3 IDs that exist in the test set
    for s1_id, ids in matching.items():
        for eid in ids:
            if not eid.startswith(("S2-", "S3-")):
                issues.append(f"[6] {s1_id}: matched ID '{eid}' is not from Source 2 or 3")
            elif test_ids and eid not in test_ids:
                issues.append(f"[7] {s1_id}: matched ID '{eid}' does not exist in test set")

    # 5. Same checks for candidate_pairs.tsv
    missing_cand = s1_test_ids - set(candidates.keys())
    if missing_cand:
        issues.append(f"[8] Missing Source 1 entities in candidate_pairs.tsv: {sorted(missing_cand)[:10]}...")

    for s1_id, ids in candidates.items():
        for eid in ids:
            if not eid.startswith(("S2-", "S3-")):
                issues.append(f"[9] {s1_id}: candidate ID '{eid}' is not from Source 2 or 3")
            elif test_ids and eid not in test_ids:
                issues.append(f"[10] {s1_id}: candidate ID '{eid}' does not exist in test set")

    # 6. Every ID in matching_results.tsv should appear in candidate_pairs.tsv
    if matching and candidates:
        for s1_id, ids in matching.items():
            cand_set = set(candidates.get(s1_id, []))
            for eid in ids:
                if eid not in cand_set:
                    issues.append(f"[11] {s1_id}: matched ID '{eid}' not found in candidate_pairs.tsv (pipeline bug?)")

    if issues:
        print("VALIDATION FAILED — issues to fix:")
        for i, issue in enumerate(issues, start=1):
            print(f"  {i}. {issue}")
        sys.exit(1)
    else:
        print("PASS")
        sys.exit(0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Validate submission files for the Amazon ML Challenge 2026")
    p.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    p.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    p.add_argument("--test-dir", required=True, help="Directory containing test source TSV files")
    args = p.parse_args()
    validate(args.matching, args.candidate, args.test_dir)