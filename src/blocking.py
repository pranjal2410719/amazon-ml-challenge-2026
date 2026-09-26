"""
Blocking / candidate generation stage -- Sorted Neighborhood Method (SNM).

REPLACES an earlier MinHash-LSH implementation that was too slow at this
dataset's real scale (~24M total records across S1+S2+S3, train+test).
Pure-Python per-record MinHash construction (via `datasketch`) does not
vectorize and does not scale to millions of rows in hackathon time budgets --
measured at ~170s for just 500 S1 x 100K S2/S3 records, which would have
projected to many hours on the full data.

SNM is a standard, well-established ER blocking technique:
  1. Sort ALL records (S1 + S2 + S3, within a country partition) by a
     normalized string key.
  2. For each S1 record, only compare against a small fixed WINDOW of
     neighbors immediately before/after it in sorted order.
  3. Do this with >1 sort key (name, address) and UNION the results, so a
     record that's noisy on one field but clean on the other still gets
     caught.

This is O(n log n) for the sort (pandas/numpy sort of millions of rows takes
seconds) plus O(n * window) for the windowing step -- no pairwise blowup,
no per-record Python object construction. Scales comfortably to 10s of
millions of records on a laptop.

Country is used as a coarse partition (generic string equality -- works for
any unseen country value, e.g. France, with no special-casing).

Candidates are additionally re-ranked by a cheap token-Jaccard score before
truncation, so a small `max_candidates` keeps the STRONGEST candidates, not
just whichever fell within the window first -- important since a smaller,
high-quality candidate_pairs.tsv is now directly rewarded (see the
mid-challenge ranking update).
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from .normalize import normalize_name, normalize_address, name_tokens, address_tokens, sorted_token_key


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _exact_name_pass(combined: pd.DataFrame, id_col: str, src_col: str, name_key_col: str) -> dict:
    """
    Exact hash-join on normalized name (within a country partition), via a
    single sort + linear scan (no per-group Python object overhead from
    pandas groupby().apply(list), which is unnecessarily memory-heavy at
    millions of rows).

    WHY THIS EXISTS: EDA on the real data found normalized business names
    that repeat up to 253 times within Source 1 alone (franchise/chain-style
    generic names, e.g. "Primary Care Group"). The windowed sorted-neighborhood
    pass alone can silently miss recall on these: if a duplicate-name group is
    larger than the window, an S1 record sitting at one end of that sorted
    block never sees candidates at the other end. This pass guarantees every
    record sharing an exact normalized name is captured as a candidate,
    regardless of group size -- closing that recall gap. Address similarity
    (used in the re-ranking step below, and heavily in the matcher's
    features) is what then separates the correct branch from the wrong one,
    since EDA also found that name+country+address collisions are ZERO in
    Source 1 -- address is a near-perfect disambiguator once name ties.
    """
    sorted_df = combined.sort_values(name_key_col, kind="mergesort").reset_index(drop=True)
    keys = sorted_df[name_key_col].to_numpy()
    ids = sorted_df[id_col].to_numpy()
    srcs = sorted_df[src_col].to_numpy()
    n = len(sorted_df)

    result = defaultdict(set)
    i = 0
    while i < n:
        j = i + 1
        while j < n and keys[j] == keys[i]:
            j += 1
        if keys[i] and (j - i) > 1:
            group_srcs = srcs[i:j]
            s1_mask = group_srcs == "S1"
            cand_mask = (group_srcs == "S2") | (group_srcs == "S3")
            if s1_mask.any() and cand_mask.any():
                cand_ids = ids[i:j][cand_mask]
                for s1_id in ids[i:j][s1_mask]:
                    result[s1_id].update(cand_ids.tolist())
        i = j
    return result


def _snm_pass(combined: pd.DataFrame, key_col: str, id_col: str, src_col: str,
              window: int) -> dict:
    """One sorted-neighborhood pass on one sort key. Returns
    {s1_entity_id: set(candidate_ids)}."""
    sorted_df = combined.sort_values(key_col, kind="mergesort").reset_index(drop=True)
    ids = sorted_df[id_col].to_numpy()
    srcs = sorted_df[src_col].to_numpy()
    n = len(sorted_df)

    is_s1 = (srcs == "S1")
    is_cand = (srcs == "S2") | (srcs == "S3")

    result = defaultdict(set)
    s1_positions = np.where(is_s1)[0]
    for pos in s1_positions:
        lo = max(0, pos - window)
        hi = min(n, pos + window + 1)
        window_mask = is_cand[lo:hi]
        if window_mask.any():
            window_ids = ids[lo:hi][window_mask]
            result[ids[pos]].update(window_ids.tolist())
    return result


def generate_candidates(source1_df, source2_df, source3_df,
                         id_col="entity_id", name_col="business_name",
                         address_col="business_address", country_col="country",
                         window=15, max_candidates=20):
    """
    Full blocking pass across all three sources, partitioned by country,
    using Sorted Neighborhood on both name and address keys (unioned).

    `window`: how many neighbors on EACH side to compare against per sort
    pass. Larger = higher recall, more compute. Start around 10-20 and tune
    against your measured recall ceiling.

    `max_candidates`: hard cap per S1 entity after re-ranking by token
    Jaccard similarity -- keeps the candidate set small AND strong.

    Returns: dict {s1_entity_id: [candidate_entity_ids, ...]}, ranked
    strongest-first.
    """
    raw_candidates = defaultdict(set)

    countries = set(source1_df[country_col].unique())
    for country in countries:
        s1_part = source1_df[source1_df[country_col] == country]
        s2_part = source2_df[source2_df[country_col] == country]
        s3_part = source3_df[source3_df[country_col] == country]

        if len(s1_part) == 0:
            continue

        combined = pd.concat([
            pd.DataFrame({
                id_col: s1_part[id_col].values,
                "_src": "S1",
                "_name_key": s1_part[name_col].apply(lambda x: sorted_token_key(name_tokens(x))).values,
                "_addr_key": s1_part[address_col].apply(lambda x: sorted_token_key(address_tokens(x))).values,
            }),
            pd.DataFrame({
                id_col: s2_part[id_col].values,
                "_src": "S2",
                "_name_key": s2_part[name_col].apply(lambda x: sorted_token_key(name_tokens(x))).values,
                "_addr_key": s2_part[address_col].apply(lambda x: sorted_token_key(address_tokens(x))).values,
            }),
            pd.DataFrame({
                id_col: s3_part[id_col].values,
                "_src": "S3",
                "_name_key": s3_part[name_col].apply(lambda x: sorted_token_key(name_tokens(x))).values,
                "_addr_key": s3_part[address_col].apply(lambda x: sorted_token_key(address_tokens(x))).values,
            }),
        ], ignore_index=True)

        for key_col in ("_name_key", "_addr_key"):
            pass_result = _snm_pass(combined, key_col, id_col, "_src", window)
            for eid, cand_set in pass_result.items():
                raw_candidates[eid].update(cand_set)

        # Exact-match pass -- closes the recall gap on large duplicate-name
        # groups (e.g. franchises) that can exceed the window size.
        exact_result = _exact_name_pass(combined, id_col, "_src", "_name_key")
        for eid, cand_set in exact_result.items():
            raw_candidates[eid].update(cand_set)

    # Re-rank each entity's candidates by a COMBINED name+address token
    # Jaccard score, truncate. Combined (not name-only) because exact-name
    # duplicate groups tie on name similarity -- address is what actually
    # separates the correct branch from the wrong one (EDA: zero
    # name+country+address collisions in Source 1).
    s1_lookup = source1_df.set_index(id_col)[[name_col, address_col]].to_dict("index")
    s2_lookup = source2_df.set_index(id_col)[[name_col, address_col]].to_dict("index")
    s3_lookup = source3_df.set_index(id_col)[[name_col, address_col]].to_dict("index")

    def resolve(eid):
        if eid.startswith("S2-"):
            return s2_lookup.get(eid)
        if eid.startswith("S3-"):
            return s3_lookup.get(eid)
        return None

    candidates = {}
    for eid in source1_df[id_col]:
        cand_ids = raw_candidates.get(eid, set())
        if not cand_ids:
            candidates[eid] = []
            continue
        s1_rec = s1_lookup.get(eid)
        s1_name_set = name_tokens(s1_rec[name_col]) if s1_rec else set()
        s1_addr_set = address_tokens(s1_rec[address_col]) if s1_rec else set()

        scored = []
        for cid in cand_ids:
            crec = resolve(cid)
            if crec is None:
                continue
            name_sim = _jaccard(s1_name_set, name_tokens(crec[name_col]))
            addr_sim = _jaccard(s1_addr_set, address_tokens(crec[address_col]))
            combined_score = 0.5 * name_sim + 0.5 * addr_sim
            scored.append((cid, combined_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        candidates[eid] = [cid for cid, _ in scored[:max_candidates]]

    return candidates


def blocking_quality_report(candidates: dict, ground_truth: dict, n_s2_s3: int = None) -> dict:
    """
    Measures what actually matters: recall ceiling AND candidate set size.

    - recall_ceiling: fraction of true matches that survived blocking
      (upper bound on what the matcher can ever achieve).
    - avg_candidates_per_entity: mean candidate set size -- directly
      rewarded when small, per the mid-challenge ranking update.
    - reduction_ratio: 1 - (avg_candidates_per_entity / n_s2_s3), if
      n_s2_s3 (total S2+S3 record count) is supplied.
    """
    total_true, total_found = 0, 0
    sizes = []
    for eid, truth_ids in ground_truth.items():
        cand_set = set(candidates.get(eid, []))
        sizes.append(len(cand_set))
        truth_set = set(truth_ids)
        total_true += len(truth_set)
        total_found += len(truth_set & cand_set)

    report = {
        "recall_ceiling": total_found / total_true if total_true else 1.0,
        "avg_candidates_per_entity": sum(sizes) / len(sizes) if sizes else 0.0,
        "max_candidates_per_entity": max(sizes) if sizes else 0,
        "n_entities": len(sizes),
    }
    if n_s2_s3:
        report["reduction_ratio"] = 1 - (report["avg_candidates_per_entity"] / n_s2_s3)
    return report
