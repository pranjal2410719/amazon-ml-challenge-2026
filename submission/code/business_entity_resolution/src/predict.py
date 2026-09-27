"""
Inference pipeline: blocking -> feature computation -> matcher scoring ->
thresholding -> writes candidate_pairs.tsv and matching_results.tsv in the
exact format required by the spec.
"""
import csv
from .blocking import generate_candidates
from .features import pair_features, TfidfSimilarity


def _record_lookup(df, id_col="entity_id"):
    return df.set_index(id_col).to_dict("index")


def run_inference(source1_df, source2_df, source3_df, booster, threshold=0.5,
                   id_col="entity_id", name_col="business_name",
                   candidate_out="output/candidate_pairs.tsv",
                   matching_out="output/matching_results.tsv",
                   blocking_kwargs=None):
    blocking_kwargs = blocking_kwargs or {}

    # --- Stage 1: blocking ---
    candidates = generate_candidates(
        source1_df, source2_df, source3_df,
        id_col=id_col, name_col=name_col, **blocking_kwargs
    )

    # Write candidate_pairs.tsv (the last-stage candidate set fed to the model)
    with open(candidate_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1_id in source1_df[id_col]:
            ids = candidates.get(s1_id, [])
            # de-duplicate while preserving order
            seen = set()
            deduped = [x for x in ids if not (x in seen or seen.add(x))]
            writer.writerow([s1_id, ",".join(deduped)])

    # --- Stage 2: feature computation + matcher scoring ---
    s1_lookup = _record_lookup(source1_df, id_col)
    s2_lookup = _record_lookup(source2_df, id_col)
    s3_lookup = _record_lookup(source3_df, id_col)

    def resolve(eid):
        if eid.startswith("S2-"):
            return s2_lookup.get(eid)
        if eid.startswith("S3-"):
            return s3_lookup.get(eid)
        return None

    all_names = (list(source1_df[name_col]) + list(source2_df[name_col]) +
                 list(source3_df[name_col]))
    all_addrs = (list(source1_df["business_address"]) +
                 list(source2_df["business_address"]) +
                 list(source3_df["business_address"]))
    name_tfidf = TfidfSimilarity(all_names)
    addr_tfidf = TfidfSimilarity(all_addrs)

    with open(matching_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])

        for s1_id in source1_df[id_col]:
            s1_rec = s1_lookup[s1_id]
            cand_ids = candidates.get(s1_id, [])
            matches = []
            if cand_ids:
                feats = []
                valid_ids = []
                for cid in cand_ids:
                    crec = resolve(cid)
                    if crec is None:
                        continue
                    feats.append(pair_features(s1_rec, crec, name_tfidf, addr_tfidf))
                    valid_ids.append(cid)
                if feats:
                    probs = booster.predict(feats)
                    matches = [vid for vid, p in zip(valid_ids, probs) if p >= threshold]
                    # de-duplicate, preserve order
                    seen = set()
                    matches = [x for x in matches if not (x in seen or seen.add(x))]
            writer.writerow([s1_id, ",".join(matches)])
