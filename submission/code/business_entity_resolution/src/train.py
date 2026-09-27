"""
Trains the matching-stage GBM classifier.

Training pairs:
  - Positives: every (S1, matched_id) pair from ground truth.
  - Negatives: HARD negatives -- candidates that survived blocking for that
    S1 entity but are NOT in ground truth. These are far more informative
    than random negatives because they are exactly the confusions the model
    will face at inference time.

Model: LightGBM binary classifier over the engineered similarity features
(see features.py). Trained entirely from scratch on competition data --
no pretrained weights, so no license/param-count concerns apply.
"""
import json
import numpy as np
import lightgbm as lgb
from .features import pair_features, TfidfSimilarity, FEATURE_NAMES


def _record_lookup(df, id_col="entity_id"):
    return df.set_index(id_col).to_dict("index")


def build_training_set(source1_df, source2_df, source3_df, ground_truth: dict,
                        candidates: dict, max_neg_per_entity: int = 10,
                        id_col="entity_id", name_col="business_name"):
    """
    ground_truth: {s1_id: [matched_ids]}
    candidates:   {s1_id: [candidate_ids]}   (blocking output)

    Returns X (features), y (labels), plus the fitted TF-IDF similarity
    objects (needed again at inference time -- persist them alongside the
    model).
    """
    s1_lookup = _record_lookup(source1_df, id_col)
    s2_lookup = _record_lookup(source2_df, id_col)
    s3_lookup = _record_lookup(source3_df, id_col)

    def resolve(eid):
        if eid.startswith("S2-"):
            return s2_lookup.get(eid)
        if eid.startswith("S3-"):
            return s3_lookup.get(eid)
        return None

    # Fit TF-IDF similarity corpora once, over ALL names / addresses across
    # sources, so vector spaces are shared and comparable.
    all_names = (list(source1_df[name_col]) + list(source2_df[name_col]) +
                 list(source3_df[name_col]))
    all_addrs = (list(source1_df["business_address"]) +
                 list(source2_df["business_address"]) +
                 list(source3_df["business_address"]))
    name_tfidf = TfidfSimilarity(all_names)
    addr_tfidf = TfidfSimilarity(all_addrs)

    X, y = [], []
    rng = np.random.default_rng(42)

    for s1_id, truth_ids in ground_truth.items():
        s1_rec = s1_lookup.get(s1_id)
        if s1_rec is None:
            continue
        truth_set = set(truth_ids)
        cand_ids = set(candidates.get(s1_id, [])) | truth_set  # ensure positives included

        pos_ids = [c for c in cand_ids if c in truth_set]
        neg_ids = [c for c in cand_ids if c not in truth_set]

        if len(neg_ids) > max_neg_per_entity:
            neg_ids = list(rng.choice(neg_ids, size=max_neg_per_entity, replace=False))

        for cid in pos_ids:
            crec = resolve(cid)
            if crec is None:
                continue
            X.append(pair_features(s1_rec, crec, name_tfidf, addr_tfidf))
            y.append(1)

        for cid in neg_ids:
            crec = resolve(cid)
            if crec is None:
                continue
            X.append(pair_features(s1_rec, crec, name_tfidf, addr_tfidf))
            y.append(0)

    return np.array(X), np.array(y), name_tfidf, addr_tfidf


def train_model(X, y, params: dict = None):
    default_params = dict(
        objective="binary",
        metric="auc",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=300,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)
    model.fit(X, y, feature_name=FEATURE_NAMES)
    return model


def save_model(model, path="model.txt"):
    model.booster_.save_model(path)


def load_model(path="model.txt"):
    booster = lgb.Booster(model_file=path)
    return booster
