"""
Pairwise feature engineering.

Runs ONLY on candidate pairs that survived blocking (small set), never on
the full cross product. All features are generic string-similarity /
structural signals -- none are tied to specific country values, so they
generalize to unseen countries (e.g. France).
"""
import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from .normalize import normalize_name, normalize_address, name_tokens, address_tokens

FEATURE_NAMES = [
    "name_ratio", "name_partial_ratio", "name_token_sort_ratio",
    "name_jaccard", "name_tfidf_cosine",
    "addr_ratio", "addr_jaccard", "addr_tfidf_cosine",
    "country_match", "name_len_diff", "name_exact_norm_match",
]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class TfidfSimilarity:
    """Fits a char n-gram TF-IDF vectorizer once over a corpus of strings,
    then gives cosine similarity between any two strings in that corpus
    (looked up by precomputed vector) or freshly-transformed strings."""

    def __init__(self, corpus, ngram_range=(2, 4)):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram_range, min_df=1
        )
        self.matrix = self.vectorizer.fit_transform(corpus)

    def cosine(self, text_a: str, text_b: str) -> float:
        va = self.vectorizer.transform([text_a])
        vb = self.vectorizer.transform([text_b])
        num = (va.multiply(vb)).sum()
        denom = np.sqrt(va.multiply(va).sum()) * np.sqrt(vb.multiply(vb).sum())
        if denom == 0:
            return 0.0
        return float(num / denom)


def pair_features(s1_record: dict, cand_record: dict, name_tfidf: TfidfSimilarity,
                   addr_tfidf: TfidfSimilarity) -> list:
    """
    s1_record / cand_record: dicts with keys business_name, business_address, country
    Returns a feature vector in the order of FEATURE_NAMES.
    """
    n1_raw, n2_raw = s1_record["business_name"], cand_record["business_name"]
    a1_raw, a2_raw = s1_record["business_address"], cand_record["business_address"]

    n1, n2 = normalize_name(n1_raw), normalize_name(n2_raw)
    a1, a2 = normalize_address(a1_raw), normalize_address(a2_raw)

    feats = [
        fuzz.ratio(n1, n2) / 100.0,
        fuzz.partial_ratio(n1, n2) / 100.0,
        fuzz.token_sort_ratio(n1, n2) / 100.0,
        _jaccard(name_tokens(n1_raw), name_tokens(n2_raw)),
        name_tfidf.cosine(n1, n2),
        fuzz.ratio(a1, a2) / 100.0,
        _jaccard(address_tokens(a1_raw), address_tokens(a2_raw)),
        addr_tfidf.cosine(a1, a2),
        1.0 if str(s1_record["country"]).strip().lower() ==
               str(cand_record["country"]).strip().lower() else 0.0,
        abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1),
        1.0 if n1 == n2 else 0.0,
    ]
    return feats
