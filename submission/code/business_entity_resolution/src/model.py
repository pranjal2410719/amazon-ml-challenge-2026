# src/model.py
"""
Entity Resolution Model Skeleton

This module provides a high‑level wrapper around the blocking, feature engineering,
and LightGBM classification pipeline defined in the existing `blocking.py`,
`features.py`, and `train.py` modules.  The wrapper is deliberately lightweight –
it does **not** perform any data cleaning or model training automatically; the
user is expected to call the appropriate functions with prepared DataFrames.

Typical usage:

```python
from src.model import EntityResolutionModel

# 1️⃣ Load your source DataFrames (e.g. via pandas.read_csv with sep='\t')
#    source1_df, source2_df, source3_df, ground_truth = ...

# 2️⃣ Initialise the model wrapper
model = EntityResolutionModel()

# 3️⃣ Generate candidate pairs (blocking)
candidates = model.block(source1_df, source2_df, source3_df)

# 4️⃣ Build training data (optional – only if you have ground truth)
X, y, name_tfidf, addr_tfidf = model.build_training_set(
    source1_df, source2_df, source3_df, ground_truth, candidates
)

# 5️⃣ Train the LightGBM classifier (you may customise hyper‑parameters)
model.train(X, y)

# 6️⃣ Save the trained model and similarity objects for inference
model.save("model.txt", name_tfidf, addr_tfidf)

# 7️⃣ Run inference on the test set
matches, candidate_pairs = model.infer(
    test_source1_df, test_source2_df, test_source3_df, threshold=0.5
)
```

The wrapper simply forwards calls to the underlying utilities, making the
workflow easier to orchestrate in notebooks or scripts while keeping the core
logic in the original modules.
"""

import os
from typing import Tuple, Dict, List
import pandas as pd
import lightgbm as lgb

# Local imports – they live in the same package (`src`)
from .blocking import generate_candidates
from .train import build_training_set, train_model, save_model, load_model
from .predict import run_inference


class EntityResolutionModel:
    """High‑level orchestrator for entity‑resolution pipelines.

    Attributes
    ----------
    booster : lgb.Booster or None
        The trained LightGBM model.  It is ``None`` until :meth:`train` is called
        or a model is loaded via :meth:`load`.
    name_tfidf : object or None
        TF‑IDF similarity helper for business names – populated during training
        and required for inference.
    addr_tfidf : object or None
        TF‑IDF similarity helper for business addresses.
    """

    def __init__(self) -> None:
        self.booster = None
        self.name_tfidf = None
        self.addr_tfidf = None

    # ---------------------------------------------------------------------
    # Blocking / candidate generation
    # ---------------------------------------------------------------------
    def block(
        self,
        source1_df: pd.DataFrame,
        source2_df: pd.DataFrame,
        source3_df: pd.DataFrame,
        **blocking_kwargs,
    ) -> Dict[str, List[str]]:
        """Run the Sorted‑Neighborhood blocking stage.

        Parameters
        ----------
        source1_df, source2_df, source3_df : pd.DataFrame
            DataFrames containing the three source files.  They must include the
            columns defined in the challenge spec (``entity_id``, ``business_name``,
            ``business_address``, ``country``).
        **blocking_kwargs : dict
            Additional arguments forwarded to :func:`generate_candidates` (e.g.
            ``window`` or ``max_candidates``).
        """
        return generate_candidates(
            source1_df, source2_df, source3_df, **blocking_kwargs
        )

    # ---------------------------------------------------------------------
    # Training utilities
    # ---------------------------------------------------------------------
    def build_training_set(
        self,
        source1_df: pd.DataFrame,
        source2_df: pd.DataFrame,
        source3_df: pd.DataFrame,
        ground_truth: Dict[str, List[str]],
        candidates: Dict[str, List[str]],
        max_neg_per_entity: int = 10,
    ) -> Tuple[object, object, object, object]:
        """Wrap :func:`train.build_training_set` to expose the TF‑IDF objects.
        """
        X, y, name_tfidf, addr_tfidf = build_training_set(
            source1_df,
            source2_df,
            source3_df,
            ground_truth,
            candidates,
            max_neg_per_entity=max_neg_per_entity,
        )
        self.name_tfidf = name_tfidf
        self.addr_tfidf = addr_tfidf
        return X, y, name_tfidf, addr_tfidf

    def train(
        self, X, y, params: dict | None = None, model_path: str = "model.txt"
    ) -> None:
        """Train a LightGBM binary classifier and optionally persist it.

        The underlying model is stored in ``self.booster``; the TF‑IDF helpers are
        kept unchanged so they can be re‑used during inference.
        """
        model = train_model(X, y, params=params)
        # LightGBM's ``LGBMClassifier`` stores the booster under ``booster_``
        self.booster = model.booster_
        if model_path:
            save_model(model, model_path)

    # ---------------------------------------------------------------------
    # Persistence helpers
    # ---------------------------------------------------------------------
    def save(self, model_path: str, name_tfidf_path: str, addr_tfidf_path: str) -> None:
        """Serialise the trained model and TF‑IDF objects.

        ``name_tfidf`` and ``addr_tfidf`` are simple Python objects; they are
        saved via ``pickle`` for convenience.
        """
        if self.booster is None:
            raise RuntimeError("No trained booster to save – call train() first.")
        # Save LightGBM model
        self.booster.save_model(model_path)
        # Save TF‑IDF helpers (they are pickle‑compatible)
        import pickle

        with open(name_tfidf_path, "wb") as f:
            pickle.dump(self.name_tfidf, f)
        with open(addr_tfidf_path, "wb") as f:
            pickle.dump(self.addr_tfidf, f)

    def load(self, model_path: str, name_tfidf_path: str, addr_tfidf_path: str) -> None:
        """Load a previously saved model and similarity helpers.
        """
        self.booster = load_model(model_path)
        import pickle

        with open(name_tfidf_path, "rb") as f:
            self.name_tfidf = pickle.load(f)
        with open(addr_tfidf_path, "rb") as f:
            self.addr_tfidf = pickle.load(f)

    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------
    def infer(
        self,
        source1_df: pd.DataFrame,
        source2_df: pd.DataFrame,
        source3_df: pd.DataFrame,
        threshold: float = 0.5,
        candidate_out: str = "output/candidate_pairs.tsv",
        matching_out: str = "output/matching_results.tsv",
        blocking_kwargs: dict | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the full inference pipeline and write the two required TSV files.
        """
        if self.booster is None or self.name_tfidf is None or self.addr_tfidf is None:
            raise RuntimeError("Model and TF‑IDF helpers must be loaded or trained before inference.")
        # ``run_inference`` expects a LightGBM Booster – we have that.
        run_inference(
            source1_df,
            source2_df,
            source3_df,
            self.booster,
            threshold=threshold,
            candidate_out=candidate_out,
            matching_out=matching_out,
            blocking_kwargs=blocking_kwargs or {},
        )
        # Load the generated TSVs as DataFrames for convenient downstream use.
        matches = pd.read_csv(matching_out, sep="\t", dtype=str)
        candidates = pd.read_csv(candidate_out, sep="\t", dtype=str)
        return matches, candidates

"""End of src/model.py"""
