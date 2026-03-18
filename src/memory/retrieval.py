from __future__ import annotations
from typing import List
import numpy as np
from src.models.nodes import CaseEntry, TaskTraits


def _trait_vector(traits: TaskTraits) -> np.ndarray:
    task_enc = {"binary": [1, 0, 0], "multiclass": [0, 1, 0], "regression": [0, 0, 1]}
    task_vec = task_enc.get(traits.task_type, [0, 0, 0])

    bucket_score = {"small": 0.0, "medium": 0.5, "large": 1.0}
    rows_score = bucket_score.get(traits.n_rows_bucket, 0.5)
    feat_score = bucket_score.get(traits.n_features_bucket, 0.5)

    balance_score = {"balanced": 1.0, "moderate": 0.5, "moderate_imbalance": 0.5,
                     "severe": 0.0, "severe_imbalance": 0.0}.get(traits.class_balance, 0.5)

    total = sum(traits.feature_types.values()) or 1
    numeric_ratio = traits.feature_types.get("numeric", 0) / total

    return np.array(task_vec + [rows_score, feat_score, balance_score, numeric_ratio],
                    dtype=float)


class CaseRetriever:
    """Ranks CaseEntry candidates by cosine similarity on task traits."""

    def rank(self, query_traits: TaskTraits, candidates: List[CaseEntry],
             top_k: int = 3) -> List[CaseEntry]:
        if not candidates:
            return []

        q_vec = _trait_vector(query_traits).reshape(1, -1)
        c_vecs = np.array([_trait_vector(c.task_traits) for c in candidates])

        q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
        c_norm = np.linalg.norm(c_vecs, axis=1, keepdims=True)
        q_safe = np.where(q_norm == 0, 1, q_norm)
        c_safe = np.where(c_norm == 0, 1, c_norm)
        sims = (q_vec / q_safe) @ (c_vecs / c_safe).T
        sims = sims.flatten()

        top_indices = np.argsort(sims)[::-1][:top_k]
        return [candidates[i] for i in top_indices]
