"""XGBoost reranking service for JEPA candidate sets."""

from __future__ import annotations

from collections.abc import Sequence
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from api.config import Settings
from api.services import data_service

logger = logging.getLogger(__name__)


class XGBService:
    def __init__(self, settings: Settings, store: data_service.DataStore):
        self.settings = settings
        self.store = store
        self.model = self._load_model(settings.xgb_checkpoint_path)
        self.num_category_features = self._infer_num_category_features()
        self.user_profiles = self._build_user_profiles()
        self.impression_positions = self._build_impression_positions()

    def _load_model(self, path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(
                f"XGBoost checkpoint not found: {path}. "
                "Set JEPA_XGB_CHECKPOINT_PATH or train baselines/xgboost_ranker.py."
            )

        suffix = path.suffix.lower()
        logger.info("Loading XGBoost reranker from %s", path)
        if suffix == ".pkl":
            with path.open("rb") as f:
                return pickle.load(f)
        if suffix == ".json":
            try:
                import xgboost as xgb
            except ImportError as exc:
                raise RuntimeError("xgboost is required to load JSON XGBRanker checkpoints") from exc

            model = xgb.XGBRanker()
            model.load_model(path)
            return model

        raise ValueError(f"Unsupported XGBoost checkpoint extension: {path.suffix}")

    def _infer_num_category_features(self) -> int:
        booster = getattr(self.model, "get_booster", lambda: None)()
        expected_features = None
        if booster is not None:
            expected_features = getattr(booster, "num_features", lambda: None)()
        if expected_features:
            return max(int(expected_features) - 6, 0)
        return int(self.store.num_categories)

    def _valid_history_ids(self, history_ids: Sequence[Any]) -> list[int]:
        valid = []
        for raw_idx in history_ids:
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                resolved = data_service.idx_for_article_id(str(raw_idx))
                if resolved is None:
                    continue
                idx = resolved
            if 0 < idx < len(self.store.text_embeddings):
                valid.append(idx)
        return valid

    def _build_user_profiles(self) -> dict[str, dict[str, Any]]:
        profiles: dict[str, dict[str, Any]] = {}
        embedding_dim = int(self.store.text_embeddings.shape[1])
        for user_id, user_data in self.store.user_sequences.items():
            history_ids = self._valid_history_ids(user_data.get("history_ids", []))
            click_rate = np.zeros(self.num_category_features, dtype=np.float32)
            mean_embedding = np.zeros(embedding_dim, dtype=np.float32)
            top_categories: set[int] = set()

            if history_ids:
                category_counts = np.zeros(self.num_category_features, dtype=np.float32)
                for idx in history_ids:
                    cat_id = int(self.store.cat_ids[idx]) if idx < len(self.store.cat_ids) else 0
                    if 0 <= cat_id < self.num_category_features:
                        category_counts[cat_id] += 1.0
                total = float(category_counts.sum())
                if total > 0:
                    click_rate = category_counts / total
                    top_categories = {int(i) for i in np.argsort(-category_counts)[:3]}
                mean_embedding = self.store.text_embeddings[history_ids].mean(axis=0).astype(np.float32)

            profiles[user_id] = {
                "click_rate_per_cat": click_rate,
                "num_clicks": len(history_ids),
                "mean_embedding": mean_embedding,
                "top_categories": top_categories,
            }
        return profiles

    def _build_impression_positions(self) -> dict[str, dict[int, int]]:
        positions: dict[str, dict[int, int]] = {}
        for user_id, impressions in self.store.impressions_by_user.items():
            user_positions: dict[int, int] = {}
            for impression in impressions:
                for pos, candidate in enumerate(impression.get("candidates", [])):
                    try:
                        idx = int(candidate)
                    except (TypeError, ValueError):
                        resolved = data_service.idx_for_article_id(str(candidate))
                        if resolved is None:
                            continue
                        idx = resolved
                    user_positions.setdefault(idx, pos)
            positions[user_id] = user_positions
        return positions

    def _feature_for_candidate(self, user_id: str, candidate: dict[str, Any], fallback_pos: int) -> np.ndarray:
        article_idx = int(candidate["article_idx"])
        profile = self.user_profiles.get(user_id)
        if profile is None:
            profile = {
                "click_rate_per_cat": np.zeros(self.num_category_features, dtype=np.float32),
                "num_clicks": 0,
                "mean_embedding": np.zeros(self.store.text_embeddings.shape[1], dtype=np.float32),
                "top_categories": set(),
            }

        feat = np.zeros(self.num_category_features + 6, dtype=np.float32)
        feat[: self.num_category_features] = profile["click_rate_per_cat"][: self.num_category_features]
        offset = self.num_category_features
        feat[offset] = float(profile["num_clicks"])

        if 0 <= article_idx < len(self.store.global_ctr):
            feat[offset + 1] = float(self.store.global_ctr[article_idx])

        if 0 <= article_idx < len(self.store.text_embeddings):
            user_emb = profile["mean_embedding"]
            cand_emb = self.store.text_embeddings[article_idx]
            denom = float(np.linalg.norm(user_emb) * np.linalg.norm(cand_emb))
            if denom > 0:
                feat[offset + 2] = float(np.dot(user_emb, cand_emb) / denom)

        cand_cat = int(self.store.cat_ids[article_idx]) if 0 <= article_idx < len(self.store.cat_ids) else 0
        feat[offset + 3] = 1.0 if cand_cat in profile["top_categories"] else 0.0
        feat[offset + 4] = float(self.store.entity_flags[article_idx]) if 0 <= article_idx < len(self.store.entity_flags) else 0.0
        feat[offset + 5] = float(self.impression_positions.get(user_id, {}).get(article_idx, fallback_pos))
        return feat

    def rerank(self, user_id: str, jepa_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not jepa_candidates:
            return []

        features = np.vstack(
            [
                self._feature_for_candidate(user_id, candidate, pos)
                for pos, candidate in enumerate(jepa_candidates)
            ]
        ).astype(np.float32)
        scores = np.asarray(self.model.predict(features), dtype=np.float32)

        enriched = []
        for candidate, score in zip(jepa_candidates, scores):
            enriched.append({**candidate, "xgb_score": float(score)})

        enriched.sort(key=lambda item: item["xgb_score"], reverse=True)
        for rank, item in enumerate(enriched, start=1):
            item["xgb_rank"] = rank
            item["rank_delta"] = int(item["jepa_rank"]) - rank
        return enriched


_service: XGBService | None = None


def load_xgb(settings: Settings, store: data_service.DataStore) -> XGBService:
    global _service
    if _service is None:
        _service = XGBService(settings, store)
    return _service


def get_service(settings: Settings | None = None, store: data_service.DataStore | None = None) -> XGBService:
    if _service is not None:
        return _service
    if settings is None or store is None:
        raise RuntimeError("XGBService has not been loaded")
    return load_xgb(settings, store)


def rerank(user_id: str, jepa_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return get_service().rerank(user_id, jepa_candidates)
