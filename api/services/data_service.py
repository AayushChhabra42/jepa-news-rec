"""Artifact loading and cached accessors for news/user/impression data."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from api.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class DataStore:
    news_by_id: dict[str, dict[str, Any]]
    news_by_idx: dict[int, dict[str, Any]]
    idx_to_news_id: dict[int, str]
    news_id_to_idx: dict[str, int]
    user_sequences: dict[str, dict[str, Any]]
    impressions_by_user: dict[str, list[dict[str, Any]]]
    text_embeddings: np.ndarray
    cat_ids: np.ndarray
    subcat_ids: np.ndarray
    entity_flags: np.ndarray
    global_ctr: np.ndarray
    num_categories: int
    num_subcategories: int


_store: DataStore | None = None


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _article_id(article: dict[str, Any], fallback_idx: int | None = None) -> str:
    value = article.get("news_id") or article.get("article_id") or article.get("id")
    if value is not None:
        return str(value)
    return str(fallback_idx) if fallback_idx is not None else ""


def _normalise_news(news_obj: Any, vocabs: dict[str, Any] | None, text_embeddings: np.ndarray) -> tuple[dict, dict, dict]:
    news_id_to_idx = (vocabs or {}).get("news_id2idx", {}) or {}
    idx_to_news_id = {idx: nid for nid, idx in news_id_to_idx.items() if nid != "<PAD>"}

    news_by_id: dict[str, dict[str, Any]] = {}
    news_by_idx: dict[int, dict[str, Any]] = {}

    if hasattr(news_obj, "to_dict"):
        news_obj = news_obj.to_dict("records")
    if isinstance(news_obj, dict) and "articles" in news_obj:
        news_obj = news_obj["articles"]

    if isinstance(news_obj, dict):
        items = news_obj.items()
    else:
        items = enumerate(news_obj or [])

    for key, raw in items:
        if raw is None:
            continue
        article = dict(raw)
        aid = _article_id(article, key if isinstance(key, int) else None)
        if not aid and not isinstance(key, int):
            aid = str(key)
        article.setdefault("news_id", aid)
        article.setdefault("title", f"Article {aid}")
        article.setdefault("category", "unknown")
        article.setdefault("subcategory", "unknown")
        article.setdefault("abstract", "")
        idx = news_id_to_idx.get(aid)
        if idx is None and isinstance(key, int):
            idx = key
        if idx is not None:
            news_by_idx[int(idx)] = article
            idx_to_news_id[int(idx)] = aid
            news_id_to_idx.setdefault(aid, int(idx))
        news_by_id[aid] = article

    if not news_by_idx and news_id_to_idx:
        for aid, idx in news_id_to_idx.items():
            if aid == "<PAD>":
                continue
            article = {
                "news_id": aid,
                "title": f"Article {aid}",
                "category": "unknown",
                "subcategory": "unknown",
                "abstract": "",
            }
            news_by_id[aid] = article
            news_by_idx[idx] = article
            idx_to_news_id[idx] = aid

    if not news_by_idx:
        for idx in range(1, len(text_embeddings)):
            aid = str(idx)
            article = {
                "news_id": aid,
                "title": f"Article {aid}",
                "category": "unknown",
                "subcategory": "unknown",
                "abstract": "",
            }
            news_by_id[aid] = article
            news_by_idx[idx] = article
            idx_to_news_id[idx] = aid
            news_id_to_idx[aid] = idx

    return news_by_id, news_by_idx, idx_to_news_id


def _parse_news_tsv(path: Path) -> dict[str, dict[str, Any]]:
    news: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                parts += [""] * (8 - len(parts))
            news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities = parts[:8]
            news[news_id] = {
                "news_id": news_id,
                "category": category or "unknown",
                "subcategory": subcategory or "unknown",
                "title": title or news_id,
                "abstract": abstract or "",
                "url": url or "",
                "title_entities": title_entities,
                "abstract_entities": abstract_entities,
            }
    return news


def _find_raw_news(settings: Settings) -> dict[str, dict[str, Any]]:
    candidates = [
        settings.processed_dir.parent / "raw" / "mind-small" / "train" / "news.tsv",
        settings.processed_dir.parent / "raw" / "mind-small" / "dev" / "news.tsv",
        settings.processed_dir.parent / "raw" / "MINDsmall_train" / "news.tsv",
        settings.processed_dir.parent / "raw" / "MINDsmall_dev" / "news.tsv",
    ]
    for news_path in settings.processed_dir.parent.glob("raw/**/news.tsv"):
        candidates.append(news_path)

    merged: dict[str, dict[str, Any]] = {}
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        parsed = _parse_news_tsv(path)
        merged.update(parsed)
        logger.info("Loaded %s article metadata rows from %s", len(parsed), path)
    return merged


def _merge_news_metadata(
    news_by_id: dict[str, dict[str, Any]],
    news_by_idx: dict[int, dict[str, Any]],
    idx_to_news_id: dict[int, str],
    metadata_by_id: dict[str, dict[str, Any]],
) -> None:
    for idx, news_id in idx_to_news_id.items():
        metadata = metadata_by_id.get(news_id)
        if not metadata:
            continue
        existing = news_by_idx.get(idx, {"news_id": news_id})
        merged = {**existing, **metadata}
        news_by_idx[idx] = merged
        news_by_id[news_id] = merged


def _extract_embeddings(news_obj: Any) -> np.ndarray | None:
    if hasattr(news_obj, "to_dict"):
        records = news_obj.to_dict("records")
        for column in ("embedding", "embeddings", "minilm_embedding", "text_embedding"):
            values = [row.get(column) for row in records if row.get(column) is not None]
            if values:
                return np.asarray(values, dtype=np.float32)
        return None
    if isinstance(news_obj, dict):
        for key in ("embeddings", "text_embeddings", "minilm_embeddings"):
            if news_obj.get(key) is not None:
                return np.asarray(news_obj[key], dtype=np.float32)
        values = list(news_obj.values())
        if values and isinstance(values[0], dict):
            for key in ("embedding", "embeddings", "minilm_embedding", "text_embedding"):
                embeds = [row.get(key) for row in values if row.get(key) is not None]
                if embeds:
                    return np.asarray(embeds, dtype=np.float32)
    return None


def _normalise_users(users_obj: Any) -> dict[str, dict[str, Any]]:
    if isinstance(users_obj, dict):
        return users_obj
    normalised: dict[str, dict[str, Any]] = {}
    for row in users_obj or []:
        uid = str(row.get("user_id"))
        history = row.get("history_ids") or row.get("history") or []
        normalised[uid] = {"history_ids": history, "impressions": row.get("impressions", [])}
    return normalised


def _normalise_impressions(raw: Any, user_sequences: dict[str, dict[str, Any]], news_id_to_idx: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    if raw is None:
        return {uid: data.get("impressions", []) for uid, data in user_sequences.items()}
    if isinstance(raw, dict):
        return raw

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in raw:
        uid = str(row.get("user_id"))
        candidates = row.get("candidates")
        labels = row.get("labels")
        if candidates is None and row.get("impressions") is not None:
            candidates, labels = [], []
            for nid, label in row["impressions"]:
                candidates.append(news_id_to_idx.get(nid, nid))
                labels.append(label)
        grouped.setdefault(uid, []).append(
            {
                "history_ids": row.get("history_ids") or row.get("history") or [],
                "candidates": candidates or [],
                "labels": labels or [],
                "time": row.get("time"),
            }
        )
    return grouped


def _global_ctr(impressions_by_user: dict[str, list[dict[str, Any]]], size: int) -> np.ndarray:
    shown = np.zeros(size, dtype=np.float32)
    clicked = np.zeros(size, dtype=np.float32)
    for impressions in impressions_by_user.values():
        for impression in impressions:
            for idx, label in zip(impression.get("candidates", []), impression.get("labels", [])):
                try:
                    i = int(idx)
                except (TypeError, ValueError):
                    continue
                if 0 <= i < size:
                    shown[i] += 1.0
                    clicked[i] += float(label)
    return np.divide(clicked, np.maximum(shown, 1.0), dtype=np.float32)


def load_data(settings: Settings) -> DataStore:
    global _store
    if _store is not None:
        return _store

    if settings.processed_data_path.exists():
        logger.info("Loading processed bundle from %s", settings.processed_data_path)
        bundle = _load_pickle(settings.processed_data_path)
        text_embeddings = np.asarray(bundle["text_embeddings"], dtype=np.float32)
        features = bundle["article_features"]
        vocabs = bundle.get("vocabs", {})
        users = _normalise_users(bundle["train_users"])
        news_obj = bundle.get("news") or bundle.get("news_dict")
        if settings.news_path.exists():
            news_obj = _load_pickle(settings.news_path)
        news_by_id, news_by_idx, idx_to_news_id = _normalise_news(news_obj, vocabs, text_embeddings)
        _merge_news_metadata(news_by_id, news_by_idx, idx_to_news_id, _find_raw_news(settings))
        impressions = _normalise_impressions(None, users, vocabs.get("news_id2idx", {}))
        news_id_to_idx = dict(vocabs.get("news_id2idx", {}))
        cat_ids = np.asarray(features["cat_ids"], dtype=np.int64)
        subcat_ids = np.asarray(features["subcat_ids"], dtype=np.int64)
        entity_flags = np.asarray(features["entity_flags"], dtype=np.float32)
        num_categories = len(vocabs.get("cat2idx", [])) or int(cat_ids.max()) + 1
        num_subcategories = len(vocabs.get("subcat2idx", [])) or int(subcat_ids.max()) + 1
    else:
        required = [settings.news_path, settings.user_sequences_path, settings.impressions_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing processed artifacts. Expected processed_data.pkl or split files: "
                + ", ".join(missing)
            )
        news_obj = _load_pickle(settings.news_path)
        users = _normalise_users(_load_pickle(settings.user_sequences_path))
        raw_impressions = _load_pickle(settings.impressions_path)
        embedding_obj = _extract_embeddings(news_obj)
        if embedding_obj is None:
            raise ValueError(f"No embeddings found in {settings.news_path}")
        text_embeddings = embedding_obj.astype(np.float32)
        news_id_to_idx = news_obj.get("news_id2idx", {}) if isinstance(news_obj, dict) else {}
        news_by_id, news_by_idx, idx_to_news_id = _normalise_news(news_obj, {"news_id2idx": news_id_to_idx}, text_embeddings)
        _merge_news_metadata(news_by_id, news_by_idx, idx_to_news_id, _find_raw_news(settings))
        impressions = _normalise_impressions(raw_impressions, users, news_id_to_idx)
        if isinstance(news_obj, dict):
            cat_ids = np.asarray(news_obj.get("cat_ids", np.zeros(len(text_embeddings))), dtype=np.int64)
            subcat_ids = np.asarray(news_obj.get("subcat_ids", np.zeros(len(text_embeddings))), dtype=np.int64)
            entity_flags = np.asarray(news_obj.get("entity_flags", np.zeros(len(text_embeddings))), dtype=np.float32)
        else:
            cat_ids = np.zeros(len(text_embeddings), dtype=np.int64)
            subcat_ids = np.zeros(len(text_embeddings), dtype=np.int64)
            entity_flags = np.zeros(len(text_embeddings), dtype=np.float32)
        num_categories = int(cat_ids.max()) + 1
        num_subcategories = int(subcat_ids.max()) + 1

    if "<PAD>" in news_id_to_idx:
        news_id_to_idx = {k: int(v) for k, v in news_id_to_idx.items()}
    else:
        news_id_to_idx = {aid: idx for idx, aid in idx_to_news_id.items()}

    _store = DataStore(
        news_by_id=news_by_id,
        news_by_idx=news_by_idx,
        idx_to_news_id={int(k): str(v) for k, v in idx_to_news_id.items()},
        news_id_to_idx=news_id_to_idx,
        user_sequences=users,
        impressions_by_user=impressions,
        text_embeddings=text_embeddings.astype(np.float32),
        cat_ids=cat_ids,
        subcat_ids=subcat_ids,
        entity_flags=entity_flags.astype(np.float32),
        global_ctr=_global_ctr(impressions, len(text_embeddings)),
        num_categories=num_categories,
        num_subcategories=num_subcategories,
    )
    logger.info("Loaded %s users and %s articles", len(_store.user_sequences), len(_store.news_by_idx))
    return _store


def get_store() -> DataStore:
    if _store is None:
        raise RuntimeError("DataStore has not been loaded")
    return _store


def get_user(user_id: str) -> dict[str, Any] | None:
    return get_store().user_sequences.get(user_id)


def article_for_idx(idx: int) -> dict[str, Any]:
    store = get_store()
    return store.news_by_idx.get(
        int(idx),
        {
            "news_id": store.idx_to_news_id.get(int(idx), str(idx)),
            "title": f"Article {idx}",
            "category": "unknown",
            "subcategory": "unknown",
            "abstract": "",
        },
    )


def article_id_for_idx(idx: int) -> str:
    return _article_id(article_for_idx(idx), idx)


def idx_for_article_id(article_id: str) -> int | None:
    value = get_store().news_id_to_idx.get(article_id)
    return int(value) if value is not None else None


def labels_for_user(user_id: str) -> dict[int, int]:
    labels: dict[int, int] = {}
    for impression in get_store().impressions_by_user.get(user_id, []):
        for idx, label in zip(impression.get("candidates", []), impression.get("labels", [])):
            try:
                labels[int(idx)] = int(label)
            except (TypeError, ValueError):
                article_idx = idx_for_article_id(str(idx))
                if article_idx is not None:
                    labels[article_idx] = int(label)
    return labels


def labeled_candidate_ids_for_user(user_id: str) -> list[int]:
    candidates: dict[int, None] = {}
    for impression in get_store().impressions_by_user.get(user_id, []):
        for idx in impression.get("candidates", []):
            try:
                article_idx = int(idx)
            except (TypeError, ValueError):
                resolved = idx_for_article_id(str(idx))
                if resolved is None:
                    continue
                article_idx = resolved
            if article_idx > 0:
                candidates[article_idx] = None
    return list(candidates.keys())


def profile_features(history_ids: list[int]) -> dict[str, Any]:
    total = max(len(history_ids), 1)
    articles = [article_for_idx(idx) for idx in history_ids]
    cats = Counter(a.get("category") or "unknown" for a in articles)
    subcats = Counter(a.get("subcategory") or "unknown" for a in articles)
    last_bucket = max(1, int(np.ceil(len(history_ids) * 0.2)))
    return {
        "top_categories": [
            {"category": category, "count": count, "pct": round(count / total, 4)}
            for category, count in cats.most_common(8)
        ],
        "top_subcategories": [
            {"subcategory": subcategory, "count": count, "pct": round(count / total, 4)}
            for subcategory, count in subcats.most_common(8)
        ],
        "avg_session_length": float(len(history_ids)),
        "recency_bias_score": round(last_bucket / total, 4) if history_ids else 0.0,
    }
