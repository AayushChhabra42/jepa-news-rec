"""Parse MIND TSV files, build user sequences, compute embeddings, and save."""

import os
import json
import pickle
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TSV Parsing
# ---------------------------------------------------------------------------

def parse_news_tsv(path: str) -> dict[str, dict]:
    """Parse news.tsv into a dict keyed by news_id.

    Columns: news_id, category, subcategory, title, abstract, url,
             title_entities, abstract_entities
    """
    news = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip("\n").split("\t")
            if len(parts) < 8:
                parts += [""] * (8 - len(parts))

            nid, cat, subcat, title, abstract, url, title_ents, abs_ents = parts[:8]

            # Parse entity JSON strings
            try:
                title_entities = json.loads(title_ents) if title_ents else []
            except json.JSONDecodeError:
                title_entities = []
            try:
                abstract_entities = json.loads(abs_ents) if abs_ents else []
            except json.JSONDecodeError:
                abstract_entities = []

            news[nid] = {
                "news_id": nid,
                "category": cat,
                "subcategory": subcat,
                "title": title,
                "abstract": abstract,
                "url": url,
                "title_entities": title_entities,
                "abstract_entities": abstract_entities,
            }

    logger.info(f"Parsed {len(news)} news articles from {path}")
    return news


def parse_behaviors_tsv(path: str) -> list[dict]:
    """Parse behaviors.tsv into a list of impression records.

    Columns: impression_id, user_id, time, history, impressions
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip("\n").split("\t")
            if len(parts) < 5:
                parts += [""] * (5 - len(parts))

            imp_id, user_id, time_str, history_str, impressions_str = parts[:5]

            # Parse history — space-separated news IDs
            history = history_str.split() if history_str else []

            # Parse impressions — space-separated "NewsID-Label" pairs
            impressions = []
            if impressions_str:
                for item in impressions_str.split():
                    if "-" in item:
                        nid, label = item.rsplit("-", 1)
                        impressions.append((nid, int(label)))

            records.append({
                "impression_id": imp_id,
                "user_id": user_id,
                "time": time_str,
                "history": history,
                "impressions": impressions,
            })

    logger.info(f"Parsed {len(records)} impression records from {path}")
    return records


# ---------------------------------------------------------------------------
# Vocabulary Building
# ---------------------------------------------------------------------------

def build_vocabs(news_dict: dict[str, dict]) -> tuple[dict, dict, dict]:
    """Build vocabularies for news IDs, categories, and subcategories.

    Returns:
        (news_id2idx, cat2idx, subcat2idx) — each maps string → int (0-indexed).
        Index 0 is reserved for <PAD> in each vocab.
    """
    news_ids = sorted(news_dict.keys())
    categories = sorted(set(n["category"] for n in news_dict.values() if n["category"]))
    subcategories = sorted(set(n["subcategory"] for n in news_dict.values() if n["subcategory"]))

    news_id2idx = {"<PAD>": 0}
    for i, nid in enumerate(news_ids, start=1):
        news_id2idx[nid] = i

    cat2idx = {"<PAD>": 0}
    for i, cat in enumerate(categories, start=1):
        cat2idx[cat] = i

    subcat2idx = {"<PAD>": 0}
    for i, sc in enumerate(subcategories, start=1):
        subcat2idx[sc] = i

    logger.info(
        f"Vocab sizes — news: {len(news_id2idx)}, "
        f"categories: {len(cat2idx)}, subcategories: {len(subcat2idx)}"
    )
    return news_id2idx, cat2idx, subcat2idx


# ---------------------------------------------------------------------------
# Text Embedding with MiniLM
# ---------------------------------------------------------------------------

def compute_text_embeddings(
    news_dict: dict[str, dict],
    news_id2idx: dict[str, int],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> np.ndarray:
    """Compute title + abstract embeddings for all articles.

    Returns:
        embeddings: (num_articles + 1, 384) — index 0 is zero-pad vector.
    """
    model = SentenceTransformer(model_name, device=device)
    embed_dim = model.get_sentence_embedding_dimension()
    num_articles = len(news_id2idx)  # includes PAD

    embeddings = np.zeros((num_articles, embed_dim), dtype=np.float32)

    # Build text list aligned with indices
    texts = []
    indices = []
    for nid, idx in news_id2idx.items():
        if nid == "<PAD>":
            continue
        article = news_dict[nid]
        text = article["title"]
        if article["abstract"]:
            text += " [SEP] " + article["abstract"]
        texts.append(text)
        indices.append(idx)

    logger.info(f"Computing embeddings for {len(texts)} articles...")

    # Batch encode
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding articles"):
        batch_texts = texts[start : start + batch_size]
        batch_indices = indices[start : start + batch_size]
        batch_embeds = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
        for idx, emb in zip(batch_indices, batch_embeds):
            embeddings[idx] = emb

    return embeddings


# ---------------------------------------------------------------------------
# Article Feature Building
# ---------------------------------------------------------------------------

def build_article_features(
    news_dict: dict[str, dict],
    news_id2idx: dict[str, int],
    cat2idx: dict[str, int],
    subcat2idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build category, subcategory, and entity flag arrays.

    Returns:
        cat_ids: (num_articles,) — category index per article.
        subcat_ids: (num_articles,) — subcategory index per article.
        entity_flags: (num_articles,) — 1 if article has entities, 0 otherwise.
    """
    num = len(news_id2idx)
    cat_ids = np.zeros(num, dtype=np.int64)
    subcat_ids = np.zeros(num, dtype=np.int64)
    entity_flags = np.zeros(num, dtype=np.float32)

    for nid, idx in news_id2idx.items():
        if nid == "<PAD>":
            continue
        article = news_dict[nid]
        cat_ids[idx] = cat2idx.get(article["category"], 0)
        subcat_ids[idx] = subcat2idx.get(article["subcategory"], 0)
        has_entities = len(article["title_entities"]) > 0 or len(article["abstract_entities"]) > 0
        entity_flags[idx] = 1.0 if has_entities else 0.0

    return cat_ids, subcat_ids, entity_flags


# ---------------------------------------------------------------------------
# User Sequence Construction
# ---------------------------------------------------------------------------

def build_user_sequences(
    behaviors: list[dict],
    news_id2idx: dict[str, int],
    max_seq_len: int = 50,
) -> dict[str, dict]:
    """Build per-user click sequences from behavior logs.

    Aggregates the history field across all impressions for each user,
    preserving temporal order (as given in the data — already sorted).

    Also collects impression-level data for fine-tuning.

    Returns:
        Dict mapping user_id → {
            "history_ids": list[int],   # full deduplicated click history as indices
            "impressions": list[dict],  # [{history_ids, candidates, labels}, ...]
        }
    """
    user_data = defaultdict(lambda: {"seen_history": [], "impressions": []})

    for record in behaviors:
        uid = record["user_id"]
        # History — convert to indices, skip unknown articles
        hist_ids = [
            news_id2idx[nid] for nid in record["history"]
            if nid in news_id2idx
        ]

        # Extend user's known history (deduplicate while preserving order)
        existing = set(user_data[uid]["seen_history"])
        for hid in hist_ids:
            if hid not in existing:
                user_data[uid]["seen_history"].append(hid)
                existing.add(hid)

        # Impression-level data for fine-tuning
        candidates = []
        labels = []
        for nid, label in record["impressions"]:
            if nid in news_id2idx:
                candidates.append(news_id2idx[nid])
                labels.append(label)

        if candidates:
            user_data[uid]["impressions"].append({
                "history_ids": hist_ids[-max_seq_len:],  # Truncate history
                "candidates": candidates,
                "labels": labels,
            })

    # Finalise: truncate full histories
    result = {}
    for uid, data in user_data.items():
        history = data["seen_history"][-max_seq_len:]  # Keep most recent
        result[uid] = {
            "history_ids": history,
            "impressions": data["impressions"],
        }

    logger.info(
        f"Built sequences for {len(result)} users. "
        f"Avg history len: {np.mean([len(v['history_ids']) for v in result.values()]):.1f}"
    )
    return result


# ---------------------------------------------------------------------------
# Co-Click Pair Extraction (for SimCSE)
# ---------------------------------------------------------------------------

def extract_coclick_pairs(
    behaviors: list[dict],
    news_id2idx: dict[str, int],
) -> list[tuple[int, int]]:
    """Extract co-click pairs for SimCSE pretraining.

    A co-click pair is two articles clicked by the same user in the
    same impression. Also generates pairs from consecutive history items.

    Returns:
        List of (article_idx_a, article_idx_b) pairs.
    """
    pairs = set()

    for record in behaviors:
        # Co-clicks within the same impression
        clicked = [
            news_id2idx[nid] for nid, label in record["impressions"]
            if label == 1 and nid in news_id2idx
        ]
        for i in range(len(clicked)):
            for j in range(i + 1, len(clicked)):
                pair = tuple(sorted([clicked[i], clicked[j]]))
                pairs.add(pair)

        # Consecutive pairs from history
        hist_ids = [
            news_id2idx[nid] for nid in record["history"]
            if nid in news_id2idx
        ]
        for i in range(len(hist_ids) - 1):
            pair = tuple(sorted([hist_ids[i], hist_ids[i + 1]]))
            pairs.add(pair)

    pairs = list(pairs)
    logger.info(f"Extracted {len(pairs)} unique co-click pairs")
    return pairs


# ---------------------------------------------------------------------------
# Hard Negative Extraction (for SimCSE)
# ---------------------------------------------------------------------------

def extract_hard_negatives(
    behaviors: list[dict],
    news_id2idx: dict[str, int],
) -> dict[int, list[int]]:
    """Extract hard negatives: articles displayed but not clicked.

    For each clicked article, collects non-clicked articles from the
    same impression.

    Returns:
        Dict mapping article_idx → list of hard negative article_idxs.
    """
    hard_negs = defaultdict(set)

    for record in behaviors:
        clicked = set()
        not_clicked = set()
        for nid, label in record["impressions"]:
            if nid not in news_id2idx:
                continue
            idx = news_id2idx[nid]
            if label == 1:
                clicked.add(idx)
            else:
                not_clicked.add(idx)

        for c in clicked:
            hard_negs[c].update(not_clicked)

    result = {k: list(v)[:50] for k, v in hard_negs.items()}  # Cap per article
    logger.info(f"Extracted hard negatives for {len(result)} articles")
    return result


# ---------------------------------------------------------------------------
# Main Preprocessing Pipeline
# ---------------------------------------------------------------------------

def preprocess_mind(
    raw_dir: str,
    processed_dir: str,
    dataset: str = "mind-small",
    max_seq_len: int = 50,
    minilm_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict:
    """Run the full preprocessing pipeline.

    Args:
        raw_dir: Path to raw MIND data (containing train/ and dev/ subdirs).
        processed_dir: Path to save processed outputs.
        dataset: "mind-small" or "mind-large".
        max_seq_len: Maximum sequence length.
        minilm_model: Sentence transformer model name.

    Returns:
        Dict with paths to all processed files.
    """
    os.makedirs(processed_dir, exist_ok=True)

    train_dir = os.path.join(raw_dir, dataset, "train")

    # --- Parse news ---
    print("=" * 60)
    print("Stage 1/6: Parsing news articles")
    print("=" * 60)
    train_news = parse_news_tsv(os.path.join(train_dir, "news.tsv"))
    print(f"  Total unique articles: {len(train_news)}")

    # --- Build vocabs ---
    print("\nStage 2/6: Building vocabularies")
    news_id2idx, cat2idx, subcat2idx = build_vocabs(train_news)

    # --- Compute text embeddings ---
    print("\nStage 3/6: Computing MiniLM text embeddings")
    emb_path = os.path.join(processed_dir, "text_embeddings.npy")
    if os.path.exists(emb_path):
        print(f"  Loading cached embeddings from {emb_path}")
        text_embeddings = np.load(emb_path)
    else:
        text_embeddings = compute_text_embeddings(
            train_news, news_id2idx, model_name=minilm_model
        )
        np.save(emb_path, text_embeddings)
        print(f"  Saved embeddings to {emb_path}")
    print(f"  Embedding matrix shape: {text_embeddings.shape}")

    # --- Build article features ---
    print("\nStage 4/6: Building article features")
    cat_ids, subcat_ids, entity_flags = build_article_features(
        train_news, news_id2idx, cat2idx, subcat2idx
    )

    # --- Parse behaviors and build sequences ---
    print("\nStage 5/6: Parsing behaviors and building user sequences")
    train_behaviors = parse_behaviors_tsv(os.path.join(train_dir, "behaviors.tsv"))
    train_users = build_user_sequences(train_behaviors, news_id2idx, max_seq_len)

    # Stats
    train_hist_lens = [len(v["history_ids"]) for v in train_users.values()]
    print(f"  Train users: {len(train_users)}")
    print(f"  Train avg history: {np.mean(train_hist_lens):.1f}, median: {np.median(train_hist_lens):.1f}")

    # --- Extract co-click pairs and hard negatives ---
    print("\nStage 6/6: Extracting co-click pairs and hard negatives")
    coclick_pairs = extract_coclick_pairs(train_behaviors, news_id2idx)
    hard_negatives = extract_hard_negatives(train_behaviors, news_id2idx)

    # --- Save everything ---
    print("\n" + "=" * 60)
    print("Saving processed data")
    print("=" * 60)

    outputs = {
        "vocabs": {
            "news_id2idx": news_id2idx,
            "cat2idx": cat2idx,
            "subcat2idx": subcat2idx,
        },
        "article_features": {
            "cat_ids": cat_ids,
            "subcat_ids": subcat_ids,
            "entity_flags": entity_flags,
        },
        "text_embeddings": text_embeddings,
        "train_users": train_users,
        "coclick_pairs": coclick_pairs,
        "hard_negatives": hard_negatives,
    }

    save_path = os.path.join(processed_dir, "processed_data.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(outputs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved all processed data to {save_path}")

    # Save vocabs separately for easy access
    vocab_path = os.path.join(processed_dir, "vocabs.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(outputs["vocabs"], f)

    # Save numpy arrays separately
    np.save(os.path.join(processed_dir, "cat_ids.npy"), cat_ids)
    np.save(os.path.join(processed_dir, "subcat_ids.npy"), subcat_ids)
    np.save(os.path.join(processed_dir, "entity_flags.npy"), entity_flags)

    print(f"\n✓ Preprocessing complete. All files saved to {processed_dir}/")
    return outputs


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Preprocess MIND dataset")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--dataset", default="mind-small")
    parser.add_argument("--max-seq-len", type=int, default=50)
    args = parser.parse_args()

    preprocess_mind(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        dataset=args.dataset,
        max_seq_len=args.max_seq_len,
    )
