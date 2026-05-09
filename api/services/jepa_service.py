"""JEPA model loading, user encoding, candidate scoring, and metrics."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from api.config import Settings
from api.services import data_service
from models.item_encoder import ItemEncoder
from models.jepa import JEPA
from models.ranking_head import RankingHead

logger = logging.getLogger(__name__)


class JEPAService:
    def __init__(self, settings: Settings, store: data_service.DataStore):
        self.settings = settings
        self.store = store
        self.device = torch.device(settings.device)
        self.model = self._build_model().to(self.device)
        self.ranking_head = RankingHead(d_model=128, hidden_dim=256, dropout=0.1, mode="dot").to(self.device)
        self._load_checkpoint(settings.checkpoint_path)
        self._load_finetuned_if_available(settings.finetuned_checkpoint_path)
        self.model.eval()
        self.ranking_head.eval()
        self.cat_ids = torch.tensor(store.cat_ids, dtype=torch.long, device=self.device)
        self.subcat_ids = torch.tensor(store.subcat_ids, dtype=torch.long, device=self.device)
        self.entity_flags = torch.tensor(store.entity_flags, dtype=torch.float32, device=self.device)
        self.global_ctr = torch.tensor(store.global_ctr, dtype=torch.float32, device=self.device)
        self._article_embedding_cache: np.ndarray | None = None

    def _build_model(self) -> JEPA:
        item_encoder = ItemEncoder(
            text_embeddings=self.store.text_embeddings.astype(np.float32),
            num_categories=self.store.num_categories,
            num_subcategories=self.store.num_subcategories,
            text_dim=int(self.store.text_embeddings.shape[1]),
        )
        return JEPA(
            item_encoder=item_encoder,
            context_encoder_cfg={
                "d_model": 128,
                "nhead": 4,
                "num_layers": 4,
                "d_ff": 512,
                "dropout": 0.1,
                "max_seq_len": self.settings.max_seq_len,
            },
            predictor_cfg={
                "type": self.settings.predictor_type,
                "d_model": 128,
                "nhead": 4,
                "num_layers": 2,
                "d_ff": 256,
                "dropout": 0.1,
                "max_target_len": self.settings.max_seq_len,
            },
            ema_cfg={"tau_start": 0.996, "tau_end": 0.9999, "total_steps": 1},
        )

    def _load_checkpoint(self, path: Path) -> None:
        if not path.exists():
            logger.error("JEPA checkpoint not found at %s. Set JEPA_CHECKPOINT_PATH or run Stage 3/4.", path)
            raise FileNotFoundError(f"JEPA checkpoint not found: {path}")
        logger.info("Loading JEPA checkpoint from %s", path)
        checkpoint = torch.load(path, map_location="cpu")
        if "jepa_state" in checkpoint:
            self.model.load_state_dict(checkpoint["jepa_state"], strict=False)
        elif "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)
        if "ranking_head_state" in checkpoint:
            self.ranking_head.load_state_dict(checkpoint["ranking_head_state"], strict=False)

    def _load_finetuned_if_available(self, path: Path) -> None:
        if path == self.settings.checkpoint_path or not path.exists():
            return
        checkpoint = torch.load(path, map_location="cpu")
        if "jepa_state" in checkpoint:
            logger.info("Overlaying fine-tuned JEPA state from %s", path)
            self.model.load_state_dict(checkpoint["jepa_state"], strict=False)
        if "ranking_head_state" in checkpoint:
            self.ranking_head.load_state_dict(checkpoint["ranking_head_state"], strict=False)

    @torch.no_grad()
    def encode_user(self, history_article_ids: list[int]) -> np.ndarray:
        history = [int(idx) for idx in history_article_ids if int(idx) > 0]
        history = history[-self.settings.max_seq_len :]
        if not history:
            return np.zeros(128, dtype=np.float32)

        ids = torch.zeros(1, self.settings.max_seq_len, dtype=torch.long, device=self.device)
        mask = torch.zeros(1, self.settings.max_seq_len, dtype=torch.bool, device=self.device)
        seq = torch.tensor(history, dtype=torch.long, device=self.device)
        ids[0, : len(history)] = seq
        mask[0, : len(history)] = True

        z_user = self.model.get_user_representation(
            ids,
            mask,
            self.cat_ids,
            self.subcat_ids,
            self.entity_flags,
            self.global_ctr,
        )
        return z_user.squeeze(0).detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def article_embeddings(self) -> np.ndarray:
        if self._article_embedding_cache is None:
            embeddings = self.model.item_encoder.get_all_embeddings(
                self.cat_ids,
                self.subcat_ids,
                self.entity_flags,
            )
            self._article_embedding_cache = embeddings.detach().cpu().numpy().astype(np.float32)
        return self._article_embedding_cache

    def score_candidates(
        self,
        user_embedding: np.ndarray,
        candidate_ids: list[int] | None = None,
        top_k: int = 50,
    ) -> list[tuple[int, float]]:
        user = user_embedding.astype(np.float32)
        candidates = np.asarray(candidate_ids if candidate_ids is not None else list(self.store.news_by_idx.keys()), dtype=np.int64)
        candidates = candidates[(candidates > 0) & (candidates < len(self.store.text_embeddings))]
        if candidates.size == 0:
            return []

        article_embeds = self.article_embeddings()[candidates].astype(np.float32)
        user_norm = np.linalg.norm(user).astype(np.float32)
        article_norms = np.linalg.norm(article_embeds, axis=1).astype(np.float32)
        denom = np.maximum(user_norm * article_norms, np.float32(1e-8))
        scores = (article_embeds @ user) / denom
        order = np.argsort(-scores)[:top_k]
        return [(int(candidates[i]), float(scores[i])) for i in order]

    @staticmethod
    def compute_metrics(scores: list[float], labels: list[int | None]) -> dict[str, float]:
        valid = [(float(score), int(label)) for score, label in zip(scores, labels) if label is not None]
        if not valid:
            return {"auc": 0.0, "mrr": 0.0, "ndcg5": 0.0, "ndcg10": 0.0}
        score_arr = np.asarray([v[0] for v in valid], dtype=np.float32)
        label_arr = np.asarray([v[1] for v in valid], dtype=np.int32)
        if len(np.unique(label_arr)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(label_arr, score_arr))

        positive_ranks = np.flatnonzero(label_arr == 1)
        mrr = float(1.0 / (positive_ranks[0] + 1)) if positive_ranks.size else 0.0
        return {
            "auc": auc,
            "mrr": mrr,
            "ndcg5": _ndcg_at_k(label_arr, 5),
            "ndcg10": _ndcg_at_k(label_arr, 10),
        }


def _dcg(labels: np.ndarray) -> float:
    gains = (2**labels.astype(np.float32)) - 1.0
    discounts = np.log2(np.arange(len(labels), dtype=np.float32) + 2.0)
    return float(np.sum(gains / discounts))


def _ndcg_at_k(labels: np.ndarray, k: int) -> float:
    observed = labels[:k]
    ideal = np.sort(labels)[::-1][:k]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0.0:
        return 0.0
    return float(_dcg(observed) / ideal_dcg)


_service: JEPAService | None = None


def load_jepa(settings: Settings, store: data_service.DataStore) -> JEPAService:
    global _service
    if _service is None:
        _service = JEPAService(settings, store)
    return _service


def get_service() -> JEPAService:
    if _service is None:
        raise RuntimeError("JEPAService has not been loaded")
    return _service
