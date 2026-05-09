"""Recommendation route."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from api.config import get_settings
from api.schemas import RecommendationItem, RecommendationsResponse
from api.services import data_service, jepa_service, xgb_service

router = APIRouter(prefix="/users", tags=["recommendations"])


@router.get("/{user_id}/recommendations", response_model=RecommendationsResponse)
def recommendations(
    user_id: str,
    top_k: int = Query(default=None, ge=1, le=500),
    stage: Literal["jepa", "xgb", "both"] = "jepa",
    label_reveal: bool = False,
) -> RecommendationsResponse:
    settings = get_settings()
    top_k = top_k or settings.top_k_default
    user = data_service.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")

    service = jepa_service.get_service()
    labels_by_idx = data_service.labels_for_user(user_id)
    labeled_candidates = data_service.labeled_candidate_ids_for_user(user_id)
    scored = service.score_candidates(
        service.encode_user(user.get("history_ids", [])),
        candidate_ids=labeled_candidates or None,
        top_k=top_k,
    )

    jepa_candidates = [
        {
            "article_idx": idx,
            "article_id": data_service.article_id_for_idx(idx),
            "jepa_rank": rank,
            "jepa_score": score,
        }
        for rank, (idx, score) in enumerate(scored, start=1)
    ]
    if stage in {"xgb", "both"}:
        try:
            ranked_candidates = xgb_service.get_service(settings, data_service.get_store()).rerank(user_id, jepa_candidates)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    else:
        ranked_candidates = [
            {**candidate, "xgb_score": None, "xgb_rank": None, "rank_delta": None}
            for candidate in jepa_candidates
        ]

    items: list[RecommendationItem] = []
    metric_labels: list[int | None] = []
    metric_scores: list[float] = []
    for final_rank, candidate in enumerate(ranked_candidates, start=1):
        idx = int(candidate["article_idx"])
        article = data_service.article_for_idx(idx)
        label = labels_by_idx.get(idx)
        score = candidate["xgb_score"] if stage in {"xgb", "both"} else candidate["jepa_score"]
        metric_scores.append(float(score or 0.0))
        metric_labels.append(label)
        displayed_rank = int(candidate["xgb_rank"] if stage == "xgb" else final_rank)
        jepa_rank = int(candidate["jepa_rank"])
        xgb_rank = candidate["xgb_rank"]
        final_rank_value = displayed_rank if stage == "xgb" else final_rank
        items.append(
            RecommendationItem(
                rank=displayed_rank,
                article_id=str(candidate["article_id"]),
                title=article.get("title", ""),
                category=article.get("category", "unknown"),
                subcategory=article.get("subcategory", "unknown"),
                abstract=article.get("abstract", ""),
                jepa_score=float(candidate["jepa_score"]),
                xgb_score=candidate["xgb_score"],
                jepa_rank=jepa_rank,
                xgb_rank=xgb_rank,
                final_rank=final_rank_value,
                rank_delta=candidate["rank_delta"],
                label=label if label_reveal else None,
            )
        )

    return RecommendationsResponse(
        stage=stage,
        user_id=user_id,
        recommendations=items,
        metrics=service.compute_metrics(metric_scores, metric_labels),
    )
