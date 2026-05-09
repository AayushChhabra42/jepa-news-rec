"""Recommendation route."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from api.config import get_settings
from api.schemas import RecommendationItem, RecommendationsResponse
from api.services import data_service, jepa_service

router = APIRouter(prefix="/users", tags=["recommendations"])


@router.get("/{user_id}/recommendations", response_model=RecommendationsResponse)
def recommendations(
    user_id: str,
    top_k: int = Query(default=None, ge=1, le=500),
    stage: Literal["jepa"] = "jepa",
    label_reveal: bool = False,
) -> RecommendationsResponse:
    settings = get_settings()
    top_k = top_k or settings.top_k_default
    user = data_service.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")

    service = jepa_service.get_service()
    labels_by_idx = data_service.labels_for_user(user_id)
    scored = service.score_candidates(service.encode_user(user.get("history_ids", [])), top_k=top_k)

    items: list[RecommendationItem] = []
    metric_labels: list[int | None] = []
    metric_scores: list[float] = []
    for rank, (idx, score) in enumerate(scored, start=1):
        article = data_service.article_for_idx(idx)
        label = labels_by_idx.get(idx)
        metric_scores.append(score)
        metric_labels.append(label)
        items.append(
            RecommendationItem(
                rank=rank,
                article_id=data_service.article_id_for_idx(idx),
                title=article.get("title", ""),
                category=article.get("category", "unknown"),
                subcategory=article.get("subcategory", "unknown"),
                abstract=article.get("abstract", ""),
                jepa_score=score,
                xgb_score=None,
                label=label if label_reveal else None,
            )
        )

    return RecommendationsResponse(
        stage=stage,
        user_id=user_id,
        recommendations=items,
        metrics=service.compute_metrics(metric_scores, metric_labels),
    )
