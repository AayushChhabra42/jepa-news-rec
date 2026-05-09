"""Pydantic response models for the recommendation API."""

from typing import Literal

from pydantic import BaseModel, Field


class UserListResponse(BaseModel):
    users: list[str]
    page: int
    page_size: int
    total: int


class HistoryItem(BaseModel):
    article_id: str
    title: str
    category: str
    subcategory: str
    timestamp: str | None = None


class CategoryFeature(BaseModel):
    category: str
    count: int
    pct: float


class SubcategoryFeature(BaseModel):
    subcategory: str
    count: int
    pct: float


class UserFeatures(BaseModel):
    top_categories: list[CategoryFeature]
    top_subcategories: list[SubcategoryFeature]
    avg_session_length: float
    recency_bias_score: float


class UserProfileResponse(BaseModel):
    user_id: str
    history: list[HistoryItem]
    features: UserFeatures


class RecommendationItem(BaseModel):
    rank: int
    article_id: str
    title: str
    category: str
    subcategory: str
    abstract: str
    jepa_score: float
    xgb_score: float | None = None
    jepa_rank: int | None = None
    xgb_rank: int | None = None
    final_rank: int | None = None
    rank_delta: int | None = None
    label: int | None = None


class Metrics(BaseModel):
    auc: float
    mrr: float
    ndcg5: float = Field(alias="ndcg5")
    ndcg10: float = Field(alias="ndcg10")


class RecommendationsResponse(BaseModel):
    stage: Literal["jepa", "xgb", "both"]
    user_id: str
    recommendations: list[RecommendationItem]
    metrics: Metrics


class HealthResponse(BaseModel):
    status: Literal["ok"]
    stage: Literal["jepa"]
