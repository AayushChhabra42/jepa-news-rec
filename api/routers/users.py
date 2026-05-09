"""User list and profile routes."""

from fastapi import APIRouter, HTTPException, Query

from api.schemas import HistoryItem, UserListResponse, UserProfileResponse
from api.services import data_service

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=UserListResponse)
def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
) -> UserListResponse:
    users = sorted(data_service.get_store().user_sequences.keys())
    start = (page - 1) * page_size
    return UserListResponse(users=users[start : start + page_size], page=page, page_size=page_size, total=len(users))


@router.get("/{user_id}/profile", response_model=UserProfileResponse)
def user_profile(user_id: str) -> UserProfileResponse:
    user = data_service.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")

    history_ids = [int(idx) for idx in user.get("history_ids", [])]
    history = []
    for idx in history_ids:
        article = data_service.article_for_idx(idx)
        history.append(
            HistoryItem(
                article_id=data_service.article_id_for_idx(idx),
                title=article.get("title", ""),
                category=article.get("category", "unknown"),
                subcategory=article.get("subcategory", "unknown"),
                timestamp=article.get("timestamp"),
            )
        )
    return UserProfileResponse(user_id=user_id, history=history, features=data_service.profile_features(history_ids))
