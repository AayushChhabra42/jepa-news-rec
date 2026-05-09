"""Stage 2 XGBoost reranker placeholder."""


def load_xgb_model() -> None:
    raise NotImplementedError(
        "Stage 2: load checkpoints/xgb_ranker.pkl, build LambdaRank features for JEPA "
        "top-K candidates, and expose rerank(jepa_candidates) with xgb_score and rank deltas."
    )


def rerank(*_args, **_kwargs) -> None:
    raise NotImplementedError(
        "Stage 2 reranking is intentionally not implemented in the JEPA-only demo."
    )
