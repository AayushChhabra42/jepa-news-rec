"""FastAPI entrypoint for the JEPA-only recommendation demo."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_settings
from api.routers import recommendations, users
from api.schemas import HealthResponse
from api.services import data_service, jepa_service, xgb_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    store = data_service.load_data(settings)
    jepa_service.load_jepa(settings, store)
    if settings.enable_xgb:
        xgb_service.load_xgb(settings, store)
        logger.info("Startup complete: JEPA and XGBoost artifacts loaded")
    else:
        logger.info("Startup complete: JEPA artifacts loaded")
    yield


app = FastAPI(title="JEPA News Recommendation API", version="0.1.0", lifespan=lifespan)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/", StaticFiles(directory="static", html=True), name="static")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", stage="jepa")


app.include_router(users.router, prefix="/api")
app.include_router(recommendations.router, prefix="/api")
