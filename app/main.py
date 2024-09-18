from fastapi import FastAPI, Depends, Query
from fastapi.responses import RedirectResponse
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.api_v1.api import api_router
from app.core.config import settings
from app.db.init_db import init_db
from app.db.session import SessionLocal
import logging

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description='Simplify the life',
    version='1.0.0'
)

# logging.info("Creating initial data")
# db = SessionLocal()
# init_db(db)
# logging.info("Initial data created")

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        # [str(origin) for origin in settings.BACKEND_CORS_ORIGINS]
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
async def redirect_swagger():
    return RedirectResponse("/docs")

app.mount("/sounds", StaticFiles(directory="files/sounds"), name="sounds")