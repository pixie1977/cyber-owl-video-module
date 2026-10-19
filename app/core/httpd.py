#!/usr/bin/env python3
"""
HTTP-сервер на FastAPI для сервомодуля с поддержкой POST, GET.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config.config import settings
from app.core.logger import get_logger
from app.core.routers import camera_router, health_router

log = get_logger(__name__)


# Управление жизненным циклом приложения
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Функция управления жизненным циклом приложения.
    Выполняется при старте и завершении сервера.
    """
    log.info("Starting up CAMERA server...")
    yield
    log.info("Shutting down CAMERA server...")
    # Останавливаем фоновые задачи
    if hasattr(camera_router, 'data_sender'):
        camera_router.data_sender.stop()
        log.info("PeriodicFacesDataSender stopped")
    if hasattr(camera_router, 'meeting_manager'):
        camera_router.meeting_manager.stop()
        log.info("MeetingManager stopped")


app = FastAPI(title="CAMERA API Server", lifespan=lifespan)
# Подключаем роутеры
app.include_router(camera_router)
app.include_router(health_router)

# Подключаем статические файлы
print(f"CAMERA_DOC_ROOT={settings.CAMERA_DOC_ROOT}")
app.mount("/static", StaticFiles(directory=settings.CAMERA_DOC_ROOT), name="static")
