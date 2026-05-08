"""
Запуск FastAPI-сервера для CAMERA-модуля.
"""

import sys
import io
import uvicorn

from app.config.config import settings
from app.core.httpd import app


# Настройка UTF-8 для stdout/stderr (актуально для Windows)
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


if __name__ == "__main__":
    """Запуск Uvicorn-сервера."""
    uvicorn.run(
        app,
        host=settings.CAMERA_HOST,
        port=settings.CAMERA_PORT,
        log_level=settings.log_level,
    )