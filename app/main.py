"""
Запуск FastAPI-сервера для CAMERA-модуля.
"""

from app.config.config import settings
from app.core.httpd import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.CAMERA_HOST,
        port=settings.CAMERA_PORT,
        log_level=settings.log_level,
    )
