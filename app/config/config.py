import os
from pathlib import Path
from typing import Optional
import logging
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Определяем базовую директорию проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# Логгер для конфигурации
logger = logging.getLogger(__name__)


class Settings:
    """
    Класс настроек приложения с валидацией и дефолтными значениями.
    """

    def __init__(self):
        # Обязательные переменные
        self.CAMERA_PORT: int = self._get_int("CAMERA_PORT")
        self.CAMERA_HOST: str = self._get_str("CAMERA_HOST")
        self.CAMERA_LOG_LEVEL: str = self._get_str("CAMERA_LOG_LEVEL")

        # Необязательные переменные с дефолтами
        self.CAMERA_DOC_ROOT: Path = self._get_path(
            "CAMERA_DOC_ROOT",
            default=BASE_DIR / "content"
        )
        self.CAMERA_LOGS_DIR: Optional[Path] = self._get_path(
            "CAMERA_LOGS_DIR",
            default=None
        )
        self.CAMERA_DEVICE_INDEX: int = self._get_int(
            "CAMERA_DEVICE_INDEX",
            default=0
        )

        # Валидация уровня логирования
        self._validate_log_level()

    def _get_str(self, key: str, default: str = None) -> str:
        """Получить строковую переменную из .env"""
        value = os.getenv(key, default)
        if value is None:
            raise ValueError(
            f"Переменная окружения '{key}' не задана и нет значения по умолчанию"
        )
        return value.strip()

    def _get_int(self, key: str, default: int = None) -> int:
        """Получить целочисленную переменную из .env"""
        value = os.getenv(key)
        if value is None:
            if default is not None:
                return default
            raise ValueError(f"Переменная окружения '{key}' обязательна и не задана")
        try:
            return int(value.strip())
        except (ValueError, TypeError):
            raise ValueError(
            f"Переменная '{key}' должна быть целым числом, получено: {value}"
        )

    def _get_path(self, key: str, default: Path = None) -> Optional[Path]:
        """Получить путь из .env или вернуть Path-объект по умолчанию"""
        value = os.getenv(key)
        if value is None:
            return default
        return Path(value).expanduser().resolve()

    def _validate_log_level(self):
        """Проверяет корректность уровня логирования"""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.CAMERA_LOG_LEVEL.upper() not in valid_levels:
            raise ValueError(
                f"Неверный уровень логирования: {self.CAMERA_LOG_LEVEL}. "
                f"Допустимые значения: {', '.join(valid_levels)}"
            )

    @property
    def log_level(self) -> int:
        """Возвращает числовой уровень логирования"""
        return getattr(logging, self.CAMERA_LOG_LEVEL.upper())

    def get_log_level(self) -> str:
        """
        Возвращает строковое представление уровня логирования.

        Returns:
            str: Уровень логирования в виде строки (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        return self.CAMERA_LOG_LEVEL  # noqa: E501

    def ensure_directories(self):
        """Создаёт необходимые директории, если их нет"""
        if self.CAMERA_DOC_ROOT and not self.CAMERA_DOC_ROOT.exists():
            self.CAMERA_DOC_ROOT.mkdir(parents=True, exist_ok=True)
            logger.info(f"Создана директория: {self.CAMERA_DOC_ROOT}")

        if self.CAMERA_LOGS_DIR:
            self.CAMERA_LOGS_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Создана директория логов: {self.CAMERA_LOGS_DIR}")

        return self


# Единый экземпляр настроек
settings = Settings()
