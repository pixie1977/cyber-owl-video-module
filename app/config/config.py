
from typing import Optional, Literal
import os
import logging
from pathlib import Path
from distutils.util import strtobool

from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Определяем базовую директорию проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# Логгер для конфигурации
logger = logging.getLogger(__name__)

# Тип для уровня логирования
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings:
    """
    Класс настроек приложения с валидацией, дефолтами и созданием директорий.
    """

    def __init__(self):
        # Обязательные переменные
        self.CAMERA_HOST: str = self._get_str("CAMERA_HOST")
        self.CAMERA_PORT: int = self._get_int("CAMERA_PORT")
        self.CAMERA_LOG_LEVEL: LogLevel = self._validate_log_level(
            self._get_str("CAMERA_LOG_LEVEL", "INFO").upper()
        )

        # Необязательные с дефолтами
        self.CAMERA_DOC_ROOT: Path = self._get_path(
            "CAMERA_DOC_ROOT", default=BASE_DIR / "content"
        )
        self.CAMERA_LOGS_DIR: Optional[Path] = self._get_path(
            "CAMERA_LOGS_DIR", default=BASE_DIR / "logs"
        )
        self.CAMERA_DEVICE_INDEX: int = self._get_int("CAMERA_DEVICE_INDEX", 0)
        self.CAMERA_FRAME_WIDTH: int = self._get_int("CAMERA_FRAME_WIDTH", 640)
        self.CAMERA_FRAME_HEIGHT: int = self._get_int("CAMERA_FRAME_HEIGHT", 480)
        self.CAMERA_FPS: int = self._get_int("CAMERA_FPS", 30)

        self.IS_JETSON: bool = self._get_bool("IS_JETSON", False)
        self.DEBUG: bool = self._get_bool("DEBUG", False)

        self.FACE_DETECTION_MODEL_NAME: str = self._get_str('FACE_DETECION_MODE_NAME', 'buffalo_l')

        # Таймауты
        self.CAMERA_STREAM_TIMEOUT: float = self._get_float("CAMERA_STREAM_TIMEOUT", 10.0)
        self.CAMERA_CAPTURE_TIMEOUT: float = self._get_float("CAMERA_CAPTURE_TIMEOUT", 5.0)

        # Формат логов
        self.LOG_FORMAT: str = self._get_str(
            "LOG_FORMAT",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        self.EMBEDDINGS_DIR: Optional[Path] = self._get_path(
            "EMBEDDINGS_DIR", default=BASE_DIR / "embeddings"
        )

        self.CAMERA_DETECTION_THRESHOLD: float = self._get_float("CAMERA_DETECTION_THRESHOLD", 0.5)

        # Валидация после инициализации
        self._post_init_validate()

    def _get_str(self, key: str, default: str = None) -> str:
        value = os.getenv(key)
        if value is None:
            if default is not None:
                return default
            raise ValueError(f"Переменная окружения '{key}' обязательна и не задана")
        return value.strip()

    def _get_int(self, key: str, default: int = None) -> int:
        value = os.getenv(key)
        if value is None:
            if default is not None:
                return default
            raise ValueError(f"Переменная окружения '{key}' обязательна и не задана")
        try:
            return int(value.strip())
        except (ValueError, TypeError):
            raise ValueError(f"Переменная '{key}' должна быть целым числом, получено: {value}")

    def _get_float(self, key: str, default: float = None) -> float:
        value = os.getenv(key)
        if value is None:
            if default is not None:
                return default
            raise ValueError(f"Переменная окружения '{key}' обязательна и не задана")
        try:
            return float(value.strip())
        except (ValueError, TypeError):
            raise ValueError(f"Переменная '{key}' должна быть числом с плавающей точкой, получено: {value}")

    def _get_bool(self, key: str, default: bool = False) -> bool:
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return bool(strtobool(value.strip().lower()))
        except ValueError:
            raise ValueError(f"Переменная '{key}' должна быть 'true' или 'false', получено: {value}")

    def _get_path(self, key: str, default: Path = None) -> Path:
        value = os.getenv(key)
        if value is None:
            if default is None:
                raise ValueError(f"Путь '{key}' не задан и нет значения по умолчанию")
            path = Path(default)
        else:
            path = Path(value).expanduser().resolve()

        return path

    def _validate_log_level(self, level: str) -> LogLevel:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in valid_levels:
            raise ValueError(
                f"Неверный уровень логирования: {level}. "
                f"Допустимые значения: {', '.join(sorted(valid_levels))}"
            )
        return level  # type: ignore

    def _post_init_validate(self):
        """Дополнительная валидация после инициализации."""
        if self.CAMERA_PORT < 1 or self.CAMERA_PORT > 65535:
            raise ValueError(f"Неверный порт: {self.CAMERA_PORT}. Должен быть в диапазоне 1–65535")

        if self.CAMERA_DOC_ROOT.exists() and not self.CAMERA_DOC_ROOT.is_dir():
            raise ValueError(f"CAMERA_DOC_ROOT существует, но это не директория: {self.CAMERA_DOC_ROOT}")

        if self.CAMERA_LOGS_DIR and self.CAMERA_LOGS_DIR.exists() and not self.CAMERA_LOGS_DIR.is_dir():
            raise ValueError(f"CAMERA_LOGS_DIR существует, но это не директория: {self.CAMERA_LOGS_DIR}")

    def ensure_directories(self) -> "Settings":
        """Создаёт необходимые директории."""
        dirs_to_create = [
            ("CAMERA_DOC_ROOT", self.CAMERA_DOC_ROOT),
            ("CAMERA_LOGS_DIR", self.CAMERA_LOGS_DIR),
        ]

        for name, path in dirs_to_create:
            if path is None:
                continue
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Создана директория: {path}")
            elif not path.is_dir():
                raise RuntimeError(f"Путь {path} должен быть директорией, но это файл или ссылка.")

        return self

    @property
    def log_level(self) -> int:
        """Возвращает числовой уровень логирования."""
        return getattr(logging, self.CAMERA_LOG_LEVEL)

    def __repr__(self) -> str:
        return (f"<Settings "
                f"host={self.CAMERA_HOST}, "
                f"port={self.CAMERA_PORT}, "
                f"log_level={self.CAMERA_LOG_LEVEL}, "
                f"doc_root={self.CAMERA_DOC_ROOT}, "
                f"device_index={self.CAMERA_DEVICE_INDEX}, "
                f"is_jetson={self.IS_JETSON}>")

    def dict(self) -> dict:
        """Возвращает словарь с основными настройками (для дебага)."""
        return {
            "CAMERA_HOST": self.CAMERA_HOST,
            "CAMERA_PORT": self.CAMERA_PORT,
            "CAMERA_LOG_LEVEL": self.CAMERA_LOG_LEVEL,
            "CAMERA_DOC_ROOT": str(self.CAMERA_DOC_ROOT),
            "CAMERA_LOGS_DIR": str(self.CAMERA_LOGS_DIR) if self.CAMERA_LOGS_DIR else None,
            "CAMERA_DEVICE_INDEX": self.CAMERA_DEVICE_INDEX,
            "CAMERA_FRAME_WIDTH": self.CAMERA_FRAME_WIDTH,
            "CAMERA_FRAME_HEIGHT": self.CAMERA_FRAME_HEIGHT,
            "CAMERA_FPS": self.CAMERA_FPS,
            "IS_JETSON": self.IS_JETSON,
            "DEBUG": self.DEBUG,
            "CAMERA_STREAM_TIMEOUT": self.CAMERA_STREAM_TIMEOUT,
            "CAMERA_CAPTURE_TIMEOUT": self.CAMERA_CAPTURE_TIMEOUT,
        }


# Единый экземпляр настроек
settings = Settings()