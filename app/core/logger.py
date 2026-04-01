"""
Модуль настройки логгера приложения.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from app.config.config import settings

# Определяем путь к каталогу логов
os.makedirs(settings.CAMERA_LOGS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(settings.CAMERA_LOGS_DIR, "stt.log")

# Создаём форматтер
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Создаём ротационный хендлер (до 5 файлов по 10 МБ)
handler = RotatingFileHandler(LOG_FILE_PATH, maxBytes=10 * 1024 * 1024, backupCount=5)
handler.setFormatter(formatter)

# Настраиваем корневой логгер
logger = logging.getLogger("MBB_logger")
logger.setLevel(settings.get_log_level())
logger.addHandler(handler)

# Добавляем вывод в консоль
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Отключаем передачу логов выше (избегаем дублирования)
logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Возвращает дочерний логгер с указанным именем.

    :param name: имя модуля или компонента
    :return: экземпляр логгера
    """
    return logger.getChild(name)
