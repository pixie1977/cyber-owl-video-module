import asyncio
import threading
from typing import Callable

from app.core.logger import get_logger
from app.integration.client import PostClient

# --- Настройка логирования ---
log = get_logger(__name__)


class PeriodicFacesDataSender:
    """Класс для периодической отправки данных в фоновом потоке."""

    def __init__(
        self,
        data_provider: Callable,
        url: str,
        interval_seconds: int = 3
    ):
        """
        Инициализация воркера.

        :param data_provider: Функция, возвращающая данные для отправки.
        :param url: Адрес доставки.
        :param interval_seconds: Интервал между отправками в секундах.
        """
        self.data_provider = data_provider
        self.url = url
        self.interval = interval_seconds

        self.is_running = False
        self._thread = None

    async def _loop(self):
        """Внутренний цикл, выполняющийся в отдельном потоке."""
        while self.is_running:
            try:
                # Получаем данные от провайдера
                faces, _ = self.data_provider()

                faces_data = [face.to_dict() for face in faces]

                log.info(f"Отправляем данные: {faces_data}")

                status = None
                # Отправляем данные
                async with PostClient(self.url) as client:
                    status = await client.post_json(json=faces_data)

                log.info(f"Результат отправки данных: {status}")

                # Пауза с возможностью прерывания
                for _ in range(self.interval):
                    if not self.is_running:
                        break
                    await asyncio.sleep(0.1)

            except Exception as e:
                log.error(f"Ошибка при отправке данных: {e}")
                await asyncio.sleep(5)

    def _start_async_loop(self):
        # Эта функция запускается внутри нового потока
        # asyncio.run автоматически создает НОВЫЙ event loop для этого потока
        asyncio.run(self._loop())

    def start(self):
        """Запуск периодической отправки данных."""
        if not self.is_running:
            self.is_running = True
            self._thread = threading.Thread(target=self._start_async_loop, daemon=True)
            self._thread.start()

    def stop(self):
        """Остановка отправки данных и ожидание завершения потока."""
        if self.is_running:
            self.is_running = False
            if self._thread:
                self._thread.join()



