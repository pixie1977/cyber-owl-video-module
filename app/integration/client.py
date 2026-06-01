"""
Клиент для взаимодействия с STT-сервером.
Отправляет запросы на распознавание и получает результаты.
"""

import asyncio
import aiohttp
from typing import Optional


class PostClient:
    """
    Асинхронный post-клиент для работы с API.
    """

    def __init__(self, url: str):
        """
        Инициализация клиента.

        :param url: URL сервера.
        """
        self.url = url
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "PostClient":
        """
        Контекстный менеджер: открывает сессию.
        """
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Контекстный менеджер: закрывает сессию.
        """
        if self.session:
            await self.session.close()

    async def post(self, text: str) -> bool:
        """
        Отправляет текстовую строку на сервер через POST-запрос.

        :param text: текст для отправки.
        :return: True, если запрос успешен.
        """
        if not self.session:
            print("❌ Сессия не открыта. Используйте контекстный менеджер.")
            return False

        try:
            async with self.session.post(
                f"{self.url}",
                json={"text": text}
            ) as resp:
                return resp.status == 200
        except Exception as e:
            print(f"❌ Ошибка при отправке текста: {e}")
            return False

    async def post_json(self, json: dict) -> bool:
        """
        Отправляет текстовую строку на сервер через POST-запрос.

        :param json:
        :return: True, если запрос успешен.
        """
        if not self.session:
            print("❌ Сессия не открыта. Используйте контекстный менеджер.")
            return False

        try:
            async with self.session.post(
                f"{self.url}",
                json=json
            ) as resp:
                return resp.status == 200
        except Exception as e:
            print(f"❌ Ошибка при отправке текста: {e}")
            return False