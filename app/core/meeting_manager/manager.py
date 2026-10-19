"""
Менеджер встреч с людьми.
Обрабатывает обнаружение лиц, определяет тип встречи и отправляет уведомления в LLM.
"""

import asyncio
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

from app.core.logger import get_logger
from app.core.meeting_manager.storage import FaceStorage
from app.integration.client import PostClient

log = get_logger(__name__)


class MeetingManager:
    """
    Управление встречами с людьми.
    
    Логика:
    1. Если в кадре новый человек - проверяем базу, сохраняем с UUID
    2. Если есть в базе как именованный персонаж - проверяем первую ли встречу сегодня
    3. Если встреча первая - отправляем уведомление в LLM
    4. Если не первая - отправляем время с последней встречи
    5-7. Обработка незнакомцев по аналогии
    8. Приоритет: именованные лица > неизвестные
    9. Несколько незнакомцев - отдельное уведомление
    """

    def __init__(
        self,
        storage_path: Path,
        llm_endpoint_url: str,
        detection_threshold: float = 0.55,
        meeting_check_interval: int = 300  # 5 минут
    ):
        self.storage = FaceStorage(storage_path)
        self.llm_url = llm_endpoint_url
        self.detection_threshold = detection_threshold
        self.meeting_check_interval = meeting_check_interval

        # Отслеживание встреч за сегодня
        self._today_meetings: Dict[str, List[datetime]] = {}
        self._meetings_date: Optional[str] = None
        self._meetings_path = storage_path.parent / "meetings_history.json"
        self._load_today_meetings()

        # Фоновый поток для отправки уведомлений
        self._is_running = False
        self._notification_queue: List[Dict[str, Any]] = []
        self._thread: Optional[threading.Thread] = None

    def _load_today_meetings(self):
        """Загрузка истории встреч за текущий день."""
        today = datetime.now().date().isoformat()

        # Если дата сменилась — сбрасываем
        if self._meetings_date == today and self._today_meetings:
            return  # Те же данные, ничего не делаем

        # Загружаем из файла если есть
        if self._meetings_path.exists():
            try:
                with open(self._meetings_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                stored_date = data.get("date")
                if stored_date == today:
                    self._today_meetings = {
                        k: [datetime.fromisoformat(dt) for dt in v]
                        for k, v in data.get("meetings", {}).items()
                    }
                    self._meetings_date = stored_date
                    log.info(f"Loaded meetings from file: {len(self._today_meetings)} faces today")
                    return
            except Exception as e:
                log.warning(f"Failed to load meetings from file: {e}")

        # Новый день — сбрасываем
        self._today_meetings = {}
        self._meetings_date = today

    def _save_today_meetings(self):
        """Сохранение истории встреч в файл."""
        try:
            data = {
                "date": self._meetings_date,
                "meetings": {
                    k: [dt.isoformat() for dt in v]
                    for k, v in self._today_meetings.items()
                }
            }
            self._meetings_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._meetings_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            log.error(f"Failed to save meetings: {e}")

    def _calculate_time_since_last_meeting(self, face_uuid: str) -> Optional[str]:
        """Вычисление времени с последней встречи."""
        face_data = self.storage.get_face(face_uuid, "known") or self.storage.get_face(face_uuid, "unknown")

        if not face_data or not face_data.get("last_seen"):
            return None

        last_seen = datetime.fromisoformat(face_data["last_seen"])
        now = datetime.now()
        diff = now - last_seen

        hours = int(diff.total_seconds() // 3600)
        minutes = int((diff.total_seconds() % 3600) // 60)

        if hours > 0:
            return f"{hours} ч. {minutes} мин."
        else:
            return f"{minutes} мин."

    def _is_first_meeting_today(self, face_uuid: str) -> bool:
        """Проверка, первая ли встреча сегодня."""
        return face_uuid not in self._today_meetings

    def _record_meeting(self, face_uuid: str):
        """Запись встречи в историю."""
        if face_uuid not in self._today_meetings:
            self._today_meetings[face_uuid] = []

        self._today_meetings[face_uuid].append(datetime.now())
        self._save_today_meetings()

    def _format_notification(self, face_uuid: str, face_type: str, is_first_meeting: bool,
                             face_name: str = None) -> str:
        """Формирование текста уведомления."""
        time_since = self._calculate_time_since_last_meeting(face_uuid)

        if face_type == "known":
            name = face_name or "неизвестно"

            if is_first_meeting:
                return f"Первая встреча сегодня с известным: {name}"
            else:
                return f"Повторная встреча с известным: {name} (прошло: {time_since})"
        else:
            if is_first_meeting:
                return f"Первая встреча с незнакомцем (UUID: {face_uuid[:8]})"
            else:
                return f"Повторная встреча с незнакомцем (UUID: {face_uuid[:8]}, прошло: {time_since})"

    def _queue_notification(self, notification: str, face_uuid: str, face_type: str,
                            is_first_meeting: bool, face_name: str = None):
        """Добавление уведомления в очередь."""
        self._notification_queue.append({
            "text": notification,
            "face_uuid": face_uuid,
            "face_type": face_type,
            "is_first_meeting": is_first_meeting,
            "face_name": face_name,
            "timestamp": datetime.now().isoformat()
        })

        # Если очередь пуста - запускаем отправку
        if len(self._notification_queue) == 1:
            self._send_next_notification()

    def _send_next_notification(self):
        """Отправка следующего уведомления из очереди."""
        if not self._notification_queue:
            return

        notification = self._notification_queue[0]

        async def _send():
            try:
                async with PostClient(self.llm_url) as client:
                    success = await client.post(text=notification["text"])
                    if success:
                        log.info(f"Notification sent: {notification['text']}")
                        self._notification_queue.pop(0)
                        # Отправляем следующее
                        self._send_next_notification()
                    else:
                        log.error(f"Failed to send notification: {notification['text']}")
            except Exception as e:
                log.error(f"Error sending notification: {e}")
                # Пробуем ещё раз через 5 секунд
                import asyncio
                asyncio.get_event_loop().call_later(5, self._send_next_notification)

        # Используем текущий event loop или создаём новый
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Если loop уже работает — запускаем в task
                asyncio.ensure_future(_send())
            else:
                loop.run_until_complete(_send())
        except RuntimeError:
            # Нет event loop — создаём новый
            asyncio.run(_send())

    def process_faces(self, detected_faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Обработка обнаруженных лиц.
        
        :param detected_faces: Список обнаруженных лиц с полями:
            - embedding: np.ndarray
            - bbox: List[int]
            - confidence: float
            - age: int (опционально)
            - sex: str (опционально)
        :return: Обновлённый список с дополнительной информацией:
            - face_uuid: str или None
            - face_type: 'known' | 'unknown' | None
            - face_name: str или None
            - is_first_meeting: bool
            - meeting_message: str
            - age: int
            - sex: str
        """
        if not detected_faces:
            return []

        result = []
        known_faces_in_frame = []
        unknown_faces_in_frame = []

        for face in detected_faces:
            embedding = face["embedding"]
            bbox = face.get("bbox", [])
            confidence = face.get("confidence", 0.0)
            age = face.get("age")
            sex = face.get("sex")

            # Ищем похожее лицо в базе
            match = self.storage.find_similar_face(embedding, self.detection_threshold)

            if match:
                face_uuid, face_type = match
                is_first_meeting = self._is_first_meeting_today(face_uuid)

                # Обновляем лицо в хранилище
                self.storage.update_face(face_uuid, face_type, embedding, is_first_meeting)

                # Записываем встречу
                self._record_meeting(face_uuid)

                # Получаем имя
                face_data = self.storage.get_face(face_uuid, face_type)
                face_name = face_data.get("name") if face_data else None

                log.info(
                    f"Match found: uuid={face_uuid[:8]}, type={face_type}, "
                    f"name={face_name}, first_meeting={is_first_meeting}"
                )

                # Формируем уведомление
                notification = self._format_notification(
                    face_uuid, face_type, is_first_meeting, face_name
                )
                self._queue_notification(notification, face_uuid, face_type, is_first_meeting, face_name)

                result.append({
                    "embedding": embedding,
                    "bbox": bbox,
                    "confidence": confidence,
                    "face_uuid": face_uuid,
                    "face_type": face_type,
                    "face_name": face_name,
                    "is_first_meeting": is_first_meeting,
                    "meeting_message": notification,
                    "age": age,
                    "sex": sex
                })

                if face_type == "known":
                    known_faces_in_frame.append(face)
                else:
                    unknown_faces_in_frame.append(face)
            else:
                # Новое лицо - сохраняем
                face_uuid = self.storage.add_unknown_face(embedding)
                is_first_meeting = True

                log.info(
                    f"New unknown face: uuid={face_uuid[:8]}, "
                    f"saved to storage"
                )

                # Записываем встречу
                self._record_meeting(face_uuid)

                notification = self._format_notification(
                    face_uuid, "unknown", is_first_meeting
                )
                self._queue_notification(notification, face_uuid, "unknown", is_first_meeting)

                result.append({
                    "embedding": embedding,
                    "bbox": bbox,
                    "confidence": confidence,
                    "face_uuid": face_uuid,
                    "face_type": "unknown",
                    "face_name": None,
                    "is_first_meeting": is_first_meeting,
                    "meeting_message": notification,
                    "age": age,
                    "sex": sex
                })
                unknown_faces_in_frame.append(face)

        # Приоритизация: если есть известные лица - оставляем только их
        if known_faces_in_frame:
            result = [r for r in result if r.get("face_type") == "known"]

        # Несколько незнакомцев - отдельное уведомление
        if len(unknown_faces_in_frame) > 1:
            unknown_uuids = [r["face_uuid"] for r in result if r.get("face_type") == "unknown"]
            all_first = all(r["is_first_meeting"] for r in result if r.get("face_type") == "unknown")

            if all_first:
                message = f"Обнаружено {len(unknown_faces_in_frame)} незнакомцев (первые встречи)"
            else:
                last_seen_times = []
                for uuid_ in unknown_uuids:
                    time_since = self._calculate_time_since_last_meeting(uuid_)
                    last_seen_times.append(time_since if time_since else "неизвестно")
                message = f"Обнаружено {len(unknown_faces_in_frame)} незнакомцев (последние встречи: {', '.join(last_seen_times)})"

            self._queue_notification(message, "multiple_unknowns", "multiple", all_first)

        return result

    def start(self):
        """Запуск менеджера встреч."""
        self._is_running = True
        log.info("MeetingManager started")

    def stop(self):
        """Остановка менеджера встреч."""
        self._is_running = False
        log.info("MeetingManager stopped")
