"""
Хранение данных о лицах и встречах.
Сохраняет в JSON файл с эмбеддингами в base64.
"""

import json
import uuid
import base64
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from app.core.logger import get_logger

log = get_logger(__name__)


class FaceStorage:
    """Хранение известных и неизвестных лиц с эмбеддингами."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.known_faces: Dict[str, Dict[str, Any]] = {}
        self.unknown_faces: Dict[str, Dict[str, Any]] = {}
        self._faiss_index = None
        self._faiss_uuids = []
        self._faiss_types = []
        self._faiss_dirty = True
        self._load()

    def _load(self):
        """Загрузка данных из JSON файла."""
        if not self.storage_path.exists():
            log.info("Storage file not found, creating new storage")
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.known_faces = {k: v for k, v in data.get("known", {}).items()}
            self.unknown_faces = {k: v for k, v in data.get("unknown", {}).items()}

            log.info(
                f"Loaded storage: {len(self.known_faces)} known, "
                f"{len(self.unknown_faces)} unknown faces"
            )
        except Exception as e:
            log.error(f"Failed to load storage: {e}")
            self.known_faces = {}
            self.unknown_faces = {}

    def _save(self):
        """Сохранение данных в JSON файл."""
        try:
            data = {
                "known": self.known_faces,
                "unknown": self.unknown_faces,
                "updated_at": datetime.now().isoformat()
            }

            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            log.debug("Storage saved successfully")
        except Exception as e:
            log.error(f"Failed to save storage: {e}")

    def _embedding_to_base64(self, embedding: np.ndarray) -> str:
        """Конвертация эмбеддинга в base64 строку."""
        return base64.b64encode(embedding.tobytes()).decode("utf-8")

    def _base64_to_embedding(self, base64_str: str) -> np.ndarray:
        """Конвертация base64 строки в эмбеддинг."""
        return np.frombuffer(base64.b64decode(base64_str), dtype=np.float32)

    def add_known_face(self, name: str, embedding: np.ndarray) -> str:
        """
        Добавление известного лица в базу.
        Возвращает UUID лица.
        """
        face_uuid = str(uuid.uuid4())
        self.known_faces[face_uuid] = {
            "uuid": face_uuid,
            "name": name,
            "embedding": self._embedding_to_base64(embedding),
            "created_at": datetime.now().isoformat(),
            "last_seen": None,
            "meetings_count": 0
        }
        self._save()
        self._mark_dirty()
        log.info(f"Added known face '{name}' with UUID {face_uuid}")
        return face_uuid

    def add_unknown_face(self, embedding: np.ndarray) -> str:
        """
        Добавление неизвестного лица в базу.
        Возвращает UUID лица.
        """
        face_uuid = str(uuid.uuid4())
        self.unknown_faces[face_uuid] = {
            "uuid": face_uuid,
            "name": f"unknown_{face_uuid[:8]}",
            "embedding": self._embedding_to_base64(embedding),
            "created_at": datetime.now().isoformat(),
            "last_seen": None,
            "meetings_count": 0
        }
        self._save()
        self._mark_dirty()
        log.info(f"Added unknown face with UUID {face_uuid}")
        return face_uuid

    def update_face(self, face_uuid: str, face_type: str, embedding: np.ndarray, is_first_meeting: bool = False):
        """
        Обновление информации о встрече с лицом.
        face_type: 'known' или 'unknown'
        """
        faces_dict = self.known_faces if face_type == "known" else self.unknown_faces

        if face_uuid not in faces_dict:
            log.warning(f"Face {face_uuid} not found in {face_type} faces")
            return

        face_data = faces_dict[face_uuid]

        if is_first_meeting:
            face_data["created_at"] = datetime.now().isoformat()

        face_data["last_seen"] = datetime.now().isoformat()
        face_data["meetings_count"] = face_data.get("meetings_count", 0) + 1

        # Обновляем эмбеддинг (среднее между старым и новым)
        old_embedding = self._base64_to_embedding(face_data["embedding"])
        face_data["embedding"] = self._embedding_to_base64(
            (old_embedding + embedding) / 2.0
        )

        self._save()
        self._mark_dirty()
        log.debug(f"Updated face {face_uuid} ({face_type}), meetings: {face_data['meetings_count']}")

    def get_face(self, face_uuid: str, face_type: str) -> Optional[Dict[str, Any]]:
        """Получение информации о лице."""
        faces_dict = self.known_faces if face_type == "known" else self.unknown_faces
        return faces_dict.get(face_uuid)

    def get_all_faces(self) -> tuple[Dict[str, Dict], Dict[str, Dict]]:
        """Возвращает все лица."""
        return self.known_faces, self.unknown_faces

    def _build_faiss_index(self):
        """Строит/обновляет FAISS индекс если есть изменения."""
        if not self._faiss_dirty:
            return

        import faiss

        self._faiss_uuids.clear()
        self._faiss_types.clear()
        all_embeddings = []

        for uuid_, data in self.known_faces.items():
            emb = self._base64_to_embedding(data["embedding"])
            all_embeddings.append(emb)
            self._faiss_uuids.append(uuid_)
            self._faiss_types.append("known")

        for uuid_, data in self.unknown_faces.items():
            emb = self._base64_to_embedding(data["embedding"])
            all_embeddings.append(emb)
            self._faiss_uuids.append(uuid_)
            self._faiss_types.append("unknown")

        if not all_embeddings:
            self._faiss_index = None
            self._faiss_dirty = False
            return

        dim = all_embeddings[0].shape[0]
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(np.array(all_embeddings).astype('float32'))
        self._faiss_dirty = False
        log.debug(f"FAISS index rebuilt: {len(all_embeddings)} vectors")

    def find_similar_face(self, embedding: np.ndarray, tolerance: float = 0.55) -> Optional[tuple[str, str]]:
        """
        Поиск похожего лица в базе с кэшированным FAISS индексом.
        Возвращает (uuid, face_type) если найдено, иначе None.
        """
        self._build_faiss_index()

        if self._faiss_index is None:
            return None

        query_emb = embedding.reshape(1, -1).astype('float32')
        distances, indices = self._faiss_index.search(query_emb, k=1)

        score = distances[0][0]
        idx = indices[0][0]

        if score > tolerance:
            return (self._faiss_uuids[idx], self._faiss_types[idx])

        return None

    def _mark_dirty(self):
        """Помечает индекс как устаревший."""
        self._faiss_dirty = True
