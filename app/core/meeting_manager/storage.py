"""
Хранение данных о лицах и встречах.
Сохраняет в JSON файл с эмбеддингами в base64.
"""

import gc
import glob
import json
import os
import uuid
import base64
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from app.core.logger import get_logger

log = get_logger(__name__)


class FaceStorage:
    """Хранение известных и неизвестных лиц с эмбеддингами."""

    def __init__(self, storage_path: Path, embeddings_dir: Optional[Path] = None):
        self.storage_path = storage_path
        self.embeddings_dir = embeddings_dir
        self.known_faces: Dict[str, Dict[str, Any]] = {}
        self.unknown_faces: Dict[str, Dict[str, Any]] = {}
        self._faiss_index = None
        self._faiss_uuids = []
        self._faiss_types = []
        self._faiss_dirty = True
        self._load()
        
        # Синхронизируем known-лица из .npy файлов
        if self.embeddings_dir:
            try:
                added = self.sync_known_faces_from_npy(str(self.embeddings_dir))
                if added:
                    log.info(f"Synced {len(added)} known faces from npy: {added}")
                else:
                    log.info("No new known faces to sync from npy")
            except Exception as e:
                log.error(f"Failed to sync known faces from npy: {e}")

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
        # Освобождаем память от временных numpy массивов
        del all_embeddings
        gc.collect()

    def find_similar_face(self, embedding: np.ndarray, tolerance: float = 0.75) -> Optional[tuple[str, str]]:
        """
        Поиск похожего лица в базе с кэшированным FAISS индексом.
        Приоритизирует known-лица над unknown.
        
        Алгоритм:
        1. Сначала ищем known-лица с мягким порогом (0.65)
        2. Если не нашли — ищем среди всех с основным порогом (0.75)
        3. Приоритет: known-лица всегда выше unknown
        
        Возвращает (uuid, face_type) если найдено, иначе None.
        """
        self._build_faiss_index()

        if self._faiss_index is None:
            return None

        query_emb = embedding.reshape(1, -1).astype('float32')
        # Ищем top-10 для полного покрытия
        distances, indices = self._faiss_index.search(query_emb, k=min(10, len(self._faiss_uuids)))

        # Фаза 1: ищем known-лица с мягким порогом
        best_known = None
        known_threshold = 0.65
        
        for i in range(len(indices[0])):
            score = distances[0][i]
            idx = indices[0][i]
            
            if self._faiss_types[idx] == "known" and score > known_threshold:
                if best_known is None or score > best_known[0]:
                    best_known = (score, self._faiss_uuids[idx])
        
        if best_known:
            log.debug(f"Match found (known): score={best_known[0]:.3f}, uuid={best_known[1][:8]}")
            return (best_known[1], "known")

        # Фаза 2: ищем среди всех с основным порогом
        best_known = None
        best_unknown = None

        for i in range(len(indices[0])):
            score = distances[0][i]
            idx = indices[0][i]
            
            if score <= tolerance:
                continue
            
            face_type = self._faiss_types[idx]
            face_uuid = self._faiss_uuids[idx]
            
            if face_type == "known" and (best_known is None or score > best_known[0]):
                best_known = (score, face_uuid)
            elif face_type == "unknown" and (best_unknown is None or score > best_unknown[0]):
                best_unknown = (score, face_uuid)
        
        # Приоритет: known-лица всегда выше unknown
        if best_known:
            log.debug(f"Match found (known): score={best_known[0]:.3f}, uuid={best_known[1][:8]}")
            return (best_known[1], "known")
        
        if best_unknown:
            log.debug(f"Match found (unknown): score={best_unknown[0]:.3f}, uuid={best_unknown[1][:8]}")
            return (best_unknown[1], "unknown")

        log.debug(f"No match: best known={best_known}, best unknown={best_unknown}")
        return None

    def _mark_dirty(self):
        """Помечает индекс как устаревший."""
        self._faiss_dirty = True

    def sync_known_faces_from_npy(self, embeddings_dir: str) -> List[str]:
        """
        Синхронизация known-лиц из .npy файлов в FaceStorage.
        Группирует файлы вида {name}-{idx}.npy по имени, усредняет эмбеддинги.
        
        Например: "мастер-1.npy", "мастер-2.npy" → known-лицо "мастер".
        
        Возвращает список добавленных/обновлённых имён.
        """
        added_names = []
        
        # Собираем эмбеддинги из .npy файлов
        npy_files = sorted(glob.glob(os.path.join(embeddings_dir, "*.npy")))
        
        # Группируем по имени: {"мастер": [emb1, emb2, ...]}
        name_embeddings: Dict[str, List[np.ndarray]] = {}
        
        for filepath in npy_files:
            filename = os.path.basename(filepath)
            # Парсим имя: {name}-{idx}.npy
            name = filename.replace(".npy", "")
            
            # Проверяем формат name-idx.npy
            parts = name.rsplit("-", 1)
            if len(parts) != 2:
                continue
            
            username, idx_str = parts
            
            # Проверяем, что idx — число
            if not idx_str.isdigit():
                continue
            
            # Группируем по имени без idx: "мастер-1" -> "мастер"
            if username not in name_embeddings:
                name_embeddings[username] = []
            name_embeddings[username].append(np.load(filepath))
        
        # Для каждой группы усредняем эмбеддинги и добавляем/обновляем known-лицо
        for username, embeddings in name_embeddings.items():
            # Усредняем все эмбеддинги для этого лица
            avg_embedding = np.mean(embeddings, axis=0)
            
            # Проверяем, есть ли уже это лицо в known_faces
            known_by_name = {
                uuid_: data for uuid_, data in self.known_faces.items()
                if data.get("name") == username
            }
            
            if known_by_name:
                # Лицо уже есть — обновляем эмбеддинг
                existing_uuid = next(iter(known_by_name))
                self.known_faces[existing_uuid]["embedding"] = self._embedding_to_base64(avg_embedding)
                log.debug(f"Updated known face '{username}' (UUID {existing_uuid[:8]})")
            else:
                # Добавляем новое known-лицо
                face_uuid = str(uuid.uuid4())
                self.known_faces[face_uuid] = {
                    "uuid": face_uuid,
                    "name": username,
                    "embedding": self._embedding_to_base64(avg_embedding),
                    "created_at": datetime.now().isoformat(),
                    "last_seen": None,
                    "meetings_count": 0
                }
                added_names.append(username)
                log.info(f"Added known face '{username}' from {len(embeddings)} embeddings")
        
        if added_names:
            self._save()
        self._mark_dirty()
        return added_names

    def remove_similar_unknown_faces(self, query_embedding: np.ndarray, 
                                      similarity_threshold: float = 0.85,
                                      max_count: int = 3) -> List[str]:
        """
        Удаляет unknown-лица, похожие на query_embedding.
        
        :param query_embedding: эмбеддинг known-лица
        :param similarity_threshold: порог сходства для удаления (cosine similarity)
        :param max_count: максимальное количество удаляемых лиц
        :return: список удалённых UUID
        """
        import faiss
        
        if not self.unknown_faces:
            return []
        
        # Собираем все unknown-эмбеддинги
        unknown_embs = []
        unknown_uuids = []
        
        for uuid_, data in self.unknown_faces.items():
            emb = self._base64_to_embedding(data["embedding"])
            unknown_embs.append(emb)
            unknown_uuids.append(uuid_)
        
        if not unknown_embs:
            return []
        
        # Строим временный FAISS-индекс для unknown-лиц
        dim = unknown_embs[0].shape[0]
        temp_index = faiss.IndexFlatIP(dim)
        temp_index.add(np.array(unknown_embs).astype('float32'))
        
        # Ищем похожие
        query_emb = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = temp_index.search(query_emb, k=min(max_count + 1, len(unknown_embs)))
        
        removed_uuids = []
        for i in range(len(indices[0])):
            score = distances[0][i]
            idx = indices[0][i]
            
            if score > similarity_threshold:
                uuid_to_remove = unknown_uuids[idx]
                del self.unknown_faces[uuid_to_remove]
                removed_uuids.append(uuid_to_remove)
                log.info(f"Removed duplicate unknown face (UUID {uuid_to_remove[:8]}) "
                        f"similar to known face (score={score:.3f})")
        
        if removed_uuids:
            self._save()
        self._mark_dirty()
        return removed_uuids

    def find_and_remove_similar_unknown(self, query_embedding: np.ndarray,
                                         similarity_threshold: float = 0.85) -> Optional[str]:
        """
        Находит и удаляет наиболее похожее unknown-лицо.
        Возвращает UUID удалённого лица или None.
        """
        if not self.unknown_faces:
            return None
        
        # Собираем unknown-эмбеддинги
        unknown_embs = []
        unknown_uuids = []
        
        for uuid_, data in self.unknown_faces.items():
            emb = self._base64_to_embedding(data["embedding"])
            unknown_embs.append(emb)
            unknown_uuids.append(uuid_)
        
        if not unknown_embs:
            return None
        
        # Строим временный FAISS-индекс
        dim = unknown_embs[0].shape[0]
        temp_index = faiss.IndexFlatIP(dim)
        temp_index.add(np.array(unknown_embs).astype('float32'))
        
        # Ищем наиболее похожее
        query_emb = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = temp_index.search(query_emb, k=1)
        
        score = distances[0][0]
        idx = indices[0][0]
        
        if score > similarity_threshold:
            uuid_to_remove = unknown_uuids[idx]
            del self.unknown_faces[uuid_to_remove]
            log.info(f"Removed similar unknown face (UUID {uuid_to_remove[:8]}) "
                    f"(score={score:.3f})")
            self._save()
            self._mark_dirty()
            return uuid_to_remove
        
        return None
