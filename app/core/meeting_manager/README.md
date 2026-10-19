# Meeting Manager Module

Модуль для управления встречами с людьми. Отслеживает известные и неизвестные лица, отправляет уведомления в LLM-модуль.

## Возможности

1. **Распознавание известных лиц** - поиск по базе с сохранением UUID
2. **Обработка незнакомцев** - сохранение новых лиц с автоматическим UUID
3. **Отслеживание встреч** - определение первой/повторной встречи
4. **Уведомления в LLM** - отправка сообщений о встречах
5. **Приоритизация** - известные лица имеют приоритет над неизвестными
6. **Групповые уведомления** - обработка нескольких незнакомцев

## Архитектура

```
meeting_manager/
├── __init__.py          # Экспорт MeetingManager
├── storage.py           # Хранение данных (JSON + FAISS)
├── manager.py           # Основная логика обработки встреч
└── README.md            # Документация
```

## Использование

### Инициализация

```python
from pathlib import Path
from app.core.meeting_manager import MeetingManager

meeting_manager = MeetingManager(
    storage_path=Path("/path/to/storage.json"),
    llm_endpoint_url="http://localhost:8085/image_detect",
    detection_threshold=0.55
)
meeting_manager.start()
```

### Обработка обнаруженных лиц

```python
detected_faces = [
    {
        "embedding": np.array([...]),  # 128-мерный вектор
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.95
    }
]

result = meeting_manager.process_faces(detected_faces)
```

### Возвращаемые данные

```python
result = [
    {
        "embedding": np.array([...]),
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.95,
        "face_uuid": "abc123",
        "face_type": "known",  # или "unknown"
        "face_name": "ivanov",  # или None
        "is_first_meeting": True,
        "meeting_message": "Первая встреча сегодня с известным: ivanov"
    }
]
```

## Логика обработки

### Известные лица
- Если встреча первая сегодня → уведомление "Первая встреча"
- Если встреча повторная → уведомление с временем с последней встречи

### Неизвестные лица
- Первое появление → сохранение с UUID, уведомление "Первая встреча с незнакомцем"
- Повторное появление → уведомление с временем с последней встречи

### Несколько незнакомцев
- Если >1 незнакомец → отдельное групповое уведомление
- Указывается количество и время последних встреч

## Хранение данных

Данные сохраняются в JSON файл:
```json
{
  "known": {
    "uuid": {
      "uuid": "...",
      "name": "ivanov",
      "embedding": "base64...",
      "created_at": "2026-10-20T...",
      "last_seen": "2026-10-20T...",
      "meetings_count": 5
    }
  },
  "unknown": {
    "uuid": {
      "uuid": "...",
      "name": "unknown_abc123",
      "embedding": "base64...",
      "created_at": "2026-10-20T...",
      "last_seen": "2026-10-20T...",
      "meetings_count": 2
    }
  },
  "updated_at": "2026-10-20T..."
}
```

## Интеграция с LLM

Уведомления отправляются на эндпоинт LLM-модуля:
```
POST {llm_endpoint_url}
Content-Type: application/json

{"text": "Первая встреча сегодня с известным: ivanov"}
```

## Настройка

Параметры инициализации:
- `storage_path` - путь к JSON файлу хранения
- `llm_endpoint_url` - URL LLM-модуля для уведомлений
- `detection_threshold` - порог сходства лиц (0.55 по умолчанию)
- `meeting_check_interval` - интервал проверки встреч (300 сек)
