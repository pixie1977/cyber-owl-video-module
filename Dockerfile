# Используем официальный образ Python 3.10 (слабая версия для минимального размера)
FROM python:3.10-slim

# Устанавливаем метаданные
LABEL maintainer="you@example.com"
LABEL description="Video server with FastAPI"

# Устанавливаем рабочую директорию
WORKDIR /app

# Системные зависимости
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libportaudio2 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы проекта
COPY . .

# Делаем start.sh исполняемым
RUN chmod +x /app/start.sh

# Открываем порт
EXPOSE $TTS_PORT

# Точка входа
CMD ["/app/start.sh"]