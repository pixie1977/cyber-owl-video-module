#!/bin/bash
set -e

echo "🚀 Запуск Cyber Owl Video сервера..."

# Проверка и активация виртуального окружения
VENV_DIR="./venv"
PYTHON="$VENV_DIR/bin/python"

if [ ! -d "$VENV_DIR" ]; then
    echo "📁 Создаём виртуальное окружение..."
    python3 -m venv "$VENV_DIR"
else
    echo "✅ Виртуальное окружение найдено."
fi

# Активация venv
source "$VENV_DIR/bin/activate"

# Установка зависимостей, если требуется
if [ ! -f "$VENV_DIR/.requirements_installed" ] || [ "requirements.txt" -nt "$VENV_DIR/.requirements_installed" ]; then
    echo "📦 Устанавливаем зависимости из requirements.txt..."
    pip install --no-cache-dir -r requirements.txt
    touch "$VENV_DIR/.requirements_installed"
else
    echo "✅ Зависимости уже установлены."
fi

# Запуск приложения
echo "▶️ Запуск app.main..."
exec python -m app.main