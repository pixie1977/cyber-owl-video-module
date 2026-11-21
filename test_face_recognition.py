#!/usr/bin/env python3
"""
Локальный скрипт для тестирования face recognition.
Загружает known-лица из storage.json и ищет похожие среди тестовых изображений.

Использование:
    python test_face_recognition.py --storage meeting_storage.json --images test_faces/
    python test_face_recognition.py --storage meeting_storage.json --image test_face.jpg
    python test_face_recognition.py --storage meeting_storage.json --embeddings embeddings/
"""

import argparse
import base64
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("❌ Установите insightface: pip install insightface")
    sys.exit(1)


def load_storage(storage_path: str) -> dict:
    """Загружает storage.json с known/unknown лицами."""
    import json
    
    with open(storage_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📦 Загружено known-лиц: {len(data.get('known', {}))}")
    print(f"📦 Загружено unknown-лиц: {len(data.get('unknown', {}))}")
    
    return data


def base64_to_embedding(base64_str: str) -> np.ndarray:
    """Конвертирует base64 строку в эмбеддинг."""
    return np.frombuffer(base64.b64decode(base64_str), dtype=np.float32)


def get_known_embeddings(storage: dict) -> list:
    """Извлекает known-эмбеддинги из storage."""
    known = storage.get('known', {})
    embeddings = []
    
    for uuid_, data in known.items():
        name = data.get('name', 'unknown')
        emb = base64_to_embedding(data['embedding'])
        embeddings.append({
            'uuid': uuid_,
            'name': name,
            'embedding': emb
        })
    
    return embeddings


def extract_embedding(image_path: str, face_app) -> tuple:
    """Извлекает embedding из изображения."""
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_app.get(rgb)
    
    if not faces:
        return img, None
    
    # Берём первое обнаруженное лицо
    face = faces[0]
    emb = face.normed_embedding
    
    return img, emb


def find_similar(emb: np.ndarray, known_embeddings: list, threshold: float = 0.55) -> list:
    """Ищет похожие known-лица."""
    matches = []
    
    for known in known_embeddings:
        score = np.dot(emb, known['embedding'])
        if score > threshold:
            matches.append({
                'name': known['name'],
                'uuid': known['uuid'],
                'score': score
            })
    
    # Сортируем по убыванию similarity
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    return matches


def test_images(image_paths: list, storage_path: str, threshold: float = 0.55):
    """Тестирует распознавание на списке изображений."""
    print("\n" + "="*60)
    print("🧪 ТЕСТ FACE RECOGNITION")
    print("="*60)
    
    # Загружаем known-лица
    print("\n📦 Загрузка known-лиц...")
    storage = load_storage(storage_path)
    known_embeddings = get_known_embeddings(storage)
    
    if not known_embeddings:
        print("❌ Нет known-лиц в storage.json")
        sys.exit(1)
    
    print(f"✅ Загружено {len(known_embeddings)} known-лиц:")
    for known in known_embeddings:
        print(f"   - {known['name']} (UUID: {known['uuid'][:8]})")
    
    # Инициализируем face_analysis
    print("\n🔍 Инициализация face_analysis...")
    face_app = FaceAnalysis(
        name='buffalo_l',
        root=str(Path(__file__).parent / 'venv' / 'insightface'),
        providers=['CPUExecutionProvider']
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("✅ Face analysis pipeline initialized")
    
    # Тестируем каждое изображение
    print(f"\n🖼 Тестируем {len(image_paths)} изображений...\n")
    
    for image_path in image_paths:
        print(f"\n{'─'*60}")
        print(f"📷 {image_path}")
        print("─"*60)
        
        img, emb = extract_embedding(image_path, face_app)
        
        if emb is None:
            print("❌ Лицо не обнаружено на изображении")
            continue
        
        print(f"✅ Лицо обнаружено, embedding extracted")
        
        # Ищем похожие
        matches = find_similar(emb, known_embeddings, threshold)
        
        if matches:
            print(f"\n✅ НАЙДЕНО {len(matches)} совпадений (threshold={threshold}):")
            for i, match in enumerate(matches, 1):
                print(f"   {i}. {match['name']}")
                print(f"      UUID: {match['uuid']}")
                print(f"      Score: {match['score']:.4f}")
        else:
            print(f"\n❌ Нет совпадений (threshold={threshold})")
            print(f"   Показываю топ-3 ближайших:")
            
            # Показываем топ-3 ближайших, даже если ниже threshold
            all_scores = []
            for known in known_embeddings:
                score = np.dot(emb, known['embedding'])
                all_scores.append({
                    'name': known['name'],
                    'score': score
                })
            
            all_scores.sort(key=lambda x: x['score'], reverse=True)
            for i, score_data in enumerate(all_scores[:3], 1):
                print(f"   {i}. {score_data['name']}: {score_data['score']:.4f}")
    
    print(f"\n{'='*60}")
    print("✅ ТЕСТ ЗАВЕРШЕН")
    print("="*60)


def test_embeddings(embeddings_dir: str, storage_path: str, threshold: float = 0.55):
    """Тестирует распознавание на .npy файлах из директории embeddings."""
    print("\n" + "="*60)
    print("🧪 ТЕСТ FACE RECOGNITION (из .npy файлов)")
    print("="*60)
    
    # Загружаем known-лица
    print("\n📦 Загрузка known-лиц...")
    storage = load_storage(storage_path)
    known_embeddings = get_known_embeddings(storage)
    
    if not known_embeddings:
        print("❌ Нет known-лиц в storage.json")
        sys.exit(1)
    
    print(f"✅ Загружено {len(known_embeddings)} known-лиц:")
    for known in known_embeddings:
        print(f"   - {known['name']} (UUID: {known['uuid'][:8]})")
    
    # Загружаем .npy файлы
    npy_files = sorted(glob.glob(os.path.join(embeddings_dir, '*.npy')))
    
    if not npy_files:
        print(f"❌ Нет .npy файлов в {embeddings_dir}")
        sys.exit(1)
    
    print(f"\n📁 Загружено {len(npy_files)} .npy файлов")
    
    # Группируем по имени (без idx)
    name_embeddings = {}
    for filepath in npy_files:
        filename = os.path.basename(filepath)
        name = filename.replace('.npy', '')
        parts = name.rsplit('-', 1)
        
        if len(parts) == 2 and parts[1].isdigit():
            username = parts[0]
        else:
            username = name
        
        emb = np.load(filepath)
        
        if username not in name_embeddings:
            name_embeddings[username] = []
        name_embeddings[username].append(emb)
    
    print(f"✅ Сгруппировано в {len(name_embeddings)} имён:")
    for name, embs in name_embeddings.items():
        print(f"   - {name} ({len(embs)} эмбеддингов)")
    
    # Тестируем каждое имя
    print(f"\n🖼 Тестируем {len(name_embeddings)} имён...\n")
    
    for name, embs in name_embeddings.items():
        # Усредняем эмбеддинги
        avg_emb = np.mean(embs, axis=0)
        
        print(f"\n{'─'*60}")
        print(f"📷 {name} ({len(embs)} эмбеддингов)")
        print("─"*60)
        
        matches = find_similar(avg_emb, known_embeddings, threshold)
        
        if matches:
            print(f"\n✅ НАЙДЕНО {len(matches)} совпадений (threshold={threshold}):")
            for i, match in enumerate(matches, 1):
                print(f"   {i}. {match['name']}")
                print(f"      UUID: {match['uuid']}")
                print(f"      Score: {match['score']:.4f}")
        else:
            print(f"\n❌ Нет совпадений (threshold={threshold})")
            print(f"   Показываю топ-3 ближайших:")
            
            all_scores = []
            for known in known_embeddings:
                score = np.dot(avg_emb, known['embedding'])
                all_scores.append({
                    'name': known['name'],
                    'score': score
                })
            
            all_scores.sort(key=lambda x: x['score'], reverse=True)
            for i, score_data in enumerate(all_scores[:3], 1):
                print(f"   {i}. {score_data['name']}: {score_data['score']:.4f}")
    
    print(f"\n{'='*60}")
    print("✅ ТЕСТ ЗАВЕРШЕН")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Тестирование face recognition на локальных данных'
    )
    parser.add_argument(
        '--storage',
        type=str,
        default='meeting_storage.json',
        help='Путь к storage.json файлу'
    )
    parser.add_argument(
        '--images',
        nargs='+',
        help='Список изображений для тестирования'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Одно изображение для тестирования'
    )
    parser.add_argument(
        '--embeddings',
        type=str,
        help='Директория с .npy файлами для тестирования'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.55,
        help='Порог similarity для known-лиц (default: 0.55)'
    )
    
    args = parser.parse_args()
    
    if not args.images and not args.image and not args.embeddings:
        parser.print_help()
        print("\n💡 Примеры:")
        print("  python test_face_recognition.py --storage meeting_storage.json --images test1.jpg test2.jpg")
        print("  python test_face_recognition.py --storage meeting_storage.json --image test.jpg")
        print("  python test_face_recognition.py --storage meeting_storage.json --embeddings core/face_analysis/embeddings")
        sys.exit(1)
    
    # Проверяем storage файл
    if not os.path.exists(args.storage):
        print(f"❌ Storage файл не найден: {args.storage}")
        sys.exit(1)
    
    # Тестируем изображения
    if args.images:
        test_images(args.images, args.storage, args.threshold)
    
    if args.image:
        test_images([args.image], args.storage, args.threshold)
    
    # Тестируем .npy файлы
    if args.embeddings:
        test_embeddings(args.embeddings, args.storage, args.threshold)


if __name__ == '__main__':
    main()
