#!/usr/bin/env python3
"""
Локальный тестер face recognition.
Сравнивает known-лица из storage.json с .npy файлами из директории embeddings.

Использование:
    python test_local_face.py --storage meeting_storage.json
    python test_local_face.py --storage meeting_storage.json --embeddings core/face_analysis/embeddings
    python test_local_face.py --threshold 0.6
"""

import argparse
import base64
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np


def load_storage(storage_path: str) -> dict:
    """Загружает storage.json с known/unknown лицами."""
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


def load_npy_embeddings(embeddings_dir: str) -> dict:
    """Загружает .npy эмбеддинги и группирует по имени."""
    npy_files = sorted(glob.glob(os.path.join(embeddings_dir, '*.npy')))
    
    if not npy_files:
        print(f"❌ Нет .npy файлов в {embeddings_dir}")
        return {}
    
    print(f"📁 Загружено {len(npy_files)} .npy файлов")
    
    # Группируем по имени (без idx): "мастер-1.npy" -> "мастер"
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
        print(f"   • {name} ({len(embs)} эмбеддингов)")
    
    return name_embeddings


def find_similar(query_emb: np.ndarray, known_embeddings: list, threshold: float = 0.55) -> list:
    """Ищет похожие known-лица по cosine similarity."""
    matches = []
    
    for known in known_embeddings:
        score = np.dot(query_emb, known['embedding'])
        if score > threshold:
            matches.append({
                'name': known['name'],
                'uuid': known['uuid'],
                'score': score
            })
    
    # Сортируем по убыванию similarity
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    return matches


def test_local(storage_path: str, embeddings_dir: str, threshold: float):
    """Тестирует распознавание known-лиц среди .npy файлов."""
    print("\n" + "="*70)
    print("🧪 ЛОКАЛЬНЫЙ ТЕСТ FACE RECOGNITION")
    print("="*70)
    
    # Загружаем known-лица
    print("\n📦 Загрузка known-лиц из storage.json...")
    storage = load_storage(storage_path)
    known_embeddings = get_known_embeddings(storage)
    
    if not known_embeddings:
        print("❌ Нет known-лиц в storage.json")
        sys.exit(1)
    
    print(f"\n✅ Загружено {len(known_embeddings)} known-лиц:")
    for known in known_embeddings:
        print(f"   • {known['name']} (UUID: {known['uuid'][:8]})")
    
    # Загружаем .npy файлы
    print(f"\n📁 Загрузка .npy файлов из {embeddings_dir}...")
    name_embeddings = load_npy_embeddings(embeddings_dir)
    
    if not name_embeddings:
        print("❌ Нет .npy файлов для тестирования")
        sys.exit(1)
    
    # Тестируем каждое имя
    print(f"\n🔍 Поиск совпадений (threshold={threshold})...\n")
    
    results = []
    matched_names = set()
    
    for name, embs in name_embeddings.items():
        # Усредняем эмбеддинги для этого лица
        avg_emb = np.mean(embs, axis=0)
        
        # Ищем похожие known-лица
        matches = find_similar(avg_emb, known_embeddings, threshold)
        
        result = {
            'name': name,
            'matches': matches,
            'all_scores': []
        }
        
        if matches:
            matched_names.add(name)
            print(f"✅ {name}: НАЙДЕНО ({len(matches)} совпадений)")
            for i, match in enumerate(matches, 1):
                print(f"   {i}. {match['name']} — score: {match['score']:.4f}")
        else:
            print(f"❌ {name}: нет совпадений")
            # Показываем топ-3 ближайших
            all_scores = []
            for known in known_embeddings:
                score = np.dot(avg_emb, known['embedding'])
                all_scores.append({
                    'name': known['name'],
                    'score': score
                })
            
            all_scores.sort(key=lambda x: x['score'], reverse=True)
            result['all_scores'] = all_scores
            
            print(f"   Топ-3 ближайших:")
            for i, score_data in enumerate(all_scores[:3], 1):
                print(f"   {i}. {score_data['name']}: {score_data['score']:.4f}")
        
        results.append(result)
        print()
    
    # Итоговая статистика
    print("\n" + "="*70)
    print("📊 ИТОГИ")
    print("="*70)
    print(f"Всего .npy файлов: {len(name_embeddings)}")
    print(f"Найдено совпадений: {len(matched_names)}")
    print(f"Не найдено: {len(name_embeddings) - len(matched_names)}")
    
    # Показываем unmatched
    unmatched = [r for r in results if not r['matches']]
    if unmatched:
        print(f"\n❌ Не распознаны:")
        for r in unmatched:
            best = r['all_scores'][0] if r['all_scores'] else None
            if best:
                print(f"   • {r['name']} (ближайший: {best['name']} — {best['score']:.4f})")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Локальный тестер face recognition'
    )
    parser.add_argument(
        '--storage',
        type=str,
        default='meeting_storage.json',
        help='Путь к storage.json файлу'
    )
    parser.add_argument(
        '--embeddings',
        type=str,
        default='core/face_analysis/embeddings',
        help='Путь к директории с .npy файлами'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.55,
        help='Порог similarity для known-лиц (default: 0.55)'
    )
    
    args = parser.parse_args()
    
    # Проверяем файлы
    if not os.path.exists(args.storage):
        print(f"❌ Storage файл не найден: {args.storage}")
        sys.exit(1)
    
    if not os.path.exists(args.embeddings):
        print(f"❌ Директория с .npy не найдена: {args.embeddings}")
        sys.exit(1)
    
    test_local(args.storage, args.embeddings, args.threshold)


if __name__ == '__main__':
    main()
