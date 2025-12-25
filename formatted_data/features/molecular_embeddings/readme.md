https://drive.google.com/drive/folders/1507fWPBzKVexMTF6BxvOMeU_2COTuuBA?usp=sharing
Код запущен на Google Colab

СТАТИСТИКА ЭМБЕДДИНГОВ:  

Общее количество: 339,061  
Размерность: 300  
Min: -1.044072  
Max: 1.102271  
Mean: -0.004316  
Std:  0.111459  
NaN values: 0  

```python
"""
Источник: https://github.com/samoturk/mol2vec  
Jaeger, S., Fulle, S., & Turk, S. (2018)  
Mol2Vec: Unsupervised Machine Learning Approach with Chemical Intuition  
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from joblib import Parallel, delayed
from gensim.models import Word2Vec
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Создаем папку для сохранения результатов обработки
# Все файлы сохранены в указанную директорию
results_dir = '/content/drive/MyDrive/Colab Notebooks/mol2vec_results'
os.makedirs(results_dir, exist_ok=True)

def patched_sentences2vec(sentences, model, unseen=None):
    """
    Исправленная версия sentences2vec для совместимости с Gensim 4.x
    Патч необходим из-за изменений в API Gensim между версиями 3.x и 4.x
    Оригинальная функция: https://github.com/samoturk/mol2vec/blob/master/mol2vec/features.py
    
    Args:
        sentences: список "предложений" молекулы (атомные окружения)
        model: предобученная Word2Vec модель
        unseen: стратегия для незнакомых слов
    Returns:
        list: вектора для каждого предложения
    """
    # Получаем словарь модели (замена устаревшему model.wv.vocab в Gensim 4.x)
    keys = model.wv.key_to_index
    # Создаем вектор для незнакомых слов (out-of-vocabulary)
    unseen_vec = None
    if unseen is not None:
        if unseen in keys:
            unseen_vec = model.wv[unseen]  # используем существующий вектор
        else:
            # генерируем случайный вектор для неизвестных слов
            unseen_vec = np.random.normal(scale=0.6, size=(model.vector_size,))
    
    vec = []
    for sentence in sentences:
        if not sentence:
            continue  # пропускаем пустые предложения
            
        word_vectors = []
        for word in sentence:
            if word in keys:
                # если слово есть в словаре, берем его вектор
                word_vectors.append(model.wv[word])
            elif unseen_vec is not None:
                # используем вектор для неизвестных слов
                word_vectors.append(unseen_vec)
        
        if word_vectors:
            # усредняем вектора всех слов в предложении
            vec.append(np.mean(word_vectors, axis=0))
        else:
            # возвращаем нулевой вектор если нет слов
            vec.append(np.zeros(model.vector_size))
    
    return vec

# Пути к входным и выходным файлам
input_path = '/content/drive/MyDrive/Colab Notebooks/full_combined_cleaned.csv'  # исходные данные
output_path_csv = f'{results_dir}/mol2vec_embeddings_full.csv'  # выходной файл с эмбеддингами
# Загрузка и предобработка SMILES данных
# Читаем только колонку с SMILES для экономии памяти
df = pd.read_csv(input_path, usecols=['smiles'])
# удаляем дубликаты SMILES
df = df.drop_duplicates(subset=['smiles']).reset_index(drop=True)

# Загрузка предобученной модели Mol2Vec
# Модель содержит 300-мерные вектора для атомных окружений
try:
    # Пытаемся загрузить локальную копию модели
    model = Word2Vec.load('/content/drive/MyDrive/Colab Notebooks/model_300dim.pkl')
except Exception as e:
    # Если локальная модель недоступна, скачиваем стандартную
    # Модель обучена на 20 миллионах соединений из ZINC
    os.system('wget -q https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl')
    model = Word2Vec.load('model_300dim.pkl')

def process_smiles(smiles):
    """
    Обрабатывает один SMILES для генерации Mol2Vec эмбеддинга
    Процесс: SMILES -> RDKit Mol -> атомные окружения -> усредненный вектор
    
    Args:
        smiles (str): SMILES строка молекулы
    Returns:
        tuple: (вектор эмбеддинга, исходный SMILES, статус обработки)
    """
    try:
        # Конвертируем SMILES в молекулярный объект RDKit
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, smiles, "invalid_smiles"  # невалидный SMILES
        # Генерируем 'предложение' из атомных окружений
        # radius=1 означает рассмотрение соседей на расстоянии 1 связи
        from mol2vec.features import mol2alt_sentence
        sentence = mol2alt_sentence(mol, radius=1)
        
        if not sentence:
            return None, smiles, "empty_sentence"  # пустое предложение
        
        # Конвертируем предложение в 300-мерный вектор
        # unseen='UNK' определяет стратегию для неизвестных атомных окружений
        vec = patched_sentences2vec([sentence], model, unseen='UNK')[0]
        
        return vec, smiles, "success"  # успешная обработка
        
    except Exception as e:
        # Обработка любых других ошибок
        return None, smiles, f"error: {str(e)}"

# Параллельная обработка всех молекул с использованием всех доступных ядер CPU
# Joblib распределяет обработку по ядрам, tqdm показывает прогресс-бар
results = Parallel(n_jobs=-1)(
    delayed(process_smiles)(smiles) 
    for smiles in tqdm(df['smiles'], desc="Generating Mol2Vec embeddings"))
# Сбор успешных результатов из параллельной обработки
successful_embeddings = []  # список успешных векторов
successful_smiles = []      # соответствующие SMILES строки

for vec, smiles, status in results:
    if status == "success":
        successful_embeddings.append(vec)
        successful_smiles.append(smiles)

# Сохранение эмбеддингов в CSV файл
if successful_embeddings:
    # Конвертируем список векторов в numpy array для эффективности
    emb_array = np.array(successful_embeddings)
    # Создаем названия колонок: mol2vec_0, mol2vec_1, ..., mol2vec_299
    emb_cols = [f"mol2vec_{i}" for i in range(emb_array.shape[1])]
    # Создаем DataFrame с эмбеддингами и SMILES
    emb_df = pd.DataFrame(emb_array, columns=emb_cols)
    emb_df.insert(0, 'smiles', successful_smiles)  # добавляем SMILES в первую колонку
    # Сохраняем в CSV файл
    emb_df.to_csv(output_path_csv, index=False)
    print(f"Размерность данных: {emb_df.shape}")
