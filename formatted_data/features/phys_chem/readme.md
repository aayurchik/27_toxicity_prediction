# Объединение частей Parquet-файлов в один DataFrame

```python
import pandas as pd
import glob

# Указать свой путь к частям Parquet-файлов (пример на моей папке)
files = glob.glob(r"C:\Users\user\Desktop\токс\Новая папка\descriptors_part*.parquet")  # звездочка перебирает номера

# читаем все части в список DataFrame
dfs = [pd.read_parquet(f) for f in files]  # каждый файл - отдельный DataFrame

# объединить все части в один DataFrame
df_full = pd.concat(dfs, ignore_index=True)

# проверка
print(df_full.shape)  # выводит размер
df_full.head()         # первые строки

# Должен быть размер: (339061, 107)


______________________________________________________________________________________________________________________

"""
Генерация физико-химических дескрипторов для молекул
Вычисляет 107 молекулярных свойств на основе SMILES структур
Источник: RDKit - Open-Source Cheminformatics Software  
List of Available Descriptors: https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors
RDKit Documentation: https://www.rdkit.org/docs/GettingStartedInPython.html
MoleculeDescriptors: https://www.rdkit.org/docs/source/rdkit.ML.Descriptors.html
Parallel processing: https://joblib.readthedocs.io/
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from joblib import Parallel, delayed
from tqdm import tqdm
import os

# Параметры обработки данных
input_path = r"C:\Users\user\Desktop\токс\full_combined_cleaned.csv"  # исходный файл с SMILES
output_path = r"C:\Users\user\Desktop\токс\smiles_physchem_descriptors.csv"  # выходной файл с дескрипторами
batch_size = 5000  # размер батча для обработки (оптимизация использования памяти)
n_jobs = -1        # количество ядер CPU для параллельной обработки (-1 = все доступные ядра)

# Загрузка и предобработка SMILES данных
# Читаем только колонку с SMILES для экономии памяти
df = pd.read_csv(input_path, usecols=['smiles'])
df = df.drop_duplicates(subset=['smiles']).reset_index(drop=True)  # удаление дубликатов SMILES

# Выбор физико-химических дескрипторов из RDKit
# Получаем все доступные дескрипторы из RDKit
all_desc = Descriptors._descList  # список всех дескрипторов (имя, функция)
all_names = [name for name, _ in all_desc]  # извлекаем только имена дескрипторов

# Ключевые слова для фильтрации дескрипторов
# Отбираем наиболее релевантные для токсикологии дескрипторы
keywords = [
    "Mol",      # молекулярные свойства (вес, рефракция)
    "LogP",     # липофильность
    "TPSA",     # полярная поверхность
    "VSA",      # ван-дер-ваальсовы поверхности
    "PEOE",     # парциальные заряды
    "SMR",      # молярная рефракция
    "SlogP",    # липофильность
    "Num",      # счетчики (атомы, связи, кольца)
    "Fraction", # доли и соотношения
    "Ring",     # свойства колец
    "Heavy",    # тяжелые атомы
    "Balaban",  # топологические индексы
    "Bertz",    # сложность молекулы
    "Kappa",    # молекулярная форма
    "Chi",      # коннективностные индексы
    "HallKier", # альфа-индексы
    "Labute",   # ASA поверхности
    "ExactMolWt", # точный молекулярный вес
    "MolMR"     # молярная рефракция]

# Фильтрация дескрипторов по ключевым словам
# Создаем множество для автоматического удаления дубликатов
selected_names = sorted({
    name for name in all_names
    if any(kw.lower() in name.lower() for kw in keywords)})

# Создание калькулятора дескрипторов
# Калькулятор будет вычислять все выбранные дескрипторы за один вызов
calc = MoleculeDescriptors.MolecularDescriptorCalculator(selected_names)
def safe_calc(smiles):
    """
    Безопасное вычисление дескрипторов для одного SMILES
    Обрабатывает ошибки и невалидные структуры
    
    Args:
        smiles (str): SMILES строка молекулы
    Returns:
        list: список значений дескрипторов или NaN при ошибке
    """
    try:
        # Конвертация SMILES в молекулярный объект RDKit
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Возвращаем NaN для всех дескрипторов если SMILES невалидный
            return [np.nan] * len(selected_names)
        # Вычисление всех выбранных дескрипторов
        return list(calc.CalcDescriptors(mol))
        
    except Exception:
        # Возвращаем NaN при любых ошибках вычисления
        return [np.nan] * len(selected_names)

# Параллельное вычисление дескрипторов батчами
results = []  # список для хранения результатов
# Обработка данных батчами для контроля использования памяти
for start in tqdm(range(0, len(df), batch_size), desc="Обработка батчей"):
    # Извлечение батча SMILES
    batch_smiles = df['smiles'].iloc[start:start+batch_size].values
    # Параллельное вычисление дескрипторов для батча
    # Используем все доступные ядра CPU
    batch_vals = Parallel(n_jobs=n_jobs)(
        delayed(safe_calc)(s) for s in batch_smiles)
    # Добавление результатов батча в общий список
    results.extend(batch_vals)

# Формирование итогового DataFrame
# Конвертация списка результатов в numpy array для эффективности
desc_array = np.array(results)

# Создание DataFrame с дескрипторами
desc_df = pd.DataFrame(desc_array, columns=selected_names)
# Добавление колонки с SMILES в начало таблицы
desc_df.insert(0, 'smiles', df['smiles'].values)

# Пост-обработка данных: удаление константных колонок
# Константные колонки не несут информации для машинного обучения
numeric = desc_df.drop(columns=['smiles'])  # временный DataFrame без SMILES

# Поиск колонок с одним или менее уникальных значений
const_cols = numeric.columns[numeric.nunique(dropna=False) <= 1].tolist()
if const_cols:
    # Удаление константных колонок из основного DataFrame
    desc_df = desc_df.drop(columns=const_cols)
    print(f"Удалено константных колонок: {len(const_cols)}")

# Сохранение итогового CSV файла
desc_df.to_csv(output_path, index=False)
print(f"Финальный CSV с дескрипторами сохранен: {output_path}")
