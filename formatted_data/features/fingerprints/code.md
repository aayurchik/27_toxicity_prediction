`morgan_fp.npz` - бинарная матрица, разреженный формат (только 2% данных не нули)  
`fp_smiles.csv` - таблица соответствий: index - номер строки в матрице, smiles - структура молекулы  

Матрица слишком большая для CSV (2.5+ GB в плотном формате)  
NPZ - эффективное хранение разреженных данных  
CSV - человекочитаемая часть   

```python
"""
Генерация Morgan Fingerprints для молекул  
Источник: RDKit Documentation - Molecular Fingerprints  
Morgan, H. L. (1965). The Generation of a Unique Machine 
Description for Chemical Structures-A Technique Developed at Chemical Abstracts Service.  
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from scipy import sparse
from tqdm import tqdm
import os

# Параметры обработки
input_csv = r"C:\Users\user\Desktop\токс\full_combined_cleaned.csv"  # входной файл с SMILES
out_dir = r"C:\Users\user\Desktop\токс\fp_data"  # директория для результатов

# Параметры Morgan fingerprints
radius = 2  # радиус для рассмотрения атомных окружений (обычно 2-3)
nBits = 2048  # размер битового вектора (стандартные значения: 1024, 2048, 4096)
batch_size = 50000  # размер батча для обработки (оптимизация использования памяти)

# Отключаем предупреждения RDKit для чистого вывода
RDLogger.DisableLog('rdApp.*')

# Чтение SMILES и фильтрация уникальных значений
# Используем set для быстрого определения уникальности
smiles_set = set()  # для проверки уникальности
smiles_list = []    # для сохранения порядка

# Чтение файла чанками для обработки больших файлов
for chunk in pd.read_csv(input_csv, usecols=['smiles'], chunksize=5000):
    for smi in chunk['smiles']:
        if pd.isna(smi):  # пропускаем NaN значения
            continue
        smi = smi.strip()  # удаляем пробелы
        if smi not in smiles_set:  # проверяем уникальность
            smiles_set.add(smi)
            smiles_list.append(smi)  # сохраняем в списке

N = len(smiles_list)  # общее количество уникальных SMILES
print(f"Всего уникальных SMILES: {N}")

# Подготовка для логирования некорректных SMILES
invalid_smiles = []  # список для невалидных SMILES

# Обработка батчами для экономии оперативной памяти
# Используем списки для координат ненулевых элементов разреженной матрицы
rows, cols = [], []  # rows - индексы молекул, cols - установленные биты

# Обрабатываем данные батчами с прогресс-баром
for start in tqdm(range(0, N, batch_size), desc="Обработка батчей"):
    batch_end = min(start + batch_size, N)  # конец текущего батча
    for i in range(start, batch_end):  # обработка молекул в батче
        smi = smiles_list[i]  # получаем SMILES строку
        mol = Chem.MolFromSmiles(smi)  # конвертируем SMILES в молекулу RDKit
        if mol is None:
            invalid_smiles.append(smi)  # сохраняем невалидные SMILES
            continue
        
        # Генерация Morgan fingerprint как битового вектора
        # Алгоритм рассматривает окружения каждого атома в заданном радиусе
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        
        # Получаем индексы установленных битов (ненулевые позиции)
        onbits = list(bv.GetOnBits())
        
        # Добавляем координаты в разреженную матрицу
        # Для каждой молекулы добавляем ее индекс и позиции установленных битов
        rows.extend([i] * len(onbits))  # дублируем индекс молекулы для каждого бита
        cols.extend(onbits)  # добавляем позиции установленных битов

# Создание финальной разреженной матрицы в формате CSR (Compressed Sparse Row)
# CSR формат эффективен для матричных операций и занимает меньше памяти
data = np.ones(len(rows), dtype=np.uint8)  # все ненулевые элементы = 1
# Создаем координатную матрицу (COO) и конвертируем в CSR
csr = sparse.coo_matrix((data, (rows, cols)), shape=(N, nBits), dtype=np.uint8).tocsr()
# Сохраняем разреженную матрицу в формате NPZ
sparse.save_npz(os.path.join(out_dir, 'morgan_fp.npz'), csr)
# Создаем CSV файл с соответствием индексов и SMILES
df_smiles = pd.DataFrame({
    'index': np.arange(N),  # индекс молекулы в матрице
    'smiles': smiles_list   # соответствующий SMILES
})
df_smiles.to_csv(os.path.join(out_dir, 'fp_smiles.csv'), index=False)

print("Обработка завершена.")
