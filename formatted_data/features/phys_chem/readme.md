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
