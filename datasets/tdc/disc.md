 ## **Overview**:

https://tdcommons.ai/single_pred_tasks/tox/#carcinogens


## **Datasets csv from**: 

https://huggingface.co/scikit-fingerprints/datasets?p=1


Можно открыть все датасеты сразу в Jupyter. Но сначала запустить это и после сделать kernel restart:

!pip install datasets

!pip install huggingface_hub[hf_xet]

Пример выгрузки датасета: 

from datasets import load_dataset
ds = load_dataset("scikit-fingerprints/TDC_carcinogens_lagunin")

# Преобразуем в pandas DataFrame
df = ds["train"].to_pandas()
print(df.head())
