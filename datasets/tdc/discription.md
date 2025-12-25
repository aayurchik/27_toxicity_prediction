## Overview

**Source:**  
[TDC Commons - Toxicity Tasks](https://tdcommons.ai/single_pred_tasks/tox/#carcinogens)

## Datasets

CSV datasets available from:  
[HuggingFace Datasets Repository](https://huggingface.co/scikit-fingerprints/datasets?p=1)

## Installation & Setup  

Datasets can be opened directly in Jupyter. First run these commands, then restart the kernel:

```bash
!pip install datasets
!pip install huggingface_hub[hf_xet]

# Example: Loading carcinogens dataset
from datasets import load_dataset

ds = load_dataset("scikit-fingerprints/TDC_carcinogens_lagunin")
df = ds["train"].to_pandas()
print(df.head())
