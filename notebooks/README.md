# Notebooks
Jupyter notebooks с экспериментами и анализом
# Датасеты 
## ClinTox
Dataset Description: The ClinTox dataset includes drugs that have failed clinical trials for toxicity reasons and also drugs that are associated with successful trials.
Task Description: Binary classification. Given a drug SMILES string, predict the clinical toxicity.
References: https://pubmed.ncbi.nlm.nih.gov/27642066/

## toxric_30_datasets: 
The expanded predictive toxicology dataset is sourced from TOXRIC, a comprehensive and standardized toxicology database

## tox 21
The Tox21 data set comprises 12,060 training samples and 647 test samples that represent chemical compounds. There are 801 "dense features" that represent chemical descriptors, such as molecular weight, solubility or surface area, and 272,776 "sparse features" that represent chemical substructures (ECFP10, DFS6, DFS8; stored in Matrix Market Format ). Machine learning methods can either use sparse or dense data or combine them. For each sample there are 12 binary labels that represent the outcome (active/inactive) of 12 different toxicological experiments. Note that the label matrix contains many missing values (NAs). The original data source and Tox21 challenge site is https://tripod.nih.gov/tox21/challenge/.