## 1. SIDER Database
**File:** `sider.csv`

**Description:**   
SIDER - Database of marketed drugs and adverse drug reactions (ADR), grouped into 27 system organ classes.

**References:**  
[MoleculeNet Datasets](https://moleculenet.org/datasets-1)

---

## 2. ToxCast Toxicology Data
**File:** `toxcast_data`

**Description:**   
ToxCast - Toxicology data for a large library of compounds based on in vitro high-throughput screening, including experiments on over 600 tasks.

**References:**  
[MoleculeNet Datasets](https://moleculenet.org/datasets-1)  
[PubMed Article](https://pubmed.ncbi.nlm.nih.gov/27367298/)

---

## 3. Ames Mutagenicity Benchmark
**File:** `ToxBenchmark_Ames_mutagenicity`

**Description:**   
ToxBenchmark - A public benchmark data set of 6,512 chemical compounds with their Ames mutagenicity test results (2-class classification problem). Designed for evaluation of in silico prediction methods.

**Scientific Reference:**  
Katja Hansen, Sebastian Mika, Timon Schroeter, Andreas Sutter, Antonius ter Laak, Thomas Steger-Hartmann, Nikolaus Heinrich and Klaus-Robert Müller.
"Benchmark Data Set for in Silico Prediction of Ames Mutagenicity". Journal of Chemical Information and Modelling, DOI: [10.1021/ci900161g](https://doi.org/10.1021/ci900161g)  

**Dataset Reference:**  
[ToxBenchmark Documentation](https://doc.ml.tu-berlin.de/toxbenchmark/)


## ClinTox
Dataset Description: The ClinTox dataset includes drugs that have failed clinical trials for toxicity reasons and also drugs that are associated with successful trials.
Task Description: Binary classification. Given a drug SMILES string, predict the clinical toxicity.
References: https://pubmed.ncbi.nlm.nih.gov/27642066/

## toxric_30_datasets: 
The expanded predictive toxicology dataset is sourced from TOXRIC, a comprehensive and standardized toxicology database

## tox 21
The Tox21 data set comprises 12,060 training samples and 647 test samples that represent chemical compounds. There are 801 "dense features" that represent chemical descriptors, such as molecular weight, solubility or surface area, and 272,776 "sparse features" that represent chemical substructures (ECFP10, DFS6, DFS8; stored in Matrix Market Format ). Machine learning methods can either use sparse or dense data or combine them. For each sample there are 12 binary labels that represent the outcome (active/inactive) of 12 different toxicological experiments. Note that the label matrix contains many missing values (NAs). The original data source and Tox21 challenge site is https://tripod.nih.gov/tox21/challenge/.

**Tox21 Assays Расшифровка:**  
- SR-HSE - Heat Shock Element Response  
- SR-ARE - Antioxidant Response Element  
- SR-MMP - Mitochondrial Membrane Potential  
- SR-p53 - p53 Tumor Suppressor Pathway  
- SR-ATAD5 - DNA Damage Response (ATAD5 biomarker)  
- NR-AR - Androgen Receptor  
- NR-AR-LBD - Androgen Receptor Ligand Binding Domain  
- NR-Aromatase - Aromatase Enzyme Inhibition  
- NR-ER - Estrogen Receptor  
- NR-ER-LBD - Estrogen Receptor Ligand Binding Domain  
- NR-AhR - Aryl Hydrocarbon Receptor  
- NR-PPAR-gamma - Peroxisome Proliferator-Activated Receptor Gamma  
- SR-анализы - измеряют клеточный стресс и повреждения ДНК
- NR-анализы - оценивают взаимодействие с гормональными рецепторами
