# Dataset Summary

## Project task

The goal is to classify each medical abstract into one of five disease categories. This is a **single-label multiclass classification** task, so every abstract must have exactly one label.

## Original dataset

The original dataset contained **14,438 rows** from:

- `data/raw/medical_tc_train.csv`
- `data/raw/medical_tc_test.csv`
- `data/raw/medical_tc_labels.csv` (label names)

We did not keep the original train/test split because the audit found duplicate abstracts, conflicting labels for identical text, and exact abstract overlap between the original train and test files. Keeping that split would risk data leakage and unreliable evaluation.

## Cleaning decision

Rows were grouped by exact abstract text. Because the project requires one label per abstract, every abstract associated with multiple labels was treated as ambiguous and removed. The remaining data contains one row per unique abstract.

The final clean dataset contains **8,298 unique abstracts**:

| Disease category | Abstracts |
|---|---:|
| Neoplasms | 2,195 |
| Digestive system diseases | 699 |
| Nervous system diseases | 1,049 |
| Cardiovascular diseases | 1,961 |
| General pathological conditions | 2,394 |

The medical text was preserved as written. It was not lowercased, stemmed, lemmatized, stripped of punctuation or stopwords, or otherwise aggressively preprocessed.

## Recreated splits

New train, validation, and test sets were created from the clean dataset using stratification by label and a fixed random seed of **42**. This keeps class proportions approximately consistent and makes the split reproducible.

- `data/processed/train.csv`: use for model training.
- `data/processed/validation.csv`: use for model comparison and tuning.
- `data/processed/test.csv`: use only for final evaluation after model choices are complete.
- `data/processed/medical_abstracts_clean.csv`: the complete cleaned master dataset; do not use it to bypass the defined splits.

All audit, cleaning, and splitting steps are reproducible through the numbered Python scripts in `src/`.
