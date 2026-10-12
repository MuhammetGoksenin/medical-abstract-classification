# Muhammet Göksenin

- Student ID: 2509011060
- GitHub: MuhammetGoksenin
- Role: Final testing and evaluation

## What I did

- **DistilBERT:** Fine-tuned `distilbert-base-uncased` with the Hugging Face `Trainer` (max length 512, learning rate 2e-5, batch size 16, up to 4 epochs with early stopping, weight decay 0.01, warmup). Validation macro F1 0.8027, test macro F1 0.7818.
- **Classical models:** Ran a 90-configuration validation grid for Naive Bayes, Logistic Regression and Linear SVM (bag-of-words vs TF-IDF, unigrams vs bigrams, regularization and class weights), and evaluated the selected configurations on the test set.
- **Final benchmark:** Collected and checked all team results, defined baselines vs new methods, separated architecture from specific model, and prepared the benchmark tables and charts.
- **Error analysis:** Analysed per-class F1 for all models and the high-confidence errors of DistilBERT; showed that most errors involve the General Pathological Conditions class.
- **Calibration analysis:** Measured how trustworthy model confidence is (reliability diagram, ECE, confidence thresholds for human review).
- **Documentation:** Wrote the README and BENCHMARK.md.

## Files I worked on

- `notebooks/05_distilbert_classifier.ipynb`
- `src/run_classical.py`
- `src/calibration_analysis.py`
- `results/`, `figures/`
- `README.md`, `BENCHMARK.md`
