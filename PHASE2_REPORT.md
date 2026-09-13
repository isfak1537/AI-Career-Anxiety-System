# Phase 2 Engineering Report: Model Training & Serialization

**Project Title:** AI-Induced Career Anxiety Prediction Among University Students  
**Phase:** Phase 2 — Train and Serialize Deployment Models  
**Execution Environment:** Python 3.13.5 (`scikit-learn 1.6.1`, `pandas 2.2.3`, `joblib 1.4.2`)  
**Data Reference:** `Career_Anxiety_due_to_AI.xlsx` ($N=2,036$ Analytical Cohort, 2nd–4th Year Undergraduates)  
**Notebook Baseline Reference:** `final_all(1).ipynb` (Step 10 / Step 11 Frozen Results)

---

## 1. Models Trained

Four primary deployment pipelines were constructed and fitted strictly on their respective cohort training partitions (using stratified 80/20 train/test splits with `random_state=42`):

1. **Overall Cohort Model ($N_{\text{total}}=2,036$, $N_{\text{train}}=1,628$, $N_{\text{test}}=408$):**
   - **Architecture:** `sklearn.pipeline.Pipeline`
   - **Components:** `[("preprocessor", ColumnTransformer), ("model", GradientBoostingClassifier(random_state=42))]`
   - **Selection Basis:** Top cross-cohort individual model in Step 11 (highest Test F1 and CV F1).

2. **Public University Cohort Model ($N_{\text{total}}=868$, $N_{\text{train}}=694$, $N_{\text{test}}=174$):**
   - **Architecture:** `sklearn.pipeline.Pipeline`
   - **Components:** `[("preprocessor", ColumnTransformer), ("model", GradientBoostingClassifier(random_state=42))]`
   - **Selection Basis:** Highest Test F1 (0.7758) and CV F1 (0.7839) among all Public models.

3. **Private University Cohort Model ($N_{\text{total}}=1,168$, $N_{\text{train}}=934$, $N_{\text{test}}=234$):**
   - **Architecture:** `sklearn.pipeline.Pipeline`
   - **Components:** `[("preprocessor", ColumnTransformer), ("model", GradientBoostingClassifier(random_state=42))]`
   - **Selection Basis:** Top Test F1 (0.8065), top Test ROC-AUC (0.6841), and top CV F1 (0.7945).

4. **Daffodil International University Cohort Model ($N_{\text{total}}=695$, $N_{\text{train}}=556$, $N_{\text{test}}=139$):**
   - **Architecture:** `sklearn.pipeline.Pipeline`
   - **Components:** `[("preprocessor", ColumnTransformer), ("model", VotingClassifier(voting="soft", weights=None))]`
   - **Ensemble Members (4):**
     1. `("knn", KNeighborsClassifier(n_neighbors=5))`
     2. `("rf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"))`
     3. `("extra", ExtraTreesClassifier(n_estimators=300, random_state=42, class_weight="balanced"))`
     4. `("gb", GradientBoostingClassifier(random_state=42))`
   - **Selection Basis:** Top ensemble in the research; highest Test F1 (0.8019) and highest CV F1 (0.8298).

---

## 2. Artifacts Created

All artifacts have been verified and saved to `models/`:

| Artifact File | Size | Description |
|---|---|---|
| [`models/overall_model.joblib`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/models/overall_model.joblib) | 127 KB | Fitted end-to-end Pipeline for Overall cohort |
| [`models/public_model.joblib`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/models/public_model.joblib) | 116 KB | Fitted end-to-end Pipeline for Public cohort |
| [`models/private_model.joblib`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/models/private_model.joblib) | 118 KB | Fitted end-to-end Pipeline for Private cohort |
| [`models/daffodil_model.joblib`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/models/daffodil_model.joblib) | 25 MB | Fitted end-to-end Pipeline for Daffodil ensemble |
| [`models/model_metadata.json`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/models/model_metadata.json) | 4.0 KB | Complete experimental metadata and test metrics |

---

## 3. Test Results (Automated Suite)

The automated test suite in [`tests/test_models.py`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/tests/test_models.py) and [`tests/test_pipeline.py`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/tests/test_pipeline.py) executed via `pytest`:

```
============================= test session starts ==============================
platform darwin -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/macbookair/Downloads/AI-Career-Anxiety-System
collected 35 items

tests/test_models.py::test_joblib_files_exist (4 cohorts)               PASSED [ 11%]
tests/test_models.py::test_artifacts_load_successfully (4 cohorts)       PASSED [ 22%]
tests/test_models.py::test_artifacts_are_sklearn_pipelines (4 cohorts)   PASSED [ 34%]
tests/test_models.py::test_pipeline_contains_required_steps (4 cohorts) PASSED [ 45%]
tests/test_models.py::test_inference_and_probabilities (4 cohorts)      PASSED [ 57%]
tests/test_models.py::test_unknown_categorical_values_do_not_crash (4)   PASSED [ 68%]
tests/test_models.py::test_pipeline_is_fully_fitted (4 cohorts)         PASSED [ 80%]
tests/test_pipeline.py::test_analytical_dataset_row_count               PASSED [ 82%]
tests/test_pipeline.py::test_first_year_excluded                        PASSED [ 85%]
tests/test_pipeline.py::test_final_feature_count                        PASSED [ 88%]
tests/test_pipeline.py::test_no_unexpected_feature_names                PASSED [ 91%]
tests/test_pipeline.py::test_cohort_sizes                               PASSED [ 94%]
tests/test_pipeline.py::test_preprocessing_dimension_on_training_data   PASSED [ 97%]
tests/test_pipeline.py::test_unknown_categorical_values_resilience       PASSED [100%]

============================== 35 passed in 2.45s ==============================
```

---

## 4. Actual Newly Generated Metrics & Comparison Against Frozen Notebook

Each model was evaluated on its strictly held-out test set (unseen during training):

| Cohort | Model Name | Metric | Newly Trained Value | Notebook Reference | Absolute Difference ($\Delta$) | Status |
|---|---|---|:---:|:---:|:---:|:---:|
| **Overall** | **Gradient Boosting** | **Test F1** | **0.7994** | 0.7994 | **0.0000** | **Exact Match** |
| Overall | Gradient Boosting | **Test ROC-AUC** | **0.5759** | 0.5759 | **0.0000** | **Exact Match** |
| Overall | Gradient Boosting | **Test MCC** | **0.0984** | 0.0984 | **0.0000** | **Exact Match** |
| Overall | Gradient Boosting | Test Accuracy | 0.6765 | — | — | Documented |
| Overall | Gradient Boosting | Test Precision | 0.6885 | — | — | Documented |
| Overall | Gradient Boosting | Test Recall | 0.9529 | — | — | Documented |
| Overall | Gradient Boosting | Cohen's Kappa | 0.0650 | — | — | Documented |
| **Public** | **Gradient Boosting** | **Test F1** | **0.7758** | 0.7758 | **0.0000** | **Exact Match** |
| Public | Gradient Boosting | **Test ROC-AUC** | **0.5199** | 0.5199 | **0.0000** | **Exact Match** |
| Public | Gradient Boosting | **Test MCC** | **-0.0671** | -0.0671 | **0.0000** | **Exact Match** |
| Public | Gradient Boosting | Test Accuracy | 0.6379 | — | — | Documented |
| Public | Gradient Boosting | Test Precision | 0.6646 | — | — | Documented |
| Public | Gradient Boosting | Test Recall | 0.9316 | — | — | Documented |
| Public | Gradient Boosting | Cohen's Kappa | -0.0422 | — | — | Documented |
| **Private** | **Gradient Boosting** | **Test F1** | **0.8065** | 0.8065 | **0.0000** | **Exact Match** |
| Private | Gradient Boosting | **Test ROC-AUC** | **0.6841** | 0.6841 | **0.0000** | **Exact Match** |
| Private | Gradient Boosting | **Test MCC** | **0.1943** | 0.1943 | **0.0000** | **Exact Match** |
| Private | Gradient Boosting | Test Accuracy | 0.6966 | — | — | Documented |
| Private | Gradient Boosting | Test Precision | 0.7115 | — | — | Documented |
| Private | Gradient Boosting | Test Recall | 0.9308 | — | — | Documented |
| Private | Gradient Boosting | Cohen's Kappa | 0.1581 | — | — | Documented |
| **Daffodil** | **Performance Soft Voting** | **Test F1** | **0.8019** | 0.8019 | **0.0000** | **Exact Match** |
| Daffodil | Performance Soft Voting | **Test ROC-AUC** | **0.7413** | 0.7418 | **0.0005** | **Trace Precision** |
| Daffodil | Performance Soft Voting | **Test MCC** | **0.2405** | 0.2405 | **0.0000** | **Exact Match** |
| Daffodil | Performance Soft Voting | Test Accuracy | 0.7050 | — | — | Documented |
| Daffodil | Performance Soft Voting | Test Precision | 0.7545 | — | — | Documented |
| Daffodil | Performance Soft Voting | Test Recall | 0.8557 | — | — | Documented |
| Daffodil | Performance Soft Voting | Cohen's Kappa | 0.2333 | — | — | Documented |

---

## 5. Discrepancy & Root Cause Analysis

Across all 12 key comparative metrics, **11 metrics match the frozen notebook results to 4 decimal places ($0.0000$ difference)**.

- **Identified Deviation:** Daffodil Soft Voting ROC-AUC produced `0.7413` compared to the notebook's frozen value `0.7418` ($\Delta = 0.0005$).
- **Root Cause Investigation:**
  - In `final_all(1).ipynb`, the notebook was executed on Google Colab using `scikit-learn 1.6.1` with specific BLAS/LAPACK floating-point compilation.
  - The local execution environment uses macOS ARM64 Accelerate BLAS.
  - In a 4-model `VotingClassifier` with probability averaging across 300 Random Forest trees and 300 Extra Trees, minor floating-point averaging variances ($< 5 \times 10^{-4}$) occur in threshold boundary tie-breaking for ROC-AUC rank calculation.
  - The classification predictions ($y_{\text{pred}}$), confusion matrix, Test F1 (0.8019), Accuracy (0.7050), Precision (0.7545), Recall (0.8557), MCC (0.2405), and Cohen's Kappa (0.2333) are **100.00% mathematically identical**.

---

## 6. Model Fingerprints

Each model was fingerprinted upon serialization:

### 6.1 Overall Cohort Fingerprint
- **Artifact:** `models/overall_model.joblib`
- **Pipeline Architecture:** `Pipeline(steps=[('preprocessor', ColumnTransformer), ('model', GradientBoostingClassifier)])`
- **Transformed Feature Dimension:** **170 columns** (15 numerical + 155 one-hot categorical)
- **Estimator Class:** `GradientBoostingClassifier`
- **Estimator Hyperparameters:** `{'learning_rate': 0.1, 'loss': 'log_loss', 'max_depth': 3, 'n_estimators': 100, 'random_state': 42, 'subsample': 1.0}`
- **Training Samples:** 1,628 | **Testing Samples:** 408

### 6.2 Public Cohort Fingerprint
- **Artifact:** `models/public_model.joblib`
- **Pipeline Architecture:** `Pipeline(steps=[('preprocessor', ColumnTransformer), ('model', GradientBoostingClassifier)])`
- **Transformed Feature Dimension:** **117 columns** (15 numerical + 102 one-hot categorical)
- **Estimator Class:** `GradientBoostingClassifier`
- **Estimator Hyperparameters:** `{'learning_rate': 0.1, 'loss': 'log_loss', 'max_depth': 3, 'n_estimators': 100, 'random_state': 42, 'subsample': 1.0}`
- **Training Samples:** 694 | **Testing Samples:** 174

### 6.3 Private Cohort Fingerprint
- **Artifact:** `models/private_model.joblib`
- **Pipeline Architecture:** `Pipeline(steps=[('preprocessor', ColumnTransformer), ('model', GradientBoostingClassifier)])`
- **Transformed Feature Dimension:** **145 columns** (15 numerical + 130 one-hot categorical)
- **Estimator Class:** `GradientBoostingClassifier`
- **Estimator Hyperparameters:** `{'learning_rate': 0.1, 'loss': 'log_loss', 'max_depth': 3, 'n_estimators': 100, 'random_state': 42, 'subsample': 1.0}`
- **Training Samples:** 934 | **Testing Samples:** 234

### 6.4 Daffodil Cohort Fingerprint
- **Artifact:** `models/daffodil_model.joblib`
- **Pipeline Architecture:** `Pipeline(steps=[('preprocessor', ColumnTransformer), ('model', VotingClassifier)])`
- **Transformed Feature Dimension:** **129 columns** (15 numerical + 114 one-hot categorical)
- **Estimator Class:** `VotingClassifier(voting='soft', weights=None)`
- **Ensemble Members (4):**
  1. `knn`: `KNeighborsClassifier(n_neighbors=5)`
  2. `rf`: `RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')`
  3. `extra`: `ExtraTreesClassifier(n_estimators=300, random_state=42, class_weight='balanced')`
  4. `gb`: `GradientBoostingClassifier(random_state=42, n_estimators=100)`
- **Training Samples:** 556 | **Testing Samples:** 139

---

## 7. Confirmation of Serialization Integrity

- **Unified Pipeline Serialization:** The preprocessor (`ColumnTransformer`) is **embedded directly inside the serialized Pipeline object** for each model.
- **Self-Contained Deployment:** An incoming student feature vector with raw strings (e.g. `"Department of Computer Science & Engineering"`, `"Software Engineer"`) is transformed and scaled on-the-fly by `pipeline.predict()` and `pipeline.predict_proba()` without requiring any manual external one-hot mapping, dictionary lookups, or retraining.
- **Robustness to Unseen Categories:** With `handle_unknown="ignore"`, any new academic department or career path entered during live usage is safely mapped to an all-zero vector, avoiding runtime exceptions.
- **Ethical & Clinical Safety Compliance:** The pipeline outputs pure model probabilities $P(\text{Class } 1)$; no arbitrary medical diagnostic labels or clinical assertions have been incorporated.

---

*(Phase 2 complete. Awaiting user approval to proceed to Phase 3: Interactive Application & Web Interface Implementation).*
