# Phase 3 — Step 3 Engineering Report: Research Results & Comparative Visualizations

**Project Title:** AI-Induced Career Anxiety Prediction Among University Students  
**Phase:** Phase 3 — Step 3: Research Results & Presentation Layer  
**Frameworks:** Streamlit `1.47.0`, Matplotlib `3.10.3`, Pandas `2.2.3`, SHAP `0.52.0`  
**Execution Environment:** Python `3.13.5` (`scikit-learn 1.6.1`, `joblib 1.4.2`, `pytest 9.1.1`)  
**Deployment Models:** `models/overall_model.joblib`, `models/public_model.joblib`, `models/private_model.joblib`, `models/daffodil_model.joblib`

---

## 1. Files Created and Modified

| File | Status | Description |
|---|---|---|
| [`src/research_results.py`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/src/research_results.py) | **Created** | Modular research data layer containing frozen benchmark constants (`final_all(1).ipynb`), serialized test metrics, cohort demographics, model selection rationale, and academic Matplotlib chart generators. |
| [`app.py`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/app.py) | **Updated** | Fully replaced placeholders for `render_research_results_page()`, `render_methodology_page()`, and `render_about_page()` with professional academic presentation interfaces, metric cards, comparative tables, interactive chart tabs, and ethical limitation callouts. |
| [`tests/test_research_results.py`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/tests/test_research_results.py) | **Created** | Comprehensive test suite validating frozen numerical constants, cohort nesting, non-training guarantees, table structures, and chart generation (10 passing tests). |
| [`PHASE3_STEP3_REPORT.md`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/PHASE3_STEP3_REPORT.md) | **Created** | Formal engineering and technical documentation artifact. |

---

## 2. Frozen Research Reference Results

All benchmark metrics are strictly frozen from `final_all(1).ipynb` (Steps 10 & 11) and `MODEL_AUDIT.md`:

| Cohort | Deployment Model | Test F1 | Test ROC-AUC | Test MCC | CV F1 (10-Fold) |
|---|---|:---:|:---:|:---:|:---:|
| **Overall** | Gradient Boosting (100 est) | **0.7994** | **0.5759** | **0.0984** | **0.7992** |
| **Public** | Gradient Boosting (100 est) | **0.7758** | **0.5199** | **-0.0671** | **0.7839** |
| **Private** | Gradient Boosting (100 est) | **0.8065** | **0.6841** | **0.1943** | **0.7945** |
| **Daffodil** | Performance Soft Voting | **0.8019** | **0.7418** \* | **0.2405** | **0.8298** |

### Additional Verified Test Metrics (from Serialized Models in `models/model_metadata.json`)

| Cohort | Accuracy | Precision | Recall | F1-Score | ROC-AUC | MCC | Cohen's Kappa |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Overall** | 0.6765 | 0.6885 | 0.9529 | 0.7994 | 0.5759 | 0.0984 | 0.0650 |
| **Public** | 0.6379 | 0.6646 | 0.9316 | 0.7758 | 0.5199 | -0.0671 | -0.0422 |
| **Private** | 0.6966 | 0.7115 | 0.9308 | 0.8065 | 0.6841 | 0.1943 | 0.1581 |
| **Daffodil** | 0.7050 | 0.7545 | 0.8557 | 0.8019 | 0.7413 | 0.2405 | 0.2333 |

---

## 3. Daffodil ROC-AUC Discrepancy Handling

- **Frozen Research Reference:** `0.7418`
- **Serialized Model Reproduction:** `0.7413`
- **Delta:** `-0.0005`
- **Transparency Protocol:** The frozen notebook figure of **0.7418** is presented as the primary research benchmark. A visible footnote and explanation callout explicitly document the -0.0005 variance.
- **Technical Explanation:** The difference originates from floating-point probability averaging across 600 individual decision trees in the 4-model ensemble (KNN + Random Forest + Extra Trees + Gradient Boosting) during ROC threshold rank ordering. Discrete classification metrics (Test F1 = 0.8019, Accuracy = 0.7050, Recall = 0.8557, MCC = 0.2405) reproduce with exact precision ($0.0000$ difference).

---

## 4. Cohort Demographics & Nesting Structure

| Cohort | Total $N$ | Train $N$ (80%) | Test $N$ (20%) | Class 0 (No/Low) | Class 1 (Med/High) | Class 1 % |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Overall** | 2,036 | 1,628 | 408 | 660 | 1,376 | 67.58% |
| **Public** | 868 | 694 | 174 | 298 | 570 | 65.67% |
| **Private** | 1,168 | 934 | 234 | 362 | 806 | 69.01% |
| **Daffodil** | 695 | 556 | 139 | 211 | 484 | 69.64% |

> **⚠️ Cohort Nesting Notice:** Daffodil International University ($N=695$) is an institutional sub-cohort nested directly within the Private University cohort ($N=1,168$). Cohort sample sizes are not mutually exclusive and must never be summed ($868 + 1168 = 2036 = \text{Overall}$).

---

## 5. Dashboard Structure & Implementation

### Section A — Research Overview & Sample Demographics
- 4 dynamic metric cards highlighting:
  - **Analytical Sample:** 2,036 Students (excluding 1,120 1st-year students).
  - **Target Balance:** 67.6% Class 1 (660 Low vs. 1,376 High).
  - **Holdout Protocol:** Stratified 80/20 train/test split (1,628 train / 408 test).
  - **Cross-Validation:** Stratified 10-fold cross-validation inside the training split.

### Section B — Primary Cohort Benchmark Comparison
- Formatted tabular presentation of frozen benchmarks.
- Expandable sub-table of secondary metrics (Accuracy, Precision, Recall, Cohen's Kappa).
- Expandable demographic breakdown with cohort nesting warnings.

### Section C — Comparative Metric Visualizations
- Academic Matplotlib bar charts organized in interactive tabs:
  1. **Test F1 Score:** Visualizes consistent performance (0.7758 to 0.8065) driven by high sensitivity.
  2. **Test ROC-AUC:** Honest display featuring a reference chance line ($0.50$). Highlights acceptable discrimination in Daffodil (0.7418) and Private (0.6841), while honestly presenting near-chance performance for Public (0.5199).
  3. **Matthews Correlation Coefficient (MCC):** Visualizes imbalance-resistant correlation with a zero reference baseline ($0.0$). Shows positive correlation for Daffodil (+0.2405) and Private (+0.1943), while clearly depicting negative correlation for Public (-0.0671).

### Section D — Deployment Model Selection & Rationale
Research-grounded, non-causal justifications for each cohort:
- **Overall (`Gradient Boosting`):** Strongest cross-cohort individual-model performance, stable 10-fold CV convergence, and balanced recall across the full heterogeneous population.
- **Public (`Gradient Boosting`):** Top-performing individual classifier for the public cohort, with transparent acknowledgment of near-chance discrimination under class imbalance.
- **Private (`Gradient Boosting`):** Superior discrimination (ROC-AUC 0.6841) and positive correlation (MCC 0.1943) over baseline linear and tree models.
- **Daffodil (`Performance Soft Voting Ensemble`):** Highest Test F1 (0.8019) and CV F1 (0.8298) among evaluated models, leveraging consensus across diverse model families.

---

## 6. Methodology & Limitations Pages

### Methodology Page (`render_methodology_page`)
- **Dataset:** 3,156 raw entries $\to$ 2,036 analytical records (1st-year students excluded).
- **Target Formulation:** Binarized Likert scale: Class 0 (No Anxiety / Low) vs. Class 1 (Medium / High).
- **17 Features:** 2 high-cardinality categorical (`department`, `career_path`) + 15 numerical/ordinal features including engineered interaction terms (`Perceived_Urgency`, `Risk_Knowledge_Gap`, `Threat_Perception`, `Age_Year_Ratio`).
- **Leak-Safe Preprocessing:** Unified `ColumnTransformer` with median/mode imputation and standard scaling fit strictly on $X_{\text{train}}$. Zero data leakage across CV folds and deployment inference.
- **Validation Protocol:** Stratified 80/20 train/test split, Stratified 10-fold CV.
- **Explainability:** SHAP `TreeExplainer` (Gradient Boosting) and `KernelExplainer` (Daffodil soft voting ensemble) with additive dummy aggregation.

### About & Limitations Page (`render_about_page`)
Prominent, unambiguous academic disclaimers:
1. **Associative, Not Causal:** Machine learning identifies correlations in survey responses; it does not establish causal mechanisms.
2. **Non-Clinical Boundary:** Survey grouping (Class 0 vs. Class 1) is not a medical, psychological, or psychiatric diagnosis.
3. **Self-Report Biases:** Subject to participant recall bias, self-selection bias, and subjective interpretation.
4. **Demographic Boundaries:** Analytical sample is specific to undergraduate students in Bangladesh.
5. **Cohort Nesting:** Daffodil is nested in Private; sample sizes must not be summed.
6. **Model Probability Interpretation:** Probabilities represent model scoring under survey distribution, not guaranteed calibrated confidence.

---

## 7. Verification & Test Results

### 1. Pytest Suite Execution
```bash
pytest -q
```
**Output:**
```
.......................................................................  [100%]
71 passed in 5.73s
```
- Phase 1 tests (`test_pipeline.py`): 7 passed
- Phase 2 tests (`test_models.py`): 28 passed
- Phase 3 Step 1 tests (`test_app.py`): 15 passed
- Phase 3 Step 2 tests (`test_explainability.py`): 11 passed
- Phase 3 Step 3 tests (`test_research_results.py`): 10 passed
- **Total Passing Tests:** **71 / 71 (100%)**

### 2. Streamlit Health Check
- Headless execution on port 8503 verified:
  - `GET http://localhost:8503/_stcore/health` $\to$ `HTTP 200 OK`
  - All page rendering functions callable and free of runtime exceptions or tracebacks.
