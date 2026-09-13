# Phase 3 — Step 4 Final Engineering Report: Integration, QA & Defense Readiness

**Project Title:** AI-Induced Career Anxiety Prediction Among University Students  
**Phase:** Phase 3 — Step 4: Final Integration, Quality Assurance & Defense Readiness  
**System Status:** Complete, Verified, and Frozen  
**Execution Environment:** Python `3.13.5` (`scikit-learn 1.6.1`, `pandas 2.2.3`, `shap 0.52.0`, `streamlit 1.47.0`, `matplotlib 3.10.3`, `pytest 9.1.1`)  
**Deployment Models:** `models/overall_model.joblib`, `models/public_model.joblib`, `models/private_model.joblib`, `models/daffodil_model.joblib`

---

## 1. Executive Summary & Project Structure

All development objectives across Phase 1 (Pipeline Extraction), Phase 2 (Model Serialization & Audit), and Phase 3 (Streamlit Foundation, SHAP Explainability, Research Dashboard, and Final QA Integration) are **100% complete**.

The repository structure is structured as follows:

```
AI-Career-Anxiety-System/
├── app.py                              # Streamlit live interactive web application
├── requirements.txt                    # Frozen dependency specifications
├── .gitignore                          # Clean ignore rules for Python, cache, and OS files
├── Career_Anxiety_due_to_AI.xlsx       # Master survey dataset (3,156 rows)
├── final_all(1).ipynb                  # Master research notebook (Steps 1–11)
├── PROJECT_ANALYSIS.md                 # Research audit and methodology mapping
├── MODEL_AUDIT.md                      # Audit of models from notebook Steps 6–11
├── PHASE2_REPORT.md                    # Model serialization and exactness report
├── PHASE3_STEP1_REPORT.md              # Application foundation report
├── PHASE3_STEP2_REPORT.md              # SHAP explainability integration report
├── PHASE3_STEP3_REPORT.md              # Research results dashboard report
├── PHASE3_STEP4_FINAL_REPORT.md        # This final engineering and QA report
├── DEFENSE_DEMO_GUIDE.md               # 3–5 min defense demo flow & 13 examiner Q&As
├── data/                               # Directory for data artifacts
├── models/
│   ├── overall_model.joblib            # Gradient Boosting (100 est)
│   ├── public_model.joblib             # Gradient Boosting (100 est)
│   ├── private_model.joblib            # Gradient Boosting (100 est)
│   ├── daffodil_model.joblib           # Performance Soft Voting (KNN+RF+ET+GB)
│   └── model_metadata.json             # Full evaluation metadata and metrics
├── src/
│   ├── __init__.py                     # Package initialization
│   ├── config.py                       # Frozen 17-feature space, mappings, universities
│   ├── feature_engineering.py          # Strict raw-to-engineered transformation
│   ├── cohort.py                       # University institutional cohort partitioning
│   ├── preprocessing.py                # Leak-safe ColumnTransformer definition
│   ├── model_training.py               # Deterministic training & serialization script
│   ├── explainability.py               # SHAP TreeExplainer & KernelExplainer engine
│   └── research_results.py             # Frozen benchmarks, demographics & chart utilities
└── tests/
    ├── __init__.py
    ├── test_pipeline.py                # Phase 1 tests (7 passed)
    ├── test_models.py                  # Phase 2 tests (28 passed)
    ├── test_app.py                     # Phase 3 Step 1 tests (15 passed)
    ├── test_explainability.py          # Phase 3 Step 2 tests (11 passed)
    ├── test_research_results.py        # Phase 3 Step 3 tests (10 passed)
    └── test_final_integration.py       # Phase 3 Step 4 final QA tests (12 passed)
```

---

## 2. Audit Wording Corrections Applied

During the Step 3 audit, two specific wording issues were identified and have been resolved:

1. **Correction 1 (Target Distribution Description):**
   - *Previous:* Described Class 1 counts as `"High"` alone or omitted explicit grouping in metric cards.
   - *Corrected:* Formally updated in `app.py` Section A metric card and demographics table to:
     `"67.6% Class 1 (Medium/High Anxiety): 660 Class 0 (No/Low) vs. 1,376 Class 1 (Medium/High)"`.
2. **Correction 2 (Overall Cohort Selection Rationale):**
   - *Previous:* Contained the wording `"balanced recall across the full heterogeneous population."`
   - *Corrected:* Replaced in `app.py` Section D and `src/research_results.py` with:
     `"strong overall Test F1 and stable 10-fold cross-validation performance across the full analytical cohort."`
3. **Cohort Selection UI Polish:**
   - Clearly separated automatic university mapping from manual overrides by labeling the selector:
     `"Research Model Override"` with dynamic status badges showing whether the active model is automatically mapped or manually overridden.

---

## 3. End-to-End Quality Assurance Results

| QA Area | Audit Criterion | Status | Verification Detail |
|---|---|:---:|---|
| **Prediction Flow** | Raw Dict $\to$ FE $\to$ Cohort $\to$ Pipeline $\to$ Predict | **PASS** | Verified across all four cohorts (`Overall`, `Public`, `Private`, `Daffodil`) using representative and boundary inputs. |
| **Probability Sanity** | $0 \le P(0) \le 1$, $0 \le P(1) \le 1$, $\sum P = 1.0$ | **PASS** | Evaluated via `test_probability_distribution_validity`. Outputs labeled strictly as `"Model-Estimated Probability P(Class 1)"`. |
| **Feature Space** | Exactly 17 features with verified column names | **PASS** | Verified via `test_feature_engineering_17_features`. Strictly matches `FINAL_FEATURES_17`. |
| **Explainability** | TreeExplainer for GB; KernelExplainer for Daffodil | **PASS** | Verified via `test_shap_explanations_across_cohorts`. Additive categorical aggregation yields 17 clean feature contributions. |
| **Cohort Mapping** | Automatic mapping based on university affiliation | **PASS** | Verified via `test_university_to_cohort_mapping`. DIU $\to$ Daffodil, AIUB $\to$ Private, DU/JU/CUET $\to$ Public, Other $\to$ Overall. |
| **Input Robustness** | Empty/multiple tools, all years, knowledge levels, timelines | **PASS** | Verified via `test_input_variation_robustness`. Zero crashes; graceful handling of boundary cases. |
| **OOD Categoricals** | Unseen departments and career paths | **PASS** | Verified via `test_unknown_categorical_handling`. `OneHotEncoder(handle_unknown="ignore")` ignores unknown levels safely. |
| **Zero-Leakage / Non-Training** | Model fitting never occurs at runtime | **PASS** | Verified via `test_zero_training_during_inference` with `Pipeline.fit` and `model.fit` mocked and asserted uncalled. |
| **Daffodil Discrepancy** | Reference $0.7418$ vs. reproduction $0.7413$ | **PASS** | Verified via `test_daffodil_roc_auc_discrepancy_preserved`. Both values and the $-0.0005$ delta are documented and displayed. |
| **Research Freeze** | Zero modifications to methodology, models, or notebook | **PASS** | Verified via `test_research_benchmark_constants_frozen`. Reference benchmarks match notebook to $0.0000$. |

---

## 4. Full Pytest Regression Results

```bash
pytest -q
```
**Output:**
```
........................................................................ [ 86%]
...........                                                              [100%]
83 passed in 6.01s
```

### Test Suite Distribution (83 Total Tests)
- `tests/test_pipeline.py`: **7 passed** (Data loading, 17-feature engineering, target mapping, leak-safe ColumnTransformer)
- `tests/test_models.py`: **28 passed** (Model training, serialization exactness, confusion matrix quadrants, F1/AUC/MCC benchmarks)
- `tests/test_app.py`: **15 passed** (Streamlit app foundation, session state, input validation, layout components)
- `tests/test_explainability.py`: **11 passed** (TreeExplainer, KernelExplainer, additive dummy aggregation, Shapley bounds)
- `tests/test_research_results.py`: **10 passed** (Benchmark constants, cohort nesting, secondary metrics, chart generation)
- `tests/test_final_integration.py`: **12 passed** (End-to-end integration, probability validity, OOD categories, non-training assertions)

---

## 5. Streamlit Server Health Check

Headless server verification was conducted on port 8504:
- Endpoint: `GET http://localhost:8504/_stcore/health` $\to$ **`HTTP 200 OK`**.
- Navigation verified across all five primary views:
  1. `Prediction`: Input forms, validation, model resolution, prediction execution, probability display.
  2. `Explainability`: Synchronized SHAP contribution table, horizontal bar chart, academic disclaimers.
  3. `Research Results`: Demographic metric cards, benchmark table, secondary metrics, comparative charts.
  4. `Methodology`: Dataset profile, 17-feature space breakdown, leak-safe protocol, validation design.
  5. `About & Limitations`: Institutional scope, self-report limitations, non-clinical ethics statements.
- Server terminated cleanly after verification; no hanging processes or leaked resources.

---

## 6. Documented Defense Demonstration Profile

To facilitate a live demonstration during defense, a representative synthetic profile was evaluated:

- **Student Profile:** 3rd Year Male, 22 years old, Department of Computer Science and Engineering at Daffodil International University.
- **AI Perceptions:** Medium AI Knowledge, beliefs AI will partially replace jobs within 6–10 years, uses ChatGPT and GitHub Copilot, perceives ChatGPT as a career threat, target career: Software Engineer.
- **Automatic Resolution:** Daffodil Cohort $\to$ `Performance Soft Voting Ensemble (KNN + RF + ExtraTrees + GB)`.
- **Inference Outcome:**
  - **Predicted Class:** `Class 1: Elevated Career Anxiety (Medium/High)`
  - **Model-Estimated Probability:** $P(\text{Class } 1) = \mathbf{0.8742}$
- **Top 5 SHAP Contributions (Marginal Probability Attributions):**
  1. *Estimated AI Takeover Timeline:* $+0.0690$ (Toward Class 1)
  2. *Aspired Career Path:* $+0.0688$ (Toward Class 1)
  3. *Perceived Urgency (Replacement × Timeline):* $+0.0320$ (Toward Class 1)
  4. *Specific AI Tool Threat Perception:* $+0.0167$ (Toward Class 1)
  5. *Age-to-Academic-Year Ratio:* $+0.0160$ (Toward Class 1)

*This walkthrough and 13 anticipated examiner Q&As are compiled in [`DEFENSE_DEMO_GUIDE.md`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/DEFENSE_DEMO_GUIDE.md).*

---

## 7. Security, Hygiene & Cleanup

1. **Path Neutrality:** Audited `app.py` and `src/*.py` for absolute user paths; confirmed all paths are dynamically resolved relative to `PROJECT_ROOT`.
2. **Localhost & URL Cleanliness:** Zero hardcoded localhost addresses or internal development URLs exist in active application code.
3. **Traceback Protection:** All user-facing submission blocks are wrapped in defensive `try/except` handlers that log errors and present clean academic advisories rather than exposing raw stack traces.
4. **Git Configuration:** `.gitignore` was created to exclude `__pycache__/`, `.pytest_cache/`, `.DS_Store`, and temporary files while preserving the dataset, notebook, model `.joblib` files, and documentation.

---

## 8. Final Defense Readiness Status

| Deliverable | Status |
|---|:---:|
| Core Prediction Engine | **DEFENSE READY** |
| SHAP Explainability Engine | **DEFENSE READY** |
| Research Results Dashboard | **DEFENSE READY** |
| Scientific Methodology Documentation | **DEFENSE READY** |
| About, Ethics & Limitations Layer | **DEFENSE READY** |
| Automated Test Suite (83 Tests Passing) | **DEFENSE READY** |
| Defense Demonstration Guide (`DEFENSE_DEMO_GUIDE.md`) | **DEFENSE READY** |

The system is completely verified, hardened, leak-safe, and ready for presentation to the final examination committee.
