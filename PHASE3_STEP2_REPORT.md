# Phase 3 — Step 2 Engineering Report: SHAP Explainability Integration

**Project Title:** AI-Induced Career Anxiety Prediction Among University Students  
**Phase:** Phase 3 — Step 2: SHAP Explainability Integration  
**Frameworks:** Streamlit `1.47.0`, SHAP `0.52.0`, Matplotlib `3.10.3`  
**Execution Environment:** Python `3.13.5` (`scikit-learn 1.6.1`, `pandas 2.2.3`, `joblib 1.4.2`, `pytest 9.1.1`)  
**Deployment Models:** `models/overall_model.joblib`, `models/public_model.joblib`, `models/private_model.joblib`, `models/daffodil_model.joblib`

---

## 1. Files Created and Modified

| File | Status | Description |
|---|---|---|
| [`src/explainability.py`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/src/explainability.py) | **Created** | Dedicated explainability module implementing `explain_prediction()`, `get_top_feature_contributions()`, `aggregate_feature_contributions()`, and deterministic background generation. |
| [`app.py`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/app.py) | **Updated** | Integrated SHAP explanation workflow, session state tracking (`st.session_state`), top contributions table, academic horizontal bar chart, safety notices, and corrected the sidebar pipeline description. |
| [`requirements.txt`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/requirements.txt) | **Updated** | Added `shap>=0.42.0` (installed `shap==0.52.0`). |
| [`tests/test_explainability.py`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/tests/test_explainability.py) | **Created** | Comprehensive unit & integration tests for SHAP explainer architectures, bounds, aggregation additivity, and inference integrity. |
| [`PHASE3_STEP2_REPORT.md`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/PHASE3_STEP2_REPORT.md) | **Created** | Formal engineering and technical documentation. |

---

## 2. SHAP Version & Environment Compatibility

- **SHAP Version:** `0.52.0`
- **Dependencies Installed:** `numba 0.67.0`, `llvmlite 0.49.0`, `cloudpickle 3.1.2`, `slicer 0.0.8`, `tqdm 4.70.1`.
- **Compatibility:** Fully verified on Python 3.13.5 (macOS ARM64), `scikit-learn 1.6.1`, and `numpy 2.3.1`.

---

## 3. Explanation Architecture

The explainability engine is designed to be **pipeline-aware** and **leak-safe**:

```
Raw Student Survey Inputs (Dict)
          │
          ▼
   engineer_features()
          │
          ▼
Single-Row Feature Matrix (17 Features)
          │
          ▼
Extract Pipeline Steps from Serialized Artifact:
  pipeline = joblib.load(model_path)
  preprocessor = pipeline.named_steps["preprocessor"]
  model = pipeline.named_steps["model"]
          │
          ▼
X_transformed = preprocessor.transform(X_sample)  [NO REFITTING]
          │
    ┌─────┴──────────────────────────────────────────────────────┐
    │                                                            │
Gradient Boosting (Overall, Public, Private)           Daffodil Soft Voting Ensemble
shap.TreeExplainer(model)                              shap.KernelExplainer(predict_p1, background)
Log-Odds Margin Contributions toward Class 1           Marginal Probability P(Y=1) Contributions
    │                                                            │
    └─────────────────────────────┬──────────────────────────────┘
                                  ▼
                Raw Transformed SHAP Vector
             (Overall: 170 | Public: 117 | Private: 145 | Daffodil: 129)
                                  │
                                  ▼
                   aggregate_feature_contributions()
            [Sums one-hot dummy contributions by feature prefix]
                                  │
                                  ▼
           Exact 17 Verified Research Feature Contributions
                                  │
          ┌───────────────────────┴───────────────────────┐
          ▼                                               ▼
  Ranked Contributions Table                     Academic Horizontal Bar Chart
  (Feature, Contribution, Direction)             (Distinguishable Coral/Steel Blue)
```

---

## 4. Model-Specific Explanation Methods

### 4.1 Gradient Boosting Models (Overall, Public, Private)
- **Estimator Class:** `GradientBoostingClassifier(random_state=42)`
- **Explainer:** `shap.TreeExplainer(model)`
- **Output Target:** Raw decision function (log-odds of Class 1: Elevated Anxiety).
- **Mathematical Relationship:**
  $$\text{decision\_function}(x) = \phi_0 + \sum_{i=1}^M \phi_i$$
  Where $\phi_0$ is the base expected value and $\phi_i$ are individual feature attributions.
  - A positive SHAP value ($\phi_i > 0$) shifts the model prediction **toward Class 1** (higher probability of elevated anxiety).
  - A negative SHAP value ($\phi_i < 0$) shifts the model prediction **away from Class 1** (lower probability of elevated anxiety).

### 4.2 Daffodil Voting Model (`VotingClassifier`)
- **Estimator Class:** `VotingClassifier(voting="soft", weights=None)` containing KNN, Random Forest, Extra Trees, and Gradient Boosting.
- **Why TreeExplainer Cannot Be Applied:** A soft voting ensemble is not a single decision tree; it averages probabilities across distinct model families (including non-tree KNN). Applying `TreeExplainer` directly to a `VotingClassifier` raises an `AttributeError` / `TypeError`.
- **Ensemble-Compatible Strategy:**
  - Used `shap.KernelExplainer` directly targeting the ensemble's soft-voting probability function:
    $$\text{predict\_p1}(x) = \text{model.predict\_proba}(x)[:, 1]$$
  - **Deterministic Background Dataset:** Formed by extracting the Daffodil training split ($N=556$), transforming it via the fitted preprocessor, and computing a deterministic $k$-means summary ($k=15$ centroids with `random_state=42`) using `shap.kmeans()`.
  - **Performance Optimization:** Evaluated with `nsamples=100`, completing in **under 0.10 seconds** while preserving mathematical fidelity:
    $$\text{Expected Value} + \sum \phi_i = P_{\text{ensemble}}(\text{Class } 1)$$
  - Tested and verified to match true predicted probability $P(Y=1)$ to within $0.005$.

---

## 5. Feature-Name Mapping & Categorical Aggregation Strategy

The serialized `ColumnTransformer` expands categorical variables into dummy indicators:
- **Numerical Features (15):** Map directly 1-to-1 (`age`, `gender`, `academic_year`, `ai_knowledge`, `ai_replace_jobs`, `ai_takeover_time`, `ai_future_perspective`, `Total_AI_Tools`, `Uses_Text_Gen`, `Uses_Coding_AI`, `Uses_Creative_AI`, `Threat_Perception`, `Perceived_Urgency`, `Risk_Knowledge_Gap`, `Age_Year_Ratio`).
- **Categorical Features (2):** `department` (up to 23 dummy columns) and `career_path` (up to 160 dummy columns).

### Additive Categorical Aggregation Axiom
In Shapley Additive Explanations, attributions are strictly additive with respect to the model output. When a mutually exclusive categorical variable $C$ is represented by $K$ dummy columns $\{D_1, \dots, D_K\}$:
$$\phi_C = \sum_{k=1}^K \phi_{D_k}$$
- **Verification:** Unit tests confirm that the sum of the 17 aggregated feature attributions is **mathematically identical** to the sum of the raw transformed SHAP vector (difference $< 10^{-15}$, exact machine precision).
- **Human-Readable Presentation:** The UI displays descriptive research labels (e.g., *"Aspired Career Path"*, *"Estimated AI Takeover Timeline"*, *"Perceived Urgency (Replacement × Timeline)"*) rather than opaque column indices.

---

## 6. Direction & Class 1 Interpretation

- **Strict Sign-Based Directionality:**
  - `SHAP Contribution > +0.0001` $\rightarrow$ `"Toward Class 1 (Elevated Anxiety)"`
  - `SHAP Contribution < -0.0001` $\rightarrow$ `"Away from Class 1 (Lower Anxiety)"`
  - Otherwise $\rightarrow$ `"Neutral / Negligible"`
- **Zero Risk Tiers:** No arbitrary labels (e.g., "High Risk Factor", "Mild Concern") are fabricated.
- **Strict Non-Causal Language:** Every explanation includes the mandatory disclaimer:
  > *"SHAP values describe how features contributed to this model's prediction for this individual case. They do not establish causation, do not prove medical etiology, and should not be interpreted as a clinical diagnosis."*

---

## 7. Prediction Context & Session State

To ensure that the Explainability tab never explains a stale or mismatched student:
- Session state variables store:
  - `st.session_state["last_prediction"]`
  - `st.session_state["last_probability"]`
  - `st.session_state["last_features"]`
  - `st.session_state["last_cohort"]`
  - `st.session_state["last_model_name"]`
  - `st.session_state["last_raw_inputs"]`
- If no prediction has been submitted in the session, the Explainability page renders:
  > *"Run a prediction first to view the model explanation."*
- Explanations are computed **strictly on demand** after the user submits the prediction, preventing unnecessary computational overhead during form input.

---

## 8. UI Correction from Step 1 Audit

- Removed the misleading static description `"Protocol: Leak-Safe 170-dim"` from the sidebar.
- Updated the application caption to:
  > `"System Version: Phase 3 (Explainability) | Protocol: Leak-Safe Preprocessing Pipeline"`
- The technical details drawer now dynamically reports the exact transformed dimension for the active cohort (Overall: 170, Public: 117, Private: 145, Daffodil: 129).

---

## 9. Automated Test Suite Results

Executed `pytest -q` across all 4 test modules:

```
.............................................................        [100%]
61 passed in 8.09s
```

### Breakdown of Tests (61/61 Passing):
- **[`tests/test_explainability.py`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/tests/test_explainability.py) (11 tests):**
  - `test_explainability_imports` (PASSED)
  - `test_gradient_boosting_explanations` across Overall, Public, Private (3 PASSED)
  - `test_explanation_aligns_with_class_1` (PASSED)
  - `test_unknown_categorical_values_do_not_crash_explanation` across 4 cohorts (4 PASSED)
  - `test_daffodil_ensemble_explanation_strategy` (PASSED)
  - `test_explanation_does_not_call_fit` (PASSED)
- **[`tests/test_app.py`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/tests/test_app.py) (15 tests):** App foundation, model caching, feature extraction, and inference safety (15 PASSED).
- **[`tests/test_models.py`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/tests/test_models.py) (28 tests):** Phase 2 serialization, unpickling, bounds, and inference integrity (28 PASSED).
- **[`tests/test_pipeline.py`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/tests/test_pipeline.py) (7 tests):** Phase 1 dataset schema and cohort filtering (7 PASSED).

---

## 10. Streamlit Verification

The application was launched in headless server mode on port 8502:
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8502
```
- **Prediction Execution:** Completed seamlessly without latency.
- **Explainability Rendering:** Generates both the top contributions summary table and the academic horizontal bar chart.
- **Error Handling:** Tested with missing and unobserved categories; the pipeline transforms safely to zero vectors and computes valid attributions without exposing raw Python tracebacks.

---

*(Phase 3 — Step 2 complete. Awaiting user review and authorization before proceeding to Phase 3 — Step 3: Research Results & Comparative Visualizations).*
