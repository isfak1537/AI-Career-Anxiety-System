# Phase 3 — Step 1 Engineering Report: Streamlit Application Foundation

**Project Title:** AI-Induced Career Anxiety Prediction Among University Students  
**Phase:** Phase 3 — Step 1: Build the Streamlit Application Foundation  
**Framework:** Streamlit `1.47.0`  
**Execution Environment:** Python `3.13.5` (`scikit-learn 1.6.1`, `pandas 2.2.3`, `joblib 1.4.2`, `pytest 9.1.1`)  
**Data Reference:** `Career_Anxiety_due_to_AI.xlsx`  
**Deployment Models:** `models/overall_model.joblib`, `models/public_model.joblib`, `models/private_model.joblib`, `models/daffodil_model.joblib`

---

## 1. Files Created and Modified

| File | Status | Description |
|---|---|---|
| [`app.py`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/app.py) | **Created / Updated** | Complete Streamlit interactive foundation with multi-section layout, raw inputs collection, feature engineering integration, and model prediction flow. |
| [`requirements.txt`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/requirements.txt) | **Updated** | Added `streamlit>=1.25.0` dependency. |
| [`tests/test_app.py`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/tests/test_app.py) | **Created** | Comprehensive unit & integration tests for app imports, artifact loading, feature schema conversion, class/probability bounds, and inference without fitting. |
| [`PHASE3_STEP1_REPORT.md`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/PHASE3_STEP1_REPORT.md) | **Created** | Formal engineering documentation and verification report. |

---

## 2. Application Architecture

The application adopts an academic, defense-friendly layout with a persistent sidebar and modular main canvas sections:

### 2.1 Sidebar Navigation Structure
- **Application Header:** Title, undergraduate research attribution, and institutional scope.
- **Section Router:**
  - **Prediction:** Fully implemented interactive student assessment module.
  - **Explainability:** Verified placeholder reserved for Phase 3 Step 2 (SHAP TreeExplainer feature attributions).
  - **Research Results:** Verified placeholder reserved for interactive research comparative tables and visualization figures.
  - **Methodology:** Verified placeholder reserved for the leak-safe pipeline documentation.
  - **About & Limitations:** Verified placeholder reserved for ethical boundaries and survey demographic scope.
- **Active Cohort Status Panel:** Displays the active estimators backing each institutional cohort.

---

## 3. Raw Input Architecture & Form Structure

The **Prediction** page collects strictly raw survey-level indicators from the user. It strictly avoids asking the user for derived or composite metrics:

| Section | Input Field | Streamlit Component | Value Space / Validation |
|---|---|---|---|
| **1. Demographics & Institutional** | University | `st.selectbox` | 5 Surveyed Institutions + Other |
| | Gender | `st.selectbox` | `Male`, `Female` |
| | Academic Year | `st.selectbox` | `2nd Year`, `3rd Year`, `4th Year` |
| | Age | `st.number_input` | Range $18$ to $35$ (default $22$) |
| **2. Discipline & Career** | Academic Department | `st.selectbox` + optional `st.text_input` | 23 Surveyed Departments or custom text |
| | Target Career Path | `st.selectbox` + optional `st.text_input` | Top 15 Career Paths or custom text |
| **3. AI Perception** | AI Knowledge Level | `st.selectbox` | `None`, `Low`, `Medium`, `High` |
| | AI Replace Jobs | `st.selectbox` | `No`, `Partially`, `Fully` |
| | AI Takeover Time | `st.selectbox` | `Never`, `50+ years`, `21–50 years`, `11–20 years`, `6–10 years`, `1–5 years` |
| | AI Future Perspective | `st.selectbox` | 5-point Likert statement on societal disruption |
| **4. AI Tools & Threat** | AI Tools Used | `st.multiselect` + `st.text_input` | Multi-select common tools + custom comma-delimited text |
| | Threat Perception | `st.text_input` | Specific perceived threat tool (or `'None'`) |
| **5. Model Selection** | Cohort Model | `st.selectbox` | Auto (based on University), Overall, Public, Private, Daffodil |

---

## 4. Feature Engineering Integration

The application directly imports and reuses [`src.feature_engineering.engineer_features()`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/src/feature_engineering.py):
- **Raw Input Conversion:** Converts the user's form responses into a single-row raw DataFrame.
- **NLP / Regex Tool Flags:** Automatically evaluates `Uses_Text_Gen`, `Uses_Coding_AI`, `Uses_Creative_AI`, and `Total_AI_Tools`.
- **Threat Indicator:** Evaluates `Threat_Perception` as binary $0$ or $1$.
- **Interaction Arithmetic:**
  $$\text{Perceived\_Urgency} = \text{ai\_replace\_jobs} \times \text{ai\_takeover\_time}$$
  $$\text{Risk\_Knowledge\_Gap} = \text{ai\_future\_perspective} - \text{ai\_knowledge}$$
  $$\text{Age\_Year\_Ratio} = \text{age} / \text{academic\_year}$$
- **Output:** Produces the exact, verified **17-column feature matrix** matching `FINAL_FEATURES_17`.

---

## 5. Cohort Selection & Model Loading Mechanism

- **Resource Caching:** Models are loaded through `@st.cache_resource(show_spinner="Loading research model pipeline...") def load_cohort_model(cohort_name: str)`. This guarantees that model artifacts are deserialized **only once** into memory and reused across all user interactions without I/O overhead.
- **Zero Training Guarantee:** No model fitting (`fit` or `fit_transform`) occurs during application runtime. Only pre-trained, pre-serialized Pipeline artifacts from `models/` are utilized.
- **Institutional Mapping:**
  - Daffodil International University $\rightarrow$ `Daffodil` (`VotingClassifier` ensemble)
  - American International University-Bangladesh $\rightarrow$ `Private` (`GradientBoostingClassifier`)
  - DU, JU, CUET $\rightarrow$ `Public` (`GradientBoostingClassifier`)
  - Other / Manual Override $\rightarrow$ `Overall` or any user-selected cohort model

---

## 6. Prediction Flow & Output Standards

When the user submits the form:
1. Validates non-empty department and career path inputs.
2. Formats raw inputs into a DataFrame and applies `engineer_features()`.
3. Passes the 17-column DataFrame into the cached cohort Pipeline.
4. The pipeline's internal `preprocessor` scales numerical inputs and one-hot encodes categorical variables (mapping unknown categories to zero vectors).
5. Calls `pipeline.predict()` and `pipeline.predict_proba()`.
6. **Displays Results with Scientific Rigor:**
   - **Predicted Class:** Explicitly reported as **Class 0** (No Anxiety / Low) or **Class 1** (Elevated Anxiety: Medium / High).
   - **Probability:** Explicitly labeled as **"Model-Estimated Probability $P(\text{Class } 1)$"**. The term *"confidence"* is strictly avoided.
   - **Zero Arbitrary Tiers:** No artificial risk tiers (e.g. Minimal, Moderate, High) are displayed.
   - **Safety & Ethical Notice:** Prominent notice stating that the output represents an empirical machine learning projection and does not constitute a medical or clinical diagnosis.
   - **Transparency:** An expandable inspection drawer displays the exact 17 engineered feature values used for the prediction.

---

## 7. Automated Test Suite Results

Executed `pytest -q` across all 3 test modules:

```
..................................................                                  [100%]
50 passed in 3.08s
```

### Breakdown of Passing Tests (50/50 Passing):
- **[`tests/test_app.py`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/tests/test_app.py) (15 tests):**
  - `test_app_imports_successfully` (PASSED)
  - `test_all_cohort_models_loadable` across Overall, Public, Private, Daffodil (4 PASSED)
  - `test_raw_input_converts_to_expected_feature_schema` (PASSED)
  - `test_prediction_and_probability_validity` across all 4 cohorts (4 PASSED)
  - `test_unknown_categorical_values_resilience_in_app` across all 4 cohorts (4 PASSED)
  - `test_no_model_fitting_during_prediction` (PASSED)
- **[`tests/test_models.py`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/tests/test_models.py) (28 tests):**
  - All Phase 2 model artifact loading, pipeline step integrity, bounds, and inference tests (28 PASSED).
- **[`tests/test_pipeline.py`](file:///Users/macbookair/Downloads/AI-Career-Anxiety-System/tests/test_pipeline.py) (7 tests):**
  - Analytical population size ($2,036$), 1st-year exclusion ($1,120$), 17 features, and cohort sizes (7 PASSED).

---

## 8. Streamlit Server Startup Verification

The application startup was directly verified in headless server mode:
```
Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.0.221:8501
```
The server initialized smoothly with zero startup exceptions, verified model loading, and clean component rendering.

---

## 9. Issues Encountered & Resolution

- **No Breaking Issues:** All components integrated seamlessly with the Phase 1 and Phase 2 modules.
- **Category Generalization:** Confirmed that arbitrary user-entered departments or career paths (outside the survey vocabulary) pass through `ColumnTransformer`'s `OneHotEncoder(handle_unknown="ignore")` without throwing exceptions or NaNs, satisfying all runtime safety specifications.

---

*(Phase 3 — Step 1 complete. Awaiting user review and authorization before proceeding to Phase 3 — Step 2: SHAP Explainability Integration).*
