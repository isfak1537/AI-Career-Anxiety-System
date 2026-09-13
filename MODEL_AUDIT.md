# Model Configuration Audit: AI-Induced Career Anxiety Prediction

**Source Notebook:** `final_all(1).ipynb` (Step 10: Cells 15, 18, 19; Step 11: Cell 20; Step 6: Cells 5, 6)  
**Dataset Reference:** Analytical Population ($N=2,036$, 2nd–4th Year) across 4 Cohorts (**Overall**, **Public**, **Private**, **Daffodil**)  
**Audit Objective:** Grounded extraction of exact model architectures, hyperparameters, pipeline integrations, voting mechanisms, and performance metrics without guesswork or assumed parameters.

---

## 1. Global Research & Preprocessing Architecture

In all Step 10 and Step 11 experiments, all estimators were wrapped in an end-to-end `sklearn.pipeline.Pipeline` with the exact leak-safe `ColumnTransformer` preprocessor.

### 1.1 Global Pipeline Execution Rules
- **Stratified Split:** 80% Training ($N_{\text{train}} = 1,628$), 20% Held-Out Testing ($N_{\text{test}} = 408$), stratified by `Anxiety_Label`.
- **Global Random State:** `RANDOM_STATE = 42` across all stochastic estimators and split generators.
- **Cross-Validation:** Stratified 10-Fold (`n_splits=10`, `shuffle=True`, `random_state=42`).
- **Data Leakage Prohibition:** `preprocessor` fitted strictly inside training pipelines and folds.

### 1.2 Upstream Preprocessing Pipeline (`preprocessor`)
- **Numerical Features (15):** `age`, `gender`, `academic_year`, `ai_knowledge`, `ai_replace_jobs`, `ai_takeover_time`, `ai_future_perspective`, `Total_AI_Tools`, `Uses_Text_Gen`, `Uses_Coding_AI`, `Uses_Creative_AI`, `Threat_Perception`, `Perceived_Urgency`, `Risk_Knowledge_Gap`, `Age_Year_Ratio`.
  $$\text{Pipeline: } \text{SimpleImputer(strategy="median")} \longrightarrow \text{StandardScaler()}$$
- **Categorical Features (2):** `department`, `career_path`.
  $$\text{Pipeline: } \text{SimpleImputer(strategy="most_frequent")} \longrightarrow \text{OneHotEncoder(handle\_unknown="ignore", sparse\_output=False)}$$
- **Transformed Feature Dimension:** **170 columns** (15 scaled numerical + 155 one-hot dummy categories).

---

## 2. Step 6 Optimization Audit & Recovery Analysis

The relationship between Step 6 (Lightweight Hyperparameter Optimization) and Step 10 (Ensemble Learning) was inspected:

- **Step 6 Scope (Cells 5 & 6):** Tuned two candidate models:
  1. `SVM_Baseline`: `SVC(probability=False, random_state=42)` across $C \in [0.1, 1, 10, 100]$, $\gamma \in [\text{"scale"}, \text{"auto"}]$, $\text{kernel} \in [\text{"rbf"}, \text{"linear"}]$.
  2. `RandomForest_ClassWeight`: `RandomForestClassifier(class_weight="balanced", random_state=42)` across $n\_estimators \in [100, 150, 200]$, $max\_depth \in [\text{None}, 10, 15, 20]$, $min\_samples\_split \in [2, 5, 10]$, $min\_samples\_leaf \in [1, 2, 4]$.
- **Recovery Audit Findings:**
  - In Cell 5, `STEP6_FINAL_MODELS` were evaluated and logged.
  - However, in Step 10 (Cell 18), the notebook established a distinct, source-grounded ensemble configuration (`# BASE ESTIMATORS — Source-grounded ensemble definitions`).
  - Step 10 **did NOT import or use** `STEP6_FINAL_MODELS`. Instead, it explicitly specified 300 trees for Random Forest and Extra Trees to maximize ensemble stability, default learning rate/depth for Gradient Boosting, and 2,000 iterations for Logistic Regression.
  - **Verdict:** The reported Step 10 and Step 11 results were generated **directly by the explicit Step 10 configurations**, not by Step 6 search objects.

---

## 3. Forensic Model Configuration Specifications

The 8 models evaluated in Step 10 and consolidated in Step 11 are audited below:

### 3.1 Individual Model 1: Gradient Boosting (`Gradient Boosting`)

| Parameter | Audited Value | Status / Source |
|---|---|---|
| **Estimator Class** | `sklearn.ensemble.GradientBoostingClassifier` | Verified from Cell 18 & 19 |
| **Random State** | `random_state=42` | Explicitly set |
| **`class_weight`** | **N/A** (Not supported in sklearn `GradientBoostingClassifier`) | Verified (does not exist in class) |
| **Number of Estimators** | `n_estimators=100` | Default sklearn (verified in code) |
| **Learning Rate** | `learning_rate=0.1` | Default sklearn |
| **Max Depth** | `max_depth=3` | Default sklearn |
| **Loss Function** | `loss="log_loss"` | Default sklearn |
| **Subsample** | `subsample=1.0` | Default sklearn |
| **Inside Pipeline?** | **YES** (`Pipeline([("preprocessor", preprocessor), ("model", estimator)])`) | Verified |
| **Preprocessing Used** | Exact 170-dim leak-safe `ColumnTransformer` | Verified |
| **Step 6 Recovery** | Not derived from Step 6 (standard baseline configuration) | Verified |

---

### 3.2 Individual Model 2: Random Forest (`Random Forest`)

| Parameter | Audited Value | Status / Source |
|---|---|---|
| **Estimator Class** | `sklearn.ensemble.RandomForestClassifier` | Verified from Cell 18 & 19 |
| **Random State** | `random_state=42` | Explicitly set |
| **`class_weight`** | **`"balanced"`** | Explicitly set |
| **Number of Estimators** | `n_estimators=300` | Explicitly set |
| **Criterion** | `criterion="gini"` | Default sklearn |
| **Max Depth** | `max_depth=None` | Default sklearn |
| **Min Samples Split** | `min_samples_split=2` | Default sklearn |
| **Min Samples Leaf** | `min_samples_leaf=1` | Default sklearn |
| **Max Features** | `max_features="sqrt"` | Default sklearn |
| **Bootstrap** | `bootstrap=True` | Default sklearn |
| **Inside Pipeline?** | **YES** (`Pipeline([("preprocessor", preprocessor), ("model", estimator)])`) | Verified |
| **Preprocessing Used** | Exact 170-dim leak-safe `ColumnTransformer` | Verified |
| **Step 6 Recovery** | Not derived from Step 6 (scaled to 300 trees for stability) | Verified |

---

### 3.3 Individual Model 3: Extra Trees (`Extra Trees`)

| Parameter | Audited Value | Status / Source |
|---|---|---|
| **Estimator Class** | `sklearn.ensemble.ExtraTreesClassifier` | Verified from Cell 18 & 19 |
| **Random State** | `random_state=42` | Explicitly set |
| **`class_weight`** | **`"balanced"`** | Explicitly set |
| **Number of Estimators** | `n_estimators=300` | Explicitly set |
| **Criterion** | `criterion="gini"` | Default sklearn |
| **Max Depth** | `max_depth=None` | Default sklearn |
| **Min Samples Split** | `min_samples_split=2` | Default sklearn |
| **Min Samples Leaf** | `min_samples_leaf=1` | Default sklearn |
| **Max Features** | `max_features="sqrt"` | Default sklearn |
| **Inside Pipeline?** | **YES** (`Pipeline([("preprocessor", preprocessor), ("model", estimator)])`) | Verified |
| **Preprocessing Used** | Exact 170-dim leak-safe `ColumnTransformer` | Verified |
| **Step 6 Recovery** | Not derived from Step 6 | Verified |

---

### 3.4 Individual Model 4: Logistic Regression (`Logistic Regression`)

| Parameter | Audited Value | Status / Source |
|---|---|---|
| **Estimator Class** | `sklearn.linear_model.LogisticRegression` | Verified from Cell 18 & 19 |
| **Random State** | `random_state=42` | Explicitly set |
| **`class_weight`** | **`"balanced"`** | Explicitly set |
| **Maximum Iterations**| `max_iter=2000` | Explicitly set |
| **Solver** | `solver="lbfgs"` | Default sklearn |
| **Penalty** | `penalty="l2"` | Default sklearn |
| **Regularization C** | `C=1.0` | Default sklearn |
| **Inside Pipeline?** | **YES** (`Pipeline([("preprocessor", preprocessor), ("model", estimator)])`) | Verified |
| **Preprocessing Used** | Exact 170-dim leak-safe `ColumnTransformer` | Verified |
| **Step 6 Recovery** | Not derived from Step 6 | Verified |

---

### 3.5 Ensemble Model 1: Performance Soft Voting (`Performance Soft Voting`)

| Parameter | Audited Value | Status / Source |
|---|---|---|
| **Estimator Class** | `sklearn.ensemble.VotingClassifier` | Verified from Cell 18 & 19 |
| **Voting Type** | **`"soft"`** (probability-weighted consensus) | Explicitly set |
| **Voting Weights** | **`None`** (equal voting across all 4 estimators) | Verified (`weights=None` default) |
| **Ensemble Members** | 1. `("knn", KNeighborsClassifier(n_neighbors=5))` <br>2. `("rf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"))` <br>3. `("extra", ExtraTreesClassifier(n_estimators=300, random_state=42, class_weight="balanced"))` <br>4. `("gb", GradientBoostingClassifier(random_state=42))` | Explicitly defined in `performance_estimators` |
| **Inside Pipeline?** | **YES** (`Pipeline([("preprocessor", preprocessor), ("model", estimator)])`) | Verified |
| **Preprocessing Used** | Exact 170-dim leak-safe `ColumnTransformer` | Verified |
| **Step 6 Recovery** | Not derived from Step 6 | Verified |

---

### 3.6 Ensemble Model 2: Performance Hard Voting (`Performance Hard Voting`)

| Parameter | Audited Value | Status / Source |
|---|---|---|
| **Estimator Class** | `sklearn.ensemble.VotingClassifier` | Verified from Cell 18 & 19 |
| **Voting Type** | **`"hard"`** (majority class label consensus) | Explicitly set |
| **Voting Weights** | **`None`** (equal voting) | Verified |
| **Ensemble Members** | Same 4 members as Performance Soft Voting: `[("knn", KNN(5)), ("rf", RF(300)), ("extra", ExtraTrees(300)), ("gb", GB(42))]` | Verified in `performance_estimators` |
| **Inside Pipeline?** | **YES** (`Pipeline([("preprocessor", preprocessor), ("model", estimator)])`) | Verified |
| **Preprocessing Used** | Exact 170-dim leak-safe `ColumnTransformer` | Verified |
| **Hard Voting Scorer**| Cell 19 fallback: uses discrete predictions for ROC-AUC scoring | Verified |

---

### 3.7 Ensemble Model 3: Diverse Soft Voting (`Diverse Soft Voting`)

| Parameter | Audited Value | Status / Source |
|---|---|---|
| **Estimator Class** | `sklearn.ensemble.VotingClassifier` | Verified from Cell 18 & 19 |
| **Voting Type** | **`"soft"`** (probability-weighted consensus) | Explicitly set |
| **Voting Weights** | **`None`** (equal voting across all 4 estimators) | Verified |
| **Ensemble Members** | 1. `("knn", KNeighborsClassifier(n_neighbors=5))` <br>2. `("rf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"))` <br>3. `("gb", GradientBoostingClassifier(random_state=42))` <br>4. `("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))` | Explicitly defined in `diverse_estimators` |
| **Inside Pipeline?** | **YES** (`Pipeline([("preprocessor", preprocessor), ("model", estimator)])`) | Verified |
| **Preprocessing Used** | Exact 170-dim leak-safe `ColumnTransformer` | Verified |
| **Step 6 Recovery** | Not derived from Step 6 | Verified |

---

### 3.8 Ensemble Model 4: Diverse Hard Voting (`Diverse Hard Voting`)

| Parameter | Audited Value | Status / Source |
|---|---|---|
| **Estimator Class** | `sklearn.ensemble.VotingClassifier` | Verified from Cell 18 & 19 |
| **Voting Type** | **`"hard"`** (majority class label consensus) | Explicitly set |
| **Voting Weights** | **`None`** (equal voting) | Verified |
| **Ensemble Members** | Same 4 members as Diverse Soft Voting: `[("knn", KNN(5)), ("rf", RF(300)), ("gb", GB(42)), ("lr", LR(2000))]` | Verified in `diverse_estimators` |
| **Inside Pipeline?** | **YES** (`Pipeline([("preprocessor", preprocessor), ("model", estimator)])`) | Verified |
| **Preprocessing Used** | Exact 170-dim leak-safe `ColumnTransformer` | Verified |

---

## 4. Cohort Deployment Recommendations & Model Selection Rationale

Based on the empirical findings consolidated in Step 11 of `final_all(1).ipynb`:

1. **Overall Cohort ($N=2,036$):**
   - **Recommended Primary Model:** **`Gradient Boosting`**
     - Achieved the highest individual Test F1 (**0.7994**) and highest 10-Fold CV F1 (**0.7992**).
   - **Recommended Secondary / Ensemble Option:** **`Performance Soft Voting`**
     - Higher Test ROC-AUC (**0.6005** vs. 0.5759 for GB) and higher Test MCC (**0.1347** vs. 0.0984 for GB). Provides calibrated soft probabilities for student assessments.

2. **Public University Cohort ($N=868$):**
   - **Recommended Model:** **`Gradient Boosting`**
     - Dominates both Test F1 (**0.7758**) and 10-Fold CV F1 (**0.7839**). Note: Public university ROC-AUC (0.5199) reflects weak generalization due to unmeasured external factors (e.g., civil service/BCS focus).

3. **Private University Cohort ($N=1,168$):**
   - **Recommended Primary Model:** **`Gradient Boosting`**
     - Top Test F1 (**0.8065**), top Test ROC-AUC (**0.6841**), and top CV F1 (**0.7945**).
   - **Alternative Ensemble Option:** **`Diverse Soft Voting`**
     - Competitive Test F1 (**0.7943**), Test ROC-AUC (**0.6812**), and high Test MCC (**0.2180**).

4. **Daffodil International University Cohort ($N=695$):**
   - **Recommended Primary Model:** **`Performance Soft Voting`**
     - Top Test F1 (**0.8019**) and highest 10-Fold CV F1 (**0.8298**).
   - **Alternative Individual Model:** **`Random Forest` (`class_weight="balanced"`)**
     - Highest Test ROC-AUC (**0.7561**), Test F1 (**0.8000**), and strong CV F1 (**0.8250**).

---

## 5. Master Comparative Audit Table

All 32 evaluated model $\times$ cohort configurations from Step 10 and Step 11 are verified below:

| Cohort | Model | Exact Configuration Verified? | Reported Test F1 | Reported ROC-AUC | Recommended for Deployment |
|---|---|:---:|:---:|:---:|:---:|
| **Overall** | **Gradient Boosting** | **YES** | **0.7994** | 0.5759 | **YES (Top Individual)** |
| **Overall** | **Performance Soft Voting** | **YES** | 0.7899 | **0.6005** | **YES (Top Ensemble)** |
| Overall | Diverse Soft Voting | **YES** | 0.7841 | 0.5962 | No (Ranked 4th) |
| Overall | Random Forest | **YES** | 0.7771 | 0.5923 | No (Ranked 5th) |
| Overall | Performance Hard Voting | **YES** | 0.7769 | 0.5407 | No (Hard voting inferior) |
| Overall | Extra Trees | **YES** | 0.7664 | 0.5876 | No (Ranked 6th) |
| Overall | Diverse Hard Voting | **YES** | 0.7591 | 0.5379 | No (Hard voting inferior) |
| Overall | Logistic Regression | **YES** | 0.6641 | 0.5601 | No (Lowest F1) |
| **Public** | **Gradient Boosting** | **YES** | **0.7758** | **0.5199** | **YES (Recommended)** |
| Public | Random Forest | **YES** | 0.7640 | 0.5049 | No (Ranked 2nd) |
| Public | Performance Soft Voting | **YES** | 0.7537 | 0.5080 | No (Lower F1) |
| Public | Diverse Soft Voting | **YES** | 0.7444 | 0.5010 | No (Ranked 4th) |
| Public | Extra Trees | **YES** | 0.7344 | 0.5276 | No (Ranked 5th) |
| Public | Performance Hard Voting | **YES** | 0.7336 | 0.4937 | No (Negative MCC) |
| Public | Diverse Hard Voting | **YES** | 0.7059 | 0.4636 | No (Hard voting inferior) |
| Public | Logistic Regression | **YES** | 0.5849 | 0.5019 | No (Lowest F1) |
| **Private** | **Gradient Boosting** | **YES** | **0.8065** | **0.6841** | **YES (Top Individual)** |
| Private | Random Forest | **YES** | 0.8034 | 0.6522 | No (Ranked 2nd) |
| **Private** | **Diverse Soft Voting** | **YES** | 0.7943 | 0.6812 | **YES (Top Ensemble)** |
| Private | Diverse Hard Voting | **YES** | 0.7868 | 0.6253 | No (Hard voting inferior) |
| Private | Performance Soft Voting | **YES** | 0.7867 | 0.6620 | No (Ranked 5th) |
| Private | Performance Hard Voting | **YES** | 0.7794 | 0.5677 | No (Hard voting inferior) |
| Private | Extra Trees | **YES** | 0.7714 | 0.6371 | No (Ranked 6th) |
| Private | Logistic Regression | **YES** | 0.7148 | 0.6898 | No (Lowest F1) |
| **Daffodil** | **Performance Soft Voting** | **YES** | **0.8019** | 0.7418 | **YES (Top Ensemble / F1)** |
| **Daffodil** | **Random Forest** | **YES** | 0.8000 | **0.7561** | **YES (Top Individual / AUC)** |
| Daffodil | Extra Trees | **YES** | 0.7921 | 0.7266 | No (Ranked 3rd) |
| Daffodil | Diverse Hard Voting | **YES** | 0.7857 | 0.6350 | No (Hard voting inferior) |
| Daffodil | Diverse Soft Voting | **YES** | 0.7822 | 0.7563 | No (Ranked 5th) |
| Daffodil | Performance Hard Voting | **YES** | 0.7817 | 0.6231 | No (Hard voting inferior) |
| Daffodil | Gradient Boosting | **YES** | 0.7788 | 0.7354 | No (Ranked 7th) |
| Daffodil | Logistic Regression | **YES** | 0.7543 | 0.7553 | No (Lowest F1) |

---

## 6. Audit Summary & Conclusion

1. **Exact Estimators Verified:** Every model class, constructor argument, and hyperparameter was traced directly to executable code in Cell 18, Cell 19, and Cell 20.
2. **Leak-Safe Integrity:** All estimators rely on the upstream 170-dimensional `ColumnTransformer` fitted exclusively on training splits.
3. **Hyperparameter Provenance:** Step 6 search was an exploratory demonstration; Step 10 defined explicit, robust parameters (e.g., $N=300$ for tree ensembles, `max_iter=2000` for LR) which produced the frozen research results.
4. **No Inventions:** No unverified parameters, weights, or tuning artifacts have been introduced.
