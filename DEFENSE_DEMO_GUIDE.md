# Undergraduate Defense Demonstration Guide

**Project Title:** AI-Induced Career Anxiety Prediction Among University Students  
**System Version:** Phase 3 — Final Integration & Defense Readiness  
**Target Audience:** Final Year Project Examination Committee / Evaluators

---

## 1. Quick 3–5 Minute Live Demonstration Flow

Follow this structured walkthrough during the live project defense:

### Step 1: Launch Application & Overview (30 seconds)
1. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Explain the system's purpose:
   > *"This interactive research prototype demonstrates machine learning inference and SHAP explainability trained on primary survey data of undergraduate students in Bangladesh. It explores how student demographics, AI literacy, and perceived career disruption relate to self-reported career anxiety."*

---

### Step 2: Input a Representative Student Profile (45 seconds)
1. Navigate to the **Prediction** page.
2. Enter the standard demonstration profile:
   - **University:** `Daffodil International University` *(note automatic cohort mapping)*
   - **Department:** `Department of Computer Science and Engineering`
   - **Age:** `22`
   - **Gender:** `Male`
   - **Academic Year:** `3rd Year`
   - **AI Knowledge:** `Medium`
   - **Belief in Job Replacement:** `Partially`
   - **Takeover Timeline:** `6–10 years`
   - **Perspective:** `I think AI will replace some human jobs but also create new opportunities.`
   - **AI Tools Used:** Select `ChatGPT` and `Copilot`
   - **Career Path:** `Software Engineer`
   - **Perceived Threat Tool:** `ChatGPT`
3. Point out Section 5:
   - Note the **Automatic Cohort Assignment:** `Daffodil` mapped to `Performance Soft Voting Ensemble (KNN + RF + ExtraTrees + GB)`.
   - Explain that manual cohort override is available strictly for academic comparative analysis under the label **Research Model Override**.

---

### Step 3: Run Prediction & Interpret Results (45 seconds)
1. Click **"Predict Career Anxiety"**.
2. Explain the displayed output:
   - **Prediction:** `Class 1: Elevated Career Anxiety (Medium/High)`
   - **Model-Estimated Probability:** $P(\text{Class } 1) = 0.8742$
3. Clarify terminology:
   > *"Notice the wording: 'Model-Estimated Probability P(Class 1)', not 'confidence' or 'certainty'. The model indicates an 87.4% mathematical probability score under its survey-trained distribution. It is an associative ranking score, not a clinical diagnosis."*
4. Open the technical expander to show the **17-Feature Engineered Space** automatically created from the raw questionnaire inputs.

---

### Step 4: Examine SHAP Feature Explainability (60 seconds)
1. Navigate to the **Explainability** page.
2. Note that the explanation corresponds strictly to the latest prediction.
3. Review the **Top Feature Contributions Table**:
   - `Estimated AI Takeover Timeline`: $+0.0690$ *(Toward Class 1)*
   - `Aspired Career Path`: $+0.0688$ *(Toward Class 1)*
   - `Perceived Urgency (Replacement × Timeline)`: $+0.0320$ *(Toward Class 1)*
   - `Specific AI Tool Threat Perception`: $+0.0167$ *(Toward Class 1)*
   - `Age-to-Academic-Year Ratio`: $+0.0160$ *(Toward Class 1)*
4. Point out the horizontal bar chart:
   - Positive contributions (Coral red) shift output **toward Class 1** (Elevated Anxiety).
   - Negative contributions (Steel blue) shift output **away from Class 1** (Lower Anxiety).
5. Read the defense disclaimer:
   > *"SHAP values describe how features contributed to this model's prediction for this individual case. They do not establish causation and should not be interpreted as a clinical diagnosis."*

---

### Step 5: Research Results Dashboard (45 seconds)
1. Navigate to **Research Results**.
2. Highlight:
   - **Section A:** Analytical sample of 2,036 students (1,120 first-year students excluded), 80/20 stratified split, 10-fold CV.
   - **Section B Table:** Frozen benchmark results across Overall, Public, Private, and Daffodil cohorts.
   - **Daffodil Footnote:** Address the 0.0005 ROC-AUC discrepancy transparently ($0.7418$ reference vs. $0.7413$ reproduction).
   - **Section C Charts:** Toggle between F1, ROC-AUC (showing chance line at 0.50), and MCC (showing zero line at 0.0). Note the near-chance performance of Public (0.5199 ROC-AUC).

---

### Step 6: Methodology & Limitations (30 seconds)
1. Briefly show **Methodology**:
   - Leak-safe `ColumnTransformer` inside unified `Pipeline`. Preprocessing fit strictly on $X_{\text{train}}$. Zero data leakage.
2. Briefly show **About & Limitations**:
   - Reiterate non-causal, non-clinical research boundaries and nested Daffodil sample structure.

---

## 2. Documented Demonstration Scenario

| Step | Parameter / Feature | Value | Explanation |
|---|---|---|---|
| **Raw Input** | University | Daffodil International University | Auto-maps to Daffodil cohort |
| **Raw Input** | Department | Dept of Computer Science and Engineering | High-cardinality categorical |
| **Raw Input** | Age | 22 | Chronological age |
| **Raw Input** | Gender | Male | Encoded as 0 |
| **Raw Input** | Academic Year | 3rd Year | Encoded as 3 |
| **Raw Input** | AI Knowledge | Medium | Encoded as 2 |
| **Raw Input** | AI Replace Jobs | Partially | Encoded as 1 |
| **Raw Input** | AI Takeover Time | 6–10 years | Encoded as 4 |
| **Raw Input** | Future Perspective | Replace some & create new | Encoded as 2 |
| **Raw Input** | AI Tools Used | ChatGPT, GitHub Copilot | Encoded as 2 tools (Text + Coding) |
| **Raw Input** | Career Path | Software Engineer | High-cardinality categorical |
| **Raw Input** | Threat Perception | ChatGPT | Non-empty specific threat (mapped to 1) |
| **Engineered** | `Total_AI_Tools` | 2 | Count of tools parsed |
| **Engineered** | `Uses_Text_Gen` | 1 | Regex match for text generators |
| **Engineered** | `Uses_Coding_AI` | 1 | Regex match for coding assistants |
| **Engineered** | `Uses_Creative_AI` | 0 | No image/creative tools selected |
| **Engineered** | `Threat_Perception` | 1 | Valid specified career threat tool |
| **Engineered** | `Perceived_Urgency` | 4 | $(6 - 4) \times 2 = 4$ |
| **Engineered** | `Risk_Knowledge_Gap` | 0 | $(1 \times 2) - 2 = 0$ |
| **Engineered** | `Age_Year_Ratio` | 7.3333 | $22 / 3$ |
| **Output** | Selected Model | Performance Soft Voting | Multi-model ensemble consensus |
| **Output** | Predicted Class | **Class 1 (Elevated Anxiety)** | Medium / High Anxiety group |
| **Output** | Estimated Probability | **0.8742** | $P(\text{Class } 1)$ |
| **SHAP** | Top 1 Contribution | AI Takeover Timeline (+0.0690) | Contributes toward Class 1 |
| **SHAP** | Top 2 Contribution | Aspired Career Path (+0.0688) | Contributes toward Class 1 |
| **SHAP** | Top 3 Contribution | Perceived Urgency (+0.0320) | Contributes toward Class 1 |

---

## 3. Anticipated Examiner Questions & Model Answers

### Q1: Why did you exclude first-year students from the analytical dataset?
> **Answer:** *"First-year university students ($N=1,120$) were excluded during data preprocessing because they have minimal academic immersion in their declared majors and lack exposure to industry hiring dynamics or professional career pathways. Including them would introduce noise into career anxiety modeling. The analytical cohort was restricted to 2nd, 3rd, and 4th-year students ($N=2,036$) who actively confront internship and graduate employment realities."*

---

### Q2: Why did you frame career anxiety as a binary classification problem rather than 4-class multi-class?
> **Answer:** *"The raw questionnaire recorded career anxiety on a 4-point Likert scale ('No Anxiety', 'Low', 'Medium', 'High'). As shown in exploratory data analysis, intermediate boundaries between 'No Anxiety' and 'Low', as well as between 'Medium' and 'High', had subjective respondent overlap and class imbalance. Binarizing into Class 0 (No Anxiety / Low; 32.4%) and Class 1 (Medium / High; 67.6%) aligns with actionable decision support—distinguishing students with elevated career anxiety requiring career counseling intervention from those without significant concern."*

---

### Q3: Why exactly 17 features? How were they derived?
> **Answer:** *"The 17-feature space represents the verified, frozen feature set established in the master research notebook (`final_all(1).ipynb`). It comprises 2 high-cardinality categorical features (`department` and `career_path`), 7 raw demographic and literacy indicators, 4 AI tool adoption categories extracted via tokenization and keyword mapping, and 4 domain-engineered interaction terms: `Threat_Perception`, `Perceived_Urgency`, `Risk_Knowledge_Gap`, and `Age_Year_Ratio`. Ablation experiments confirmed that this 17-feature representation retained optimal predictive performance over reduced subsets."*

---

### Q4: Why are different models deployed for different cohorts?
> **Answer:** *"The research evaluated multiple model families independently across institutional cohorts because student populations exhibit different levels of demographic homogeneity and feature relationships. For the Overall, Public, and Private cohorts, Gradient Boosting achieved the strongest cross-validation stability and generalization among individual classifiers. However, for the Daffodil cohort, a soft voting ensemble combining KNN, Random Forest, Extra Trees, and Gradient Boosting produced superior consensus, reaching a Test ROC-AUC of 0.7418 and CV F1 of 0.8298."*

---

### Q5: Why was Gradient Boosting chosen over simpler models like Logistic Regression?
> **Answer:** *"Gradient Boosting builds decision trees sequentially to correct residual errors from previous trees, enabling it to model complex non-linear feature interactions (such as the interaction between perceived takeover timeline and career path) without manual polynomial expansion. In baseline benchmark evaluations across 10 classifiers, Gradient Boosting consistently outperformed linear models across the major cohorts."*

---

### Q6: Why does Daffodil use an ensemble while Overall, Public, and Private use Gradient Boosting?
> **Answer:** *"Daffodil International University is a single institution with a relatively homogeneous student body (predominantly computing and business students). In the research notebook's ensemble experiments, soft probability voting across diverse model families (KNN, Random Forest, Extra Trees, and Gradient Boosting) reduced individual model variance and achieved the highest Test F1 (0.8019), CV F1 (0.8298), and ROC-AUC (0.7418), outperforming any single standalone classifier on that cohort."*

---

### Q7: What does a SHAP value represent in your application?
> **Answer:** *"SHAP (SHapley Additive exPlanations) is rooted in cooperative game theory. In our system, a SHAP value quantifies the marginal contribution of an individual feature to the difference between the model's prediction for that student and the baseline expected value. For tree models, it operates in margin log-odds; for the Daffodil ensemble, it operates on predicted probability $P(Y=1)$. We aggregate dummy variables back to the 17 verified features using the additive efficiency axiom."*

---

### Q8: Does a positive SHAP value prove that this feature causes the student's anxiety?
> **Answer:** *"No, absolutely not. SHAP values describe feature attributions within the mathematical model's decision function. Because our models are trained on observational, cross-sectional survey data, SHAP identifies statistical associations and predictive importance—it does not establish causality. We strictly state in our UI: 'SHAP values describe how features contributed to this model's prediction... they do not establish causation.'"*

---

### Q9: Why is the Public cohort model's ROC-AUC only 0.5199? Is the model failing?
> **Answer:** *"A Test ROC-AUC of 0.5199 indicates that for public university students, the model discriminates between Class 0 and Class 1 barely above random chance (0.50), with a negative MCC of -0.0671. This is an important empirical finding: it demonstrates that the survey features regarding AI tools and timelines do not adequately explain career anxiety variance among public university students, who may be more concerned with government civil service exams (BCS) or general macroeconomic factors than private-sector AI disruption. We report this result honestly without inflating or concealing it."*

---

### Q10: Why is the Daffodil ROC-AUC reported as 0.7418 in the notebook, but 0.7413 in the serialized model reproduction?
> **Answer:** *"The difference is exactly 0.0005. Both the notebook and serialized model use identical training configurations, seed 42, and the exact same 600 decision trees across KNN, Random Forest, Extra Trees, and Gradient Boosting. Discrete classification metrics (Test F1 = 0.8019, Accuracy = 0.7050, MCC = 0.2405) match to 0.0000 exactness. The 0.0005 variance in ROC-AUC occurs because soft-voting averages continuous floating-point probability arrays across four distinct scikit-learn estimators, producing micro-level floating-point differences during threshold rank ordering. We document both numbers transparently."*

---

### Q11: Is this system intended for clinical psychological diagnosis?
> **Answer:** *"No. This is strictly a research decision-support prototype. The target variable is an empirical survey grouping, not a clinically validated psychiatric diagnosis like DSM-5 or GAD-7 clinical assessment. It must never be used for medical triage, psychiatric diagnosis, or formal clinical evaluation."*

---

### Q12: Why do you avoid calling the model probability 'confidence'?
> **Answer:** *"In machine learning, `predict_proba()` outputs the model's mathematical class scoring based on its training distribution. Calling it 'confidence' or 'certainty' falsely implies Bayesian posterior certainty or calibrated real-world confidence. Because empirical models can be overconfident on imbalanced datasets, we strictly label the output as 'Model-Estimated Probability P(Class 1)'."*

---

### Q13: Why is Daffodil described as nested within Private?
> **Answer:** *"Daffodil International University (DIU) is a private university in Bangladesh. Therefore, all 695 Daffodil students are also included within the 1,168 Private University students. The Daffodil cohort was evaluated separately in the research to examine whether institutional homogeneity improves predictive performance, but it is not mutually exclusive from the Private cohort. Their sample sizes must not be summed ($868 + 1168 = 2036 = \text{Overall}$)."*
