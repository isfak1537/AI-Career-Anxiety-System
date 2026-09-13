"""
AI-Induced Career Anxiety Prediction System
Streamlit Interactive Application Foundation & SHAP Explainability (Phase 3 - Step 2)

Final Year Undergraduate Research Project Prototype:
"AI-Induced Career Anxiety Prediction Among University Students"
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Ensure writable matplotlib config directory for serverless environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Expose ASGI application instance for Vercel Functions Python runtime
try:
    from api.index import app as app
except Exception:
    app = None

from src.config import (
    FINAL_FEATURES_17,
    TARGET_ACADEMIC_YEARS,
    PUBLIC_UNIVERSITIES,
    PRIVATE_UNIVERSITIES,
    DAFFODIL_UNIVERSITIES,
)
from src.feature_engineering import engineer_features, extract_features_and_target
from src.explainability import (
    explain_prediction,
    get_top_feature_contributions,
    FEATURE_DISPLAY_NAMES,
)
from src.research_results import (
    FROZEN_BENCHMARK_RESULTS,
    SERIALIZED_MODEL_METRICS,
    DAFFODIL_ROC_AUC_DISCREPANCY,
    COHORT_DEMOGRAPHICS,
    DATASET_OVERVIEW,
    get_comparison_table,
    get_secondary_metrics_table,
    create_metric_comparison_figure,
)

# =========================================================================
# MODEL REGISTRY CONFIGURATION
# =========================================================================
MODEL_PATHS = {
    "Overall": PROJECT_ROOT / "models" / "overall_model.joblib",
    "Public": PROJECT_ROOT / "models" / "public_model.joblib",
    "Private": PROJECT_ROOT / "models" / "private_model.joblib",
    "Daffodil": PROJECT_ROOT / "models" / "daffodil_model.joblib",
}

MODEL_NAMES = {
    "Overall": "Gradient Boosting (100 estimators)",
    "Public": "Gradient Boosting (100 estimators)",
    "Private": "Gradient Boosting (100 estimators)",
    "Daffodil": "Performance Soft Voting Ensemble (KNN + RF + ExtraTrees + GB)",
}

SURVEYED_UNIVERSITIES = [
    "Daffodil International University",
    "American International University-Bangladesh",
    "University of Dhaka",
    "Jahangirnagar University",
    "Chittagong University of Engineering and Technology",
    "Other / Non-Surveyed Institution",
]

SURVEYED_DEPARTMENTS = [
    "Department of Computer Science and Engineering",
    "Department of Software Engineering",
    "Department of Computer Science & Engineering",
    "Department of Pharmacy",
    "Department of Computer Science",
    "Department of English",
    "Department of Electrical & Electronic Engineering",
    "Department of Management",
    "Department of Business Administration",
    "Department of Business Economics",
    "Department of Nutrition & Food Engineering",
    "Department of Law",
    "Department of Economics",
    "Department of Civil Engineering",
    "Department of Textile Engineering",
    "Other Department",
]

TOP_CAREER_PATHS = [
    "Software Engineer",
    "Researcher",
    "Data Scientist",
    "Cyber Security Expert",
    "Pharmacist",
    "Web Developer",
    "Teacher",
    "Software Developer",
    "Businessman",
    "Chartered Accountant",
    "Banker",
    "Data Analyst",
    "UI/UX Designer",
    "AI Engineer",
    "Project Manager",
    "Other Career Path",
]

AI_FUTURE_PERSPECTIVE_OPTIONS = [
    "I strongly believe AI cannot take over human activities and will be a helpful tool for humans.",
    "I believe AI will mostly assist humans, with limited job replacement.",
    "I think AI will replace some human jobs but also create new opportunities.",
    "I believe AI will significantly replace human jobs and create major challenges.",
    "I strongly believe AI will completely take over human jobs and pose serious threat to humanity.",
]

COMMON_AI_TOOLS = [
    "ChatGPT",
    "Claude",
    "Gemini",
    "Quillbot",
    "Grammarly",
    "Copilot",
    "Cursor",
    "Replit",
    "Deepseek",
    "Midjourney",
    "DALL-E",
    "Stable Diffusion",
    "Canva",
    "Napkin",
    "Photomath",
]


# =========================================================================
# CACHED MODEL LOADER
# =========================================================================
@st.cache_resource(show_spinner="Loading research model pipeline...")
def load_cohort_model(cohort_name: str):
    """
    Load a pre-trained, leak-safe Pipeline from disk without retraining.
    Cached via Streamlit resource caching to prevent reloading on interaction.
    """
    if cohort_name not in MODEL_PATHS:
        raise ValueError(f"Unknown cohort '{cohort_name}'. Available: {list(MODEL_PATHS.keys())}")

    filepath = MODEL_PATHS[cohort_name]
    if not filepath.exists():
        raise FileNotFoundError(
            f"Model artifact not found for '{cohort_name}' at {filepath}. "
            f"Please run 'python src/model_training.py' to serialize deployment artifacts."
        )

    return joblib.load(filepath)


def resolve_cohort_from_university(university: str) -> str:
    """
    Map an institutional affiliation to the primary research cohort:
    - Daffodil -> 'Daffodil'
    - AIUB -> 'Private'
    - DU, JU, CUET -> 'Public'
    - Others -> 'Overall'
    """
    if university in DAFFODIL_UNIVERSITIES:
        return "Daffodil"
    elif university in PRIVATE_UNIVERSITIES:
        return "Private"
    elif university in PUBLIC_UNIVERSITIES:
        return "Public"
    return "Overall"


def process_raw_student_inputs(raw_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert raw input dictionary into a single-row DataFrame,
    apply exact feature engineering, and extract the 17 features.
    """
    raw_df = pd.DataFrame([raw_dict])
    engineered_df = engineer_features(raw_df)
    X, _ = extract_features_and_target(engineered_df, include_target=False)
    return X


def predict_career_anxiety(
    raw_inputs: Dict[str, Any], cohort_name: str
) -> Tuple[int, float, pd.DataFrame]:
    """
    Execute pipeline prediction on a single student record:
    Returns: (predicted_class, p_class_1, engineered_feature_row)
    """
    pipeline = load_cohort_model(cohort_name)
    X_sample = process_raw_student_inputs(raw_inputs)

    # Inference without fitting
    pred_class = int(pipeline.predict(X_sample)[0])
    probabilities = pipeline.predict_proba(X_sample)[0]
    p_class_1 = float(probabilities[1])

    return pred_class, p_class_1, X_sample


def plot_feature_contributions_chart(top_df: pd.DataFrame, cohort_title: str):
    """
    Render a clean, academic horizontal bar chart of top SHAP contributions.
    Positive (toward Class 1) and negative (away from Class 1) are visually distinct.
    """
    # Reverse so top-ranked feature appears at the top
    df_plot = top_df.copy().iloc[::-1]
    values = [float(v) for v in df_plot["SHAP_Contribution"]]
    feature_labels = df_plot["Feature"].tolist()

    fig, ax = plt.subplots(figsize=(9, max(4.5, len(feature_labels) * 0.42)))

    # Coral/Crimson for positive (toward Class 1), Steel Blue for negative (away from Class 1)
    colors = ["#d9534f" if v >= 0 else "#337ab7" for v in values]
    y_pos = np.arange(len(feature_labels))

    bars = ax.barh(y_pos, values, color=colors, height=0.6, alpha=0.9)

    ax.axvline(0, color="#444444", linestyle="--", linewidth=0.9, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_labels, fontsize=9.5)
    ax.set_xlabel("SHAP Contribution (Marginal effect on model prediction)", fontsize=9.5)
    ax.set_title(
        f"Top Feature Contributions (SHAP) — {cohort_title} Model",
        fontsize=10.5,
        fontweight="bold",
        pad=10,
    )

    # Value annotations on bars
    for bar, val in zip(bars, values):
        x_offset = 0.002 if val >= 0 else -0.002
        ha = "left" if val >= 0 else "right"
        ax.text(
            val + x_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}",
            va="center",
            ha=ha,
            fontsize=8.5,
            color="#222222",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    return fig


# =========================================================================
# UI PAGES & NAVIGATION
# =========================================================================
def render_prediction_page():
    st.markdown("### Student Career Anxiety Assessment")
    st.markdown(
        """
        Enter raw undergraduate student survey indicators below. The system automatically computes
        the exact **17-feature analytical space** and generates an empirical prediction via the 
        corresponding leak-safe cohort pipeline.
        """
    )

    with st.form("prediction_form"):
        # Section 1: Institutional & Demographic Profile
        st.subheader("1. Institutional & Demographic Profile")
        col1, col2 = st.columns(2)

        with col1:
            selected_university = st.selectbox(
                "University / Institution",
                options=SURVEYED_UNIVERSITIES,
                help="Select institutional affiliation to determine the appropriate research cohort model.",
            )
            gender = st.selectbox(
                "Gender",
                options=["Male", "Female"],
            )

        with col2:
            academic_year = st.selectbox(
                "Academic Year",
                options=TARGET_ACADEMIC_YEARS,
                index=1,
                help="Analytical population was restricted to 2nd, 3rd, and 4th year students (1st-year students were excluded in the research).",
            )
            age = st.number_input(
                "Chronological Age",
                min_value=18,
                max_value=35,
                value=22,
                step=1,
            )

        # Section 2: Academic Discipline & Career Aspirations
        st.subheader("2. Discipline & Target Career Path")
        col3, col4 = st.columns(2)

        with col3:
            dept_selection = st.selectbox(
                "Academic Department",
                options=SURVEYED_DEPARTMENTS,
            )
            if dept_selection == "Other Department":
                department = st.text_input("Enter Department Name", value="Department of General Studies").strip()
            else:
                department = dept_selection

        with col4:
            career_selection = st.selectbox(
                "Aspired Career Path / Professional Role",
                options=TOP_CAREER_PATHS,
            )
            if career_selection == "Other Career Path":
                career_path = st.text_input("Enter Target Career Path", value="General Professional").strip()
            else:
                career_path = career_selection

        # Section 3: AI Literacy & Perception
        st.subheader("3. AI Knowledge, Timeline & Job Replacement Beliefs")
        col5, col6 = st.columns(2)

        with col5:
            ai_knowledge = st.selectbox(
                "Self-Assessed AI Knowledge",
                options=["None", "Low", "Medium", "High"],
                index=2,
                help="Self-reported technical literacy regarding artificial intelligence systems.",
            )
            ai_replace_jobs = st.selectbox(
                "Do You Believe AI Will Replace Human Jobs?",
                options=["No", "Partially", "Fully"],
                index=1,
            )

        with col6:
            ai_takeover_time = st.selectbox(
                "Estimated Timeline for Substantial AI Takeover",
                options=["Never", "50+ years", "21–50 years", "11–20 years", "6–10 years", "1–5 years"],
                index=4,
            )

        ai_future_perspective = st.selectbox(
            "AI Future Perspective (Displacement vs. Opportunity)",
            options=AI_FUTURE_PERSPECTIVE_OPTIONS,
            index=2,
            help="Validated 5-point Likert statement on general societal and economic job impact.",
        )

        # Section 4: Tool Usage & Threat Perception
        st.subheader("4. AI Tool Usage & Perceived Threat")
        selected_tools = st.multiselect(
            "AI Tools Used (Select all that apply)",
            options=COMMON_AI_TOOLS,
            default=["ChatGPT", "Copilot"],
            help="Multi-select tools to extract total tool count and regex indicators.",
        )
        additional_tools = st.text_input(
            "Additional AI Tools (Comma-separated, optional)",
            placeholder="e.g., Perplexity, Midjourney",
        ).strip()

        # Combine tools
        all_tools_list = list(selected_tools)
        if additional_tools:
            extra = [t.strip() for t in additional_tools.split(",") if t.strip()]
            all_tools_list.extend(extra)
        ai_tools_used_str = ", ".join(all_tools_list) if all_tools_list else ""

        ai_tool_perception = st.text_input(
            "Specific AI Tool Perceived as Direct Threat to Your Career",
            value="ChatGPT",
            help="Name of specific AI tool perceived as a threat. Enter 'None' or 'I don't know for now' if none perceived.",
        ).strip()

        # Cohort Selection Options
        st.subheader("5. Evaluation Model Selection")
        auto_cohort = resolve_cohort_from_university(selected_university)
        st.info(f"**Automatic Cohort Assignment:** `{auto_cohort}` (mapped from university: *{selected_university}*)")
        override_cohort = st.selectbox(
            "Research Model Override",
            options=["Auto (Use Automatic University Mapping)", "Overall", "Public", "Private", "Daffodil"],
            index=0,
            help="By default, the system evaluates with the automatically assigned institutional cohort model. Select a specific cohort model only if you wish to apply a Research Model Override for academic comparison.",
        )

        is_override = override_cohort != "Auto (Use Automatic University Mapping)"
        cohort_to_use = override_cohort if is_override else auto_cohort

        mode_desc = "Research Model Override" if is_override else "Automatically Mapped"
        st.caption(
            f"**Resolved Cohort:** `{cohort_to_use}` ({mode_desc}) | "
            f"**Pipeline Model:** `{MODEL_NAMES.get(cohort_to_use, 'Custom')}`"
        )

        submitted = st.form_submit_button("Predict Career Anxiety", type="primary", use_container_width=True)

    # Execution upon submission
    if submitted:
        # 1. Validation
        if not department:
            st.warning("Please specify an academic department.")
            return
        if not career_path:
            st.warning("Please specify a target career path.")
            return

        raw_data = {
            "university": selected_university,
            "department": department,
            "age": int(age),
            "gender": gender,
            "academic_year": academic_year,
            "ai_knowledge": ai_knowledge,
            "ai_tools_used": ai_tools_used_str,
            "career_path": career_path,
            "ai_tool_perception": ai_tool_perception,
            "ai_future_perspective": ai_future_perspective,
            "ai_replace_jobs": ai_replace_jobs,
            "ai_takeover_time": ai_takeover_time,
        }

        try:
            with st.spinner("Processing feature engineering and executing model pipeline..."):
                pred_class, p_class_1, engineered_features = predict_career_anxiety(
                    raw_data, cohort_to_use
                )

            # Persist state for explainability and section routing
            st.session_state["last_prediction"] = pred_class
            st.session_state["last_probability"] = p_class_1
            st.session_state["last_features"] = engineered_features
            st.session_state["last_cohort"] = cohort_to_use
            st.session_state["last_model_name"] = MODEL_NAMES.get(cohort_to_use)
            st.session_state["last_raw_inputs"] = raw_data

            # 2. Render Results
            st.markdown("---")
            st.markdown("### Prediction Result")

            col_res1, col_res2, col_res3 = st.columns(3)

            with col_res1:
                if pred_class == 1:
                    st.error("#### Predicted Class: **Class 1**\n(Elevated Anxiety: Medium / High)")
                else:
                    st.success("#### Predicted Class: **Class 0**\n(No Anxiety / Low)")

            with col_res2:
                st.metric(
                    label="Model Probability P(Class 1)",
                    value=f"{p_class_1 * 100:.2f}%",
                    help="Model-estimated probability P(Y=1) of elevated career anxiety.",
                )

            with col_res3:
                st.info(
                    f"**Cohort Evaluated:** {cohort_to_use}\n\n"
                    f"**Estimator:** {MODEL_NAMES.get(cohort_to_use)}"
                )

            # 3. Model Interpretation Notice
            st.info(
                "**Notice on Interpretation:** This prediction represents an empirical machine learning projection "
                "from self-reported survey data. In accordance with project safety standards, it does not constitute "
                "a medical or clinical diagnosis."
            )

            # 4. Instant Explainability Preview
            st.markdown("---")
            st.markdown("### Model Feature Contribution (SHAP Preview)")
            pipeline = load_cohort_model(cohort_to_use)
            explanation = explain_prediction(pipeline, engineered_features, cohort_name=cohort_to_use)
            top_df = get_top_feature_contributions(explanation["aggregated_shap"], top_k=7)

            st.caption("These are the features that contributed most to this model's prediction for the selected student:")
            st.dataframe(top_df, use_container_width=True)
            st.caption("For full graphical attributions and methodology, visit the **Explainability** section in the sidebar.")

            # 5. Transparent Feature Breakdown
            with st.expander("Inspect Transformed 17-Feature Space"):
                st.dataframe(engineered_features, use_container_width=True)

        except FileNotFoundError as fnf_err:
            st.error(f"Configuration Error: {fnf_err}")
        except Exception as ex:
            st.error(
                f"An error occurred during prediction execution: {str(ex)}. "
                "Please check the inputs or consult developer logs."
            )


def render_explainability_page():
    st.markdown("### Model Explainability & Feature Attribution (SHAP)")
    st.markdown(
        """
        This section provides research-grounded local feature attributions using **SHAP (SHapley Additive exPlanations)**.
        It explains which input features contributed toward or away from the model's **Class 1 (Elevated Anxiety)** prediction.
        """
    )

    # Check prediction context
    if "last_features" not in st.session_state or "last_cohort" not in st.session_state:
        st.info("Run a prediction first to view the model explanation.")
        st.markdown(
            "Please navigate to the **Prediction** page in the sidebar, fill in the student survey inputs, and click "
            "**Predict Career Anxiety**."
        )
        return

    last_features = st.session_state["last_features"]
    last_cohort = st.session_state["last_cohort"]
    last_model_name = st.session_state.get("last_model_name", last_cohort)
    last_pred = st.session_state.get("last_prediction", 0)
    last_prob = st.session_state.get("last_probability", 0.0)

    # Header summary metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Predicted Class", f"Class {last_pred} ({'Elevated' if last_pred==1 else 'Low/None'})")
    with col_m2:
        st.metric("Model Probability P(Class 1)", f"{last_prob * 100:.2f}%")
    with col_m3:
        st.metric("Active Model", f"{last_cohort} ({last_model_name.split('(')[0].strip()})")

    st.markdown("---")

    try:
        with st.spinner("Computing SHAP feature attributions..."):
            pipeline = load_cohort_model(last_cohort)
            explanation = explain_prediction(pipeline, last_features, cohort_name=last_cohort)
            top_df = get_top_feature_contributions(explanation["aggregated_shap"], top_k=10)

        # A. Explanation Summary
        st.subheader("Explanation Summary")
        st.markdown(
            "**These are the features that contributed most to this model's prediction for the selected student.**"
        )
        st.caption(
            f"**Explainer Architecture:** `{explanation['explainer_type']}` | "
            f"**Transformed Feature Space:** `{explanation['transformed_dimension']} columns` aggregated to 17 research features."
        )

        # B. Top Contributions Table
        st.markdown("#### Top Feature Contributions Table")
        st.dataframe(top_df, use_container_width=True)

        # C. Visual Explanation Chart
        st.markdown("#### Visual Feature Attribution Chart")
        fig = plot_feature_contributions_chart(top_df, cohort_title=last_cohort)
        st.pyplot(fig)

        # D. Important Disclaimer
        st.warning(
            "**Important Notice:** SHAP values describe how features contributed to this model's prediction for this "
            "individual case. They do not establish causation, do not prove medical etiology, and should not be "
            "interpreted as a clinical diagnosis."
        )

        # Technical expansion for defense committee
        with st.expander("Technical Details: Additive Categorical Aggregation"):
            st.markdown(
                """
                - **TreeExplainer vs. KernelExplainer:**
                  - For Gradient Boosting (Overall, Public, Private), exact tree path traversal calculates marginal log-odds contributions.
                  - For the Daffodil Soft Voting ensemble, model-agnostic kernel estimation calculates marginal contributions to ensemble probability $P(Y=1)$ using a deterministic 15-centroid background reference.
                - **Categorical Aggregation:**
                  - Because `department` and `career_path` are one-hot encoded, individual dummy variable contributions are summed by prefix:
                    $$\\phi_{\\text{Department}} = \\sum_{j \\in \\text{dept dummies}} \\phi_j, \\quad \\phi_{\\text{Career Path}} = \\sum_{k \\in \\text{career dummies}} \\phi_k$$
                  - By the additive efficiency axiom of Shapley values, this sum represents the exact net marginal contribution of the categorical feature.
                """
            )
            st.json({
                "cohort": last_cohort,
                "explainer": explanation["explainer_type"],
                "base_expected_value": round(explanation["expected_value"], 4),
                "aggregated_17_features": {k: round(v, 4) for k, v in explanation["aggregated_shap"].items()},
            })

    except Exception as e:
        st.error(f"Could not compute explanation: {str(e)}")
        st.caption("Detailed error logged. The core prediction engine remains active.")


def render_research_results_page():
    st.markdown("## 📊 Empirical Research Results Dashboard")
    st.caption("Frozen benchmark results from master research notebook `final_all(1).ipynb` and verified deployment evaluations")

    # =========================================================================
    # SECTION A: RESEARCH OVERVIEW
    # =========================================================================
    st.markdown("### Section A — Research Overview & Sample Demographics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Analytical Sample",
            value="2,036 Students",
            help="Filtered from 3,156 raw responses; excludes 1,120 1st-year students lacking career immersion.",
        )
    with col2:
        st.metric(
            label="Target Balance",
            value="67.6% Class 1 (Medium/High Anxiety)",
            delta="660 Class 0 (No/Low) vs. 1,376 Class 1 (Medium/High)",
            delta_color="off",
            help="Binary grouping: 660 Class 0 (No/Low) vs. 1,376 Class 1 (Medium/High Anxiety).",
        )
    with col3:
        st.metric(
            label="Holdout Protocol",
            value="Stratified 80 / 20",
            delta="1,628 Train / 408 Test",
            delta_color="off",
            help="Fixed random_state=42 with stratification preserving class ratio.",
        )
    with col4:
        st.metric(
            label="Cross-Validation",
            value="10-Fold Stratified",
            help="Evaluated inside training split to prevent evaluation leakage.",
        )

    st.markdown("---")

    # =========================================================================
    # SECTION B: COHORT BENCHMARK COMPARISON
    # =========================================================================
    st.markdown("### Section B — Primary Benchmark Results by Cohort")
    st.markdown(
        "Below are the frozen benchmark results from the research reference (`final_all(1).ipynb` Steps 10–11) "
        "across the four targeted student cohorts:"
    )

    df_bench = get_comparison_table()
    st.dataframe(df_bench, use_container_width=True, hide_index=True)

    # Daffodil reproduction footnote
    st.info(
        f"**\\* Note on Daffodil ROC-AUC Discrepancy:** The frozen research reference value is "
        f"**{DAFFODIL_ROC_AUC_DISCREPANCY['frozen_reference']:.4f}**, whereas the local serialized reproduction "
        f"yielded **{DAFFODIL_ROC_AUC_DISCREPANCY['serialized_reproduction']:.4f}** "
        f"(difference = {DAFFODIL_ROC_AUC_DISCREPANCY['difference']:+.4f}). "
        f"{DAFFODIL_ROC_AUC_DISCREPANCY['explanation']}"
    )

    # Secondary metrics expander
    with st.expander("View Additional Verified Test Metrics (Accuracy, Precision, Recall, Cohen's Kappa)"):
        st.markdown(
            "These metrics were evaluated directly on the holdout test set using the verified serialized models "
            "(`models/model_metadata.json`):"
        )
        df_sec = get_secondary_metrics_table()
        st.dataframe(df_sec, use_container_width=True, hide_index=True)

    # Cohort demographics & nesting structure
    with st.expander("Cohort Sample Demographics & Nesting Structure"):
        st.markdown(
            """
            | Cohort | Total $N$ | Train $N$ (80%) | Test $N$ (20%) | Class 0 (No/Low) | Class 1 (Medium/High) | Institutional Scope |
            |---|---|---|---|---|---|---|
            | **Overall** | 2,036 | 1,628 | 408 | 660 (32.4%) | 1,376 (67.6%) | Full sample across public & private institutions |
            | **Public** | 868 | 694 | 174 | 298 (34.3%) | 570 (65.7%) | Public universities (DU, JU, CUET, etc.) |
            | **Private** | 1,168 | 934 | 234 | 362 (31.0%) | 806 (69.0%) | Private universities (DIU, AIUB, etc.) |
            | **Daffodil** | 695 | 556 | 139 | 211 (30.4%) | 484 (69.6%) | Daffodil International University *(nested in Private)* |
            """
        )
        st.warning(
            "⚠️ **Cohort Nesting Notice:** Daffodil International University ($N = 695$) is an institutional sub-cohort "
            "nested directly within the Private University cohort ($N = 1,168$). Do NOT sum cohort sample sizes as if they "
            "were mutually exclusive categories."
        )

    st.markdown("---")

    # =========================================================================
    # SECTION C: METRIC COMPARISON VISUALIZATIONS
    # =========================================================================
    st.markdown("### Section C — Metric Comparison Across Cohorts")
    st.caption("Visualizing model discrimination, precision-recall balance, and correlation under class imbalance")

    viz_tab1, viz_tab2, viz_tab3 = st.tabs([
        "📈 Test F1 Score",
        "🎯 Test ROC-AUC",
        "⚖️ Matthews Correlation (MCC)",
    ])

    with viz_tab1:
        fig_f1, _ = create_metric_comparison_figure("f1")
        st.pyplot(fig_f1, use_container_width=True)
        st.markdown(
            "**Observation:** F1 scores remain consistently strong across all cohorts (0.7758 to 0.8065). "
            "Because Class 1 represents ~67% of students, high sensitivity (recall > 0.93 for Gradient Boosting models) "
            "elevates the harmonic mean."
        )

    with viz_tab2:
        fig_roc, _ = create_metric_comparison_figure("roc_auc")
        st.pyplot(fig_roc, use_container_width=True)
        st.markdown(
            "**Observation:** ROC-AUC demonstrates meaningful cohort differentiation. Daffodil (0.7418) and "
            "Private (0.6841) achieve positive discriminative power. Conversely, the **Public cohort (0.5199)** "
            "operates near the random chance baseline (0.50), demonstrating that survey features provide minimal "
            "discriminative separation between classes for public university students."
        )

    with viz_tab3:
        fig_mcc, _ = create_metric_comparison_figure("mcc")
        st.pyplot(fig_mcc, use_container_width=True)
        st.markdown(
            "**Observation:** Matthews Correlation Coefficient (MCC) is especially sensitive to class imbalance. "
            "The Daffodil ensemble achieves the highest positive correlation (+0.2405), followed by Private (+0.1943) "
            "and Overall (+0.0984). The Public model exhibits negative MCC (-0.0671), honestly demonstrating that "
            "the classifier struggles to separate true negatives from false positives in the public student population."
        )

    st.markdown("---")

    # =========================================================================
    # SECTION D: DEPLOYMENT MODEL SELECTION
    # =========================================================================
    st.markdown("### Section D — Deployment Model Selection & Rationale")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown(
            """
            #### 🌐 Overall Cohort
            - **Selected Model:** `Gradient Boosting (100 estimators)`
            - **Test F1:** `0.7994` | **ROC-AUC:** `0.5759` | **MCC:** `0.0984` | **CV F1:** `0.7992`
            - **Selection Rationale:** Gradient Boosting was selected based on the strongest cross-cohort individual-model performance, exhibiting strong overall Test F1 and stable 10-fold cross-validation performance across the full analytical cohort.
            """
        )
        st.markdown(
            """
            #### 🏫 Public University Cohort
            - **Selected Model:** `Gradient Boosting (100 estimators)`
            - **Test F1:** `0.7758` | **ROC-AUC:** `0.5199` | **MCC:** `-0.0671` | **CV F1:** `0.7839`
            - **Selection Rationale:** Gradient Boosting was selected as the top-performing individual classifier for the public university cohort. However, its ROC-AUC (0.5199) and negative MCC indicate near-chance discrimination under class imbalance.
            """
        )

    with col_d2:
        st.markdown(
            """
            #### 🏢 Private University Cohort
            - **Selected Model:** `Gradient Boosting (100 estimators)`
            - **Test F1:** `0.8065` | **ROC-AUC:** `0.6841` | **MCC:** `0.1943` | **CV F1:** `0.7945`
            - **Selection Rationale:** Gradient Boosting was selected based on superior discrimination (ROC-AUC 0.6841) and positive correlation (MCC 0.1943), outperforming baseline linear and tree models in capturing complex non-linear feature interactions.
            """
        )
        st.markdown(
            """
            #### 🌼 Daffodil International University (DIU)
            - **Selected Model:** `Performance Soft Voting Ensemble`
            - **Ensemble Members:** `KNN + Random Forest + Extra Trees + Gradient Boosting`
            - **Test F1:** `0.8019` | **ROC-AUC:** `0.7418` | **MCC:** `0.2405` | **CV F1:** `0.8298`
            - **Selection Rationale:** Performance Soft Voting was selected based on its highest Test F1 and CV F1 among the evaluated Daffodil models. Soft probability averaging across diverse model families provides superior discrimination on this homogeneous cohort.
            """
        )

    # =========================================================================
    # METRIC INTERPRETATION & LIMITATIONS EXPANDERS
    # =========================================================================
    with st.expander("📖 How to Interpret These Research Metrics"):
        st.markdown(
            """
            - **F1 Score:** The harmonic mean of Precision and Recall:
              $$F_1 = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$
              In datasets with high positive prevalence (67.6%), models with high sensitivity naturally achieve elevated F1 scores.
            - **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):** Measures the probability that the model will rank a randomly chosen Class 1 instance higher than a randomly chosen Class 0 instance across all decision thresholds.
              - **0.50:** Represents random chance guessing (e.g., Public model at 0.5199).
              - **0.70–0.80:** Represents acceptable discrimination (e.g., Daffodil model at 0.7418).
            - **Matthews Correlation Coefficient (MCC):** Evaluates binary classification correlation incorporating all four confusion matrix quadrants:
              $$\\text{MCC} = \\frac{TP \\times TN - FP \\times FN}{\\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$
              Ranges from $-1.0$ (complete disagreement) to $+1.0$ (perfect agreement), with $0.0$ indicating no correlation. It is widely recognized as the most reliable single metric under class imbalance.
            - **CV F1 (10-Fold):** The mean F1 score computed across 10 stratified cross-validation folds on the training split, providing an unbiased estimate of generalization stability.
            """
        )

    st.markdown(
        """
        > **⚠️ Important Research Disclaimer:**
        > The models developed in this study are predictive and associative instruments trained on observational survey data.
        > They do **not** establish causal mechanisms, prove medical etiology, or constitute clinically validated diagnostic tools.
        > The target represents an empirical survey grouping: **Class 0 = No Anxiety / Low** and **Class 1 = Medium / High**.
        > Class 1 predictions describe model feature associations with elevated self-reported career concern, not a clinical diagnosis.
        """
    )


def render_methodology_page():
    st.markdown("## 🔬 Research Methodology & Pipeline Integrity")
    st.caption("End-to-end scientific methodology, feature formulation, leak-safe preprocessing, and model validation")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("### 1. Dataset & Target Formulation")
        st.markdown(
            """
            - **Source Survey Data:** Collected across undergraduate university students in Bangladesh via structured digital questionnaires (`Career_Anxiety_due_to_AI.xlsx`).
            - **Raw Records:** 3,156 completed survey entries.
            - **Inclusion / Exclusion Protocol:**
              - Included: 2nd, 3rd, and 4th-year students actively enrolled in academic programs.
              - Excluded: 1st-year students ($N=1,120$) due to lack of academic and career immersion.
            - **Analytical Sample Size:** **2,036 students** (Class 0: 660, Class 1: 1,376).
            - **Target Operationalization:**
              - Survey question: *"How much anxiety do you feel about AI affecting your future career?"*
              - Original Likert levels: `No Anxiety`, `Low`, `Medium`, `High`.
              - **Binarized Target:**
                - **Class 0:** `No Anxiety` + `Low` (32.4%)
                - **Class 1:** `Medium` + `High` (67.6%, Elevated Career Anxiety)
            """
        )

        st.markdown("### 2. 17 Verified Predictive Features")
        st.markdown(
            """
            The feature space comprises **17 verified predictive variables**:
            - **2 Categorical Features (High Cardinality):**
              - `department`: Academic department / major of study
              - `career_path`: Primary intended professional career trajectory
            - **15 Numerical / Ordinal Features:**
              - Demographic: `age`, `gender`, `academic_year`
              - AI Literacy: `ai_knowledge`, `ai_replace_jobs`, `ai_takeover_time`, `ai_future_perspective`
              - Tool Adoption: `Total_AI_Tools`, `Uses_Text_Gen`, `Uses_Coding_AI`, `Uses_Creative_AI`
              - Engineered Psychological Constructs:
                - `Threat_Perception`: 1 if student articulated specific job threats; 0 otherwise
                - `Perceived_Urgency`: $(6 - \\text{ai\\_takeover\\_time}) \\times \\text{ai\\_replace\\_jobs}$
                - `Risk_Knowledge_Gap`: $(\\text{ai\\_replace\\_jobs} \\times 2) - \\text{ai\\_knowledge}$
                - `Age_Year_Ratio`: $\\text{age} / \\text{academic\\_year}$
            """
        )

    with col_m2:
        st.markdown("### 3. Leak-Safe Preprocessing Protocol")
        st.markdown(
            """
            To eliminate data leakage, preprocessing is implemented strictly through `sklearn.compose.ColumnTransformer`
            encapsulated within a unified `Pipeline`:
            - **Numerical Pipeline:**
              - `SimpleImputer(strategy="median")`
              - `StandardScaler()`
            - **Categorical Pipeline:**
              - `SimpleImputer(strategy="most_frequent")`
              - `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`
            - **Zero Data Leakage Enforcement:**
              - Imputation statistics (medians, modes) and standard scaling parameters are fitted **exclusively on the training split** ($X_{\\text{train}}$).
              - The fitted preprocessor is applied downstream to $X_{\\text{test}}$ and production inference without refitting.
            """
        )

        st.markdown("### 4. Model Validation & Selection")
        st.markdown(
            """
            - **Holdout Separation:** Stratified 80/20 train-test split (`random_state=42`).
            - **Cross-Validation:** Stratified 10-fold cross-validation conducted on the training split.
            - **Evaluated Architectures:** Gradient Boosting, Random Forest, Extra Trees, Logistic Regression, K-Nearest Neighbors, Hard and Soft Voting Ensembles.
            - **Cohort-Specific Optimization:**
              - Gradient Boosting selected for Overall, Public, and Private cohorts based on individual-classifier generalization.
              - Performance Soft Voting Ensemble (KNN + RF + Extra Trees + GB) selected for Daffodil based on superior ensemble consensus.
            """
        )

    st.markdown("---")
    st.markdown("### 5. Explainability Architecture (SHAP)")
    st.markdown(
        """
        - **Gradient Boosting Models (Overall, Public, Private):** Explained via `shap.TreeExplainer` computing exact Shapley margins on the log-odds decision boundary toward Class 1.
        - **Daffodil Soft Voting Ensemble:** Explained via `shap.KernelExplainer` evaluating marginal shifts in ensemble predicted probability $P(Y=1)$, utilizing a deterministic 15-centroid $k$-means background reference.
        - **Additive Dummy Aggregation:** Transformed dummy variables are summed by prefix back into the 17 verified research features, satisfying the additive efficiency axiom of cooperative game theory.
        """
    )

    with st.expander("Technical Deep Dive: Transformed Feature Dimensionality"):
        st.markdown(
            """
            Because one-hot encoding expands categorical levels observed in the training split,
            the transformed feature matrix dimension varies across cohorts:
            - **Overall Cohort:** 170 dimensions (15 numerical + 155 one-hot dummy columns)
            - **Public Cohort:** 117 dimensions (15 numerical + 102 one-hot dummy columns)
            - **Private Cohort:** 145 dimensions (15 numerical + 130 one-hot dummy columns)
            - **Daffodil Cohort:** 129 dimensions (15 numerical + 114 one-hot dummy columns)
            
            The `ColumnTransformer` embedded in each `.joblib` artifact handles this transformation deterministically without runtime manual encoding.
            """
        )


def render_about_page():
    st.markdown("## ℹ️ About the Project & Research Limitations")
    st.caption("Undergraduate Final Year Research Project | Computer Science & Engineering")

    st.markdown(
        """
        ### Project Title
        **AI-Induced Career Anxiety Prediction Among University Students**
        
        ### Research Purpose & Scope
        This web application serves as an interactive research demonstration prototype developed for an undergraduate 
        final year capstone defense. The system demonstrates empirical machine learning prediction and SHAP-based 
        model explainability grounded entirely in primary survey data gathered from undergraduate university students 
        in Bangladesh.
        
        The objective is to analyze how student demographics, academic disciplines, self-reported AI literacy, 
        and perceived labor-market disruption associate with self-reported career anxiety.
        """
    )

    st.markdown("---")
    st.markdown("### ⚠️ Methodological Limitations & Ethical Disclaimers")

    st.markdown(
        """
        1. **Observational and Associative Nature:**
           The machine learning models developed in this research identify statistical patterns and feature associations 
           present in the survey dataset. **They do not establish causal relationships.** An elevated model contribution 
           indicates statistical co-occurrence with higher self-reported anxiety, not that AI perceptions cause psychological distress.
        
        2. **Non-Clinical and Non-Diagnostic Instrument:**
           The target variable `Career_Anxiety_due_to_AI` reflects a subjective Likert-scale response operationalized into 
           binary research groups:
           - **Class 0:** No Anxiety / Low
           - **Class 1:** Medium / High (Elevated Anxiety)
           This system is **not** a psychological or psychiatric diagnostic tool, is not calibrated against standardized 
           clinical criteria (such as GAD-7 clinical thresholds), and must never be utilized for medical triage or psychological evaluation.
        
        3. **Survey Self-Report Biases:**
           Data was collected via self-administered questionnaires. Responses are subject to self-selection bias, 
           social desirability bias, and subjective interpretations of AI knowledge and job replacement urgency.
        
        4. **Geographic and Demographic Scope:**
           The analytical sample ($N=2,036$) comprises 2nd- through 4th-year undergraduate university students in Bangladesh. 
           Findings reflect this specific socio-economic and educational context and cannot be generalized to graduate students, 
           working professionals, or international cohorts without independent empirical validation.
        
        5. **Cohort Nesting & Interdependence:**
           Daffodil International University ($N=695$) is an institutional sub-cohort nested within the Private University 
           cohort ($N=1,168$). Their sample populations overlap and must not be treated as mutually exclusive datasets.
        
        6. **Interpretation of Model Probabilities:**
           Displayed probabilities represent model-estimated class probabilities under the empirical survey distribution. 
           They are not guaranteed calibrated risk estimates and should be viewed solely as decision-support ranking scores.
        """
    )

    st.markdown("---")
    st.caption(
        "AI Career Anxiety Prediction System — Phase 3 (Research Results & Presentation Layer) | "
        "Strictly Non-Clinical Research Decision Support Prototype"
    )


# =========================================================================
# APPLICATION ENTRYPOINT
# =========================================================================
def main():
    st.set_page_config(
        page_title="AI Career Anxiety Prediction System",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar Header & Navigation
    st.sidebar.title("AI Career Anxiety")
    st.sidebar.caption("Undergraduate Final Year Research Project Prototype")

    nav_selection = st.sidebar.radio(
        "Navigation",
        options=[
            "Prediction",
            "Explainability",
            "Research Results",
            "Methodology",
            "About & Limitations",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Active Cohort Models")
    for c_name, m_name in MODEL_NAMES.items():
        st.sidebar.markdown(f"- **{c_name}:** `{m_name.split('(')[0].strip()}`")

    st.sidebar.markdown("---")
    # Corrected wording from Step 1 audit: avoid claiming '170-dim' universally
    st.sidebar.caption("System Version: Phase 3 (Explainability) | Protocol: Leak-Safe Preprocessing Pipeline")

    # Main Canvas Header
    st.title("AI-Induced Career Anxiety Prediction Among University Students")
    st.caption("A Leak-Safe Comparative Machine Learning System Across Public and Private University Cohorts")

    # Route navigation
    if nav_selection == "Prediction":
        render_prediction_page()
    elif nav_selection == "Explainability":
        render_explainability_page()
    elif nav_selection == "Research Results":
        render_research_results_page()
    elif nav_selection == "Methodology":
        render_methodology_page()
    elif nav_selection == "About & Limitations":
        render_about_page()


if __name__ == "__main__":
    main()
