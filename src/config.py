"""
Research Configuration and Global Constants
Extracted strictly from final_all(1).ipynb without modifications.
"""

from typing import Dict, List

# ============================================================
# 1. GLOBAL EXPERIMENTAL CONFIGURATION
# ============================================================
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 10
CV_SHUFFLE: bool = True
PRIMARY_SCORING: str = "f1"

TARGET_COLUMN: str = "Anxiety_Label"
RAW_TARGET_COLUMN: str = "career_anxiety"

TARGET_ACADEMIC_YEARS: List[str] = [
    "2nd Year",
    "3rd Year",
    "4th Year",
]

# ============================================================
# 2. EXACT 17 FINAL PREDICTIVE FEATURES
# ============================================================
CATEGORICAL_FEATURES: List[str] = [
    "department",
    "career_path",
]

NUMERICAL_FEATURES: List[str] = [
    "age",
    "gender",
    "academic_year",
    "ai_knowledge",
    "ai_replace_jobs",
    "ai_takeover_time",
    "ai_future_perspective",
    "Total_AI_Tools",
    "Uses_Text_Gen",
    "Uses_Coding_AI",
    "Uses_Creative_AI",
    "Threat_Perception",
    "Perceived_Urgency",
    "Risk_Knowledge_Gap",
    "Age_Year_Ratio",
]

FINAL_FEATURES_17: List[str] = [
    "department",
    "career_path",
    "age",
    "gender",
    "academic_year",
    "ai_knowledge",
    "ai_replace_jobs",
    "ai_takeover_time",
    "ai_future_perspective",
    "Total_AI_Tools",
    "Uses_Text_Gen",
    "Uses_Coding_AI",
    "Uses_Creative_AI",
    "Threat_Perception",
    "Perceived_Urgency",
    "Risk_Knowledge_Gap",
    "Age_Year_Ratio",
]

# 15 raw source columns present in Career_Anxiety_due_to_AI.xlsx
RAW_SOURCE_COLUMNS: List[str] = [
    "participant_id",
    "university",
    "department",
    "age",
    "gender",
    "academic_year",
    "ai_knowledge",
    "ai_tools_used",
    "career_path",
    "ai_tool_perception",
    "ai_future_perspective",
    "ai_replace_jobs",
    "ai_takeover_time",
    "career_anxiety",
    "consent",
]

# ============================================================
# 3. EXACT FEATURE ENGINEERING MAPPINGS
# ============================================================
ANXIETY_MAP: Dict[str, int] = {
    "No Anxiety": 0,
    "Low": 0,
    "Medium": 1,
    "High": 1,
}

ACADEMIC_YEAR_MAP: Dict[str, int] = {
    "2nd Year": 2,
    "3rd Year": 3,
    "4th Year": 4,
}

GENDER_MAP: Dict[str, int] = {
    "Male": 0,
    "Female": 1,
}

AI_KNOWLEDGE_MAP: Dict[str, int] = {
    "None": 0,
    "Low": 1,
    "Medium": 2,
    "High": 3,
}

AI_REPLACE_MAP: Dict[str, int] = {
    "No": 0,
    "Partially": 1,
    "Fully": 2,
}

AI_TAKEOVER_MAP: Dict[str, int] = {
    "Never": 0,
    "50+ years": 1,
    "21–50 years": 2,
    "11–20 years": 3,
    "6–10 years": 4,
    "1–5 years": 5,
}

FUTURE_PERSPECTIVE_MAP: Dict[str, int] = {
    "I strongly believe AI cannot take over human activities and will be a helpful tool for humans.": 0,
    "I believe AI will mostly assist humans, with limited job replacement.": 1,
    "I think AI will replace some human jobs but also create new opportunities.": 2,
    "I believe AI will significantly replace human jobs and create major challenges.": 3,
    "I strongly believe AI will completely take over human jobs and pose serious threat to humanity.": 4,
}

# ============================================================
# 4. REGEX PATTERNS FOR AI TOOLS
# ============================================================
TEXT_GEN_PATTERN: str = r"ChatGPT|Claude|Gemini|Quillbot|Quiltbot|Grammarly"
CODING_AI_PATTERN: str = r"Copilot|Cursor|Replit|Deepseek"
CREATIVE_AI_PATTERN: str = r"Midjourney|DALL[- ]?E|Stable Diffusion|Canva|Napkin|Photomath"

# ============================================================
# 5. COHORT UNIVERSITY DEFINITIONS
# ============================================================
PUBLIC_UNIVERSITIES: List[str] = [
    "University of Dhaka",
    "Jahangirnagar University",
    "Chittagong University of Engineering and Technology",
]

PRIVATE_UNIVERSITIES: List[str] = [
    "Daffodil International University",
    "American International University-Bangladesh",
]

DAFFODIL_UNIVERSITIES: List[str] = [
    "Daffodil International University",
]

COHORT_ORDER: List[str] = [
    "Overall",
    "Public",
    "Private",
    "Daffodil",
]
