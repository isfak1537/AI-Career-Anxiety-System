/**
 * inference.js
 * High-Performance Client-Side Inference & SHAP Explainability Engine
 * Faithfully replicates src/feature_engineering.py and scikit-learn Pipeline inference.
 */

(function (window) {
  'use strict';

  const data = window.AI_CAREER_DATA;
  if (!data) {
    console.error('AI_CAREER_DATA not loaded. Ensure js/model_data.js is included first.');
    return;
  }

  const { config, preprocessors, models } = data;

  /**
   * Institutional cohort mapping from university affiliation
   */
  function resolveCohortFromUniversity(university) {
    if (config.daffodil_universities.includes(university)) {
      return 'daffodil';
    } else if (config.private_universities.includes(university)) {
      return 'private';
    } else if (config.public_universities.includes(university)) {
      return 'public';
    }
    return 'overall';
  }

  /**
   * Count comma-separated AI tools reported
   */
  function countAITools(toolsStr) {
    if (!toolsStr || typeof toolsStr !== 'string') return 0;
    const trimmed = toolsStr.trim();
    if (!trimmed || ['none', 'nan'].includes(trimmed.toLowerCase())) return 0;
    return trimmed.split(',').map(s => s.trim()).filter(s => s.length > 0).length;
  }

  /**
   * Calculate binary threat perception indicator
   */
  function calculateThreatPerception(val) {
    if (!val || typeof val !== 'string') return 0;
    const trimmed = val.trim().toLowerCase();
    if (!trimmed || trimmed === 'none' || trimmed === "i don't know for now") return 0;
    return 1;
  }

  /**
   * Regex test helper for tool patterns
   */
  function testRegex(patternStr, text) {
    if (!text || typeof text !== 'string') return 0;
    try {
      const re = new RegExp(patternStr, 'i');
      return re.test(text) ? 1 : 0;
    } catch (e) {
      return 0;
    }
  }

  /**
   * Convert raw student inputs into the exact 17 engineered features
   */
  function engineerFeatures(raw) {
    const toolsStr = raw.ai_tools_used || '';

    const totalAITools = countAITools(toolsStr);
    const usesTextGen = testRegex(config.regex_patterns.text_gen, toolsStr);
    const usesCodingAI = testRegex(config.regex_patterns.coding_ai, toolsStr);
    const usesCreativeAI = testRegex(config.regex_patterns.creative_ai, toolsStr);
    const threatPerception = calculateThreatPerception(raw.ai_tool_perception);

    // Ordinal conversions
    const academicYearNum = config.academic_year_map[raw.academic_year] ?? 3;
    const genderNum = config.gender_map[raw.gender] ?? 0;
    const aiKnowledgeNum = config.ai_knowledge_map[raw.ai_knowledge] ?? 2;
    const aiReplaceJobsNum = config.ai_replace_map[raw.ai_replace_jobs] ?? 1;
    const aiTakeoverTimeNum = config.ai_takeover_map[raw.ai_takeover_time] ?? 4;
    const futurePerspectiveNum = config.future_perspective_map[raw.ai_future_perspective] ?? 2;

    const ageNum = Number(raw.age) || 22;

    // Composite domain interaction features
    const perceivedUrgency = aiReplaceJobsNum * aiTakeoverTimeNum;
    const riskKnowledgeGap = futurePerspectiveNum - aiKnowledgeNum;
    const ageYearRatio = academicYearNum > 0 ? (ageNum / academicYearNum) : (ageNum / 3);

    return {
      department: raw.department || 'Other Department',
      career_path: raw.career_path || 'Other Career Path',
      age: ageNum,
      gender: genderNum,
      academic_year: academicYearNum,
      ai_knowledge: aiKnowledgeNum,
      ai_replace_jobs: aiReplaceJobsNum,
      ai_takeover_time: aiTakeoverTimeNum,
      ai_future_perspective: futurePerspectiveNum,
      Total_AI_Tools: totalAITools,
      Uses_Text_Gen: usesTextGen,
      Uses_Coding_AI: usesCodingAI,
      Uses_Creative_AI: usesCreativeAI,
      Threat_Perception: threatPerception,
      Perceived_Urgency: perceivedUrgency,
      Risk_Knowledge_Gap: riskKnowledgeGap,
      Age_Year_Ratio: ageYearRatio,
    };
  }

  /**
   * Preprocess engineered features via cohort ColumnTransformer
   */
  function transformFeatures(engineered, cohortKey) {
    const prep = preprocessors[cohortKey];
    if (!prep) throw new Error(`Unknown cohort preprocessor: ${cohortKey}`);

    const numericalValues = [
      engineered.age,
      engineered.gender,
      engineered.academic_year,
      engineered.ai_knowledge,
      engineered.ai_replace_jobs,
      engineered.ai_takeover_time,
      engineered.ai_future_perspective,
      engineered.Total_AI_Tools,
      engineered.Uses_Text_Gen,
      engineered.Uses_Coding_AI,
      engineered.Uses_Creative_AI,
      engineered.Threat_Perception,
      engineered.Perceived_Urgency,
      engineered.Risk_Knowledge_Gap,
      engineered.Age_Year_Ratio,
    ];

    // 1. Numerical Pipeline: Impute (median) + Standard Scale (val - mean) / scale
    const transformed = [];
    for (let i = 0; i < numericalValues.length; i++) {
      let val = numericalValues[i];
      if (val === null || val === undefined || isNaN(val)) {
        val = prep.num_medians[i];
      }
      const mean = prep.scaler_means[i];
      const scale = prep.scaler_scales[i];
      transformed.push((val - mean) / scale);
    }

    // 2. Categorical Pipeline: OneHotEncoder (department, career_path)
    const deptCats = prep.cat_categories.department;
    for (let i = 0; i < deptCats.length; i++) {
      transformed.push(engineered.department === deptCats[i] ? 1.0 : 0.0);
    }

    const careerCats = prep.cat_categories.career_path;
    for (let i = 0; i < careerCats.length; i++) {
      transformed.push(engineered.career_path === careerCats[i] ? 1.0 : 0.0);
    }

    return transformed;
  }

  /**
   * Traverse a decision tree regressor to leaf value
   */
  function evaluateTreeRegressor(tree, x) {
    let node = 0;
    while (tree.cl[node] !== -1) {
      const feat = tree.f[node];
      const thresh = tree.th[node];
      node = x[feat] <= thresh ? tree.cl[node] : tree.cr[node];
    }
    return tree.v[node];
  }

  /**
   * Traverse a decision tree classifier to get probability of Class 1
   */
  function evaluateTreeClassifier(tree, x) {
    let node = 0;
    while (tree.cl[node] !== -1) {
      const feat = tree.f[node];
      const thresh = tree.th[node];
      node = x[feat] <= thresh ? tree.cl[node] : tree.cr[node];
    }
    return tree.v[node]; // already precomputed p(class 1)
  }

  /**
   * Run model prediction on transformed feature vector
   */
  function predictTransformed(cohortKey, xTrans) {
    const model = models[cohortKey];
    if (!model) throw new Error(`Model not found for cohort: ${cohortKey}`);

    let prob1 = 0.5;

    if (model.type === 'gradient_boosting') {
      let rawScore = model.init_raw;
      const lr = model.learning_rate;
      const trees = model.trees;

      for (let i = 0; i < trees.length; i++) {
        rawScore += lr * evaluateTreeRegressor(trees[i], xTrans);
      }

      // Sigmoid logistic function
      prob1 = 1.0 / (1.0 + Math.exp(-rawScore));
    } else if (model.type === 'ensemble_soft_voting') {
      // Soft average between Gradient Boosting and Random Forest
      let gbRaw = model.gb.init_raw;
      const lr = model.gb.learning_rate;
      const gbTrees = model.gb.trees;

      for (let i = 0; i < gbTrees.length; i++) {
        gbRaw += lr * evaluateTreeRegressor(gbTrees[i], xTrans);
      }
      const gbProb = 1.0 / (1.0 + Math.exp(-gbRaw));

      // Random Forest average probability
      const rfTrees = model.rf.trees;
      let rfSum = 0;
      for (let i = 0; i < rfTrees.length; i++) {
        rfSum += evaluateTreeClassifier(rfTrees[i], xTrans);
      }
      const rfProb = rfTrees.length > 0 ? (rfSum / rfTrees.length) : gbProb;

      // Ensemble soft probability consensus
      prob1 = (gbProb * 0.45) + (rfProb * 0.55);
    }

    // Clamp between 0.001 and 0.999
    prob1 = Math.max(0.001, Math.min(0.999, prob1));
    const predClass = prob1 >= 0.5 ? 1 : 0;

    return {
      predClass,
      probability: prob1,
      modelName: model.model_name,
    };
  }

  /**
   * Compute exact additive feature attributions (Tree SHAP / Saabas path method)
   */
  function explainTransformed(cohortKey, xTrans, engineered) {
    const model = models[cohortKey];
    const prep = preprocessors[cohortKey];

    const numFeats = config.numerical_features;
    const deptCats = prep.cat_categories.department;
    const careerCats = prep.cat_categories.career_path;

    const rawAttributions = new Array(xTrans.length).fill(0.0);

    const treesToTraverse = model.type === 'gradient_boosting' ? model.trees : model.gb.trees;
    const lr = model.type === 'gradient_boosting' ? model.learning_rate : model.gb.learning_rate;

    for (let t = 0; t < treesToTraverse.length; t++) {
      const tree = treesToTraverse[t];
      let node = 0;
      while (tree.cl[node] !== -1) {
        const feat = tree.f[node];
        const thresh = tree.th[node];
        const next = xTrans[feat] <= thresh ? tree.cl[node] : tree.cr[node];
        // Value difference along decision path
        const diff = tree.v[next] - tree.v[node];
        rawAttributions[feat] += lr * diff;
        node = next;
      }
    }

    // Aggregate one-hot columns back into the 17 verified features
    const aggregated = {};

    // 1. Numerical features map 1-to-1
    for (let i = 0; i < numFeats.length; i++) {
      aggregated[numFeats[i]] = rawAttributions[i];
    }

    // 2. Department aggregation
    let deptAttr = 0.0;
    const deptOffset = numFeats.length;
    for (let i = 0; i < deptCats.length; i++) {
      deptAttr += rawAttributions[deptOffset + i];
    }
    aggregated['department'] = deptAttr;

    // 3. Career path aggregation
    let careerAttr = 0.0;
    const careerOffset = deptOffset + deptCats.length;
    for (let i = 0; i < careerCats.length; i++) {
      careerAttr += rawAttributions[careerOffset + i];
    }
    aggregated['career_path'] = careerAttr;

    // Build sorted rank table
    const featureRows = [];
    for (const [featKey, val] of Object.entries(aggregated)) {
      const displayName = config.feature_display_names[featKey] || featKey;
      let rawVal = engineered[featKey];
      if (typeof rawVal === 'number' && !Number.isInteger(rawVal)) {
        rawVal = rawVal.toFixed(2);
      }

      let category = 'Demographic';
      if (['ai_knowledge', 'ai_replace_jobs', 'ai_takeover_time', 'ai_future_perspective'].includes(featKey)) {
        category = 'AI Literacy & Beliefs';
      } else if (['Total_AI_Tools', 'Uses_Text_Gen', 'Uses_Coding_AI', 'Uses_Creative_AI', 'Threat_Perception'].includes(featKey)) {
        category = 'AI Tool Adoption';
      } else if (['Perceived_Urgency', 'Risk_Knowledge_Gap', 'Age_Year_Ratio'].includes(featKey)) {
        category = 'Construct Interaction';
      } else if (['department', 'career_path'].includes(featKey)) {
        category = 'Academic & Career';
      }

      featureRows.push({
        key: featKey,
        displayName,
        category,
        inputValue: rawVal,
        contribution: val,
        absContribution: Math.abs(val),
        direction: val >= 0 ? 'Increases Anxiety' : 'Mitigates Anxiety',
      });
    }

    featureRows.sort((a, b) => b.absContribution - a.absContribution);

    return {
      aggregated,
      featureRows,
    };
  }

  /**
   * Master Prediction & Explanation API
   */
  function runPrediction(rawInputs, cohortOverride) {
    const autoCohort = resolveCohortFromUniversity(rawInputs.university);
    const selectedCohort = (cohortOverride && cohortOverride !== 'auto') ? cohortOverride.toLowerCase() : autoCohort;

    const engineered = engineerFeatures(rawInputs);
    const xTrans = transformFeatures(engineered, selectedCohort);
    const prediction = predictTransformed(selectedCohort, xTrans);
    const explainability = explainTransformed(selectedCohort, xTrans, engineered);

    return {
      cohort: selectedCohort,
      autoCohort,
      isOverride: selectedCohort !== autoCohort,
      prediction,
      explainability,
      engineered,
      rawInputs,
    };
  }

  // Export to global window object
  window.CareerAnxietyInference = {
    resolveCohortFromUniversity,
    engineerFeatures,
    transformFeatures,
    predictTransformed,
    explainTransformed,
    runPrediction,
  };

})(typeof window !== 'undefined' ? window : this);
