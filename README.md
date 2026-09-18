# AI-Induced Career Anxiety Prediction System

> **Undergraduate Final Year Research Project | Department of Computer Science and Engineering**  
> *"A Leak-Safe Comparative Machine Learning Decision-Support System Across Public and Private University Cohorts in Bangladesh"*

---

## 🚀 Live Vercel Deployment Guide

This repository includes a native, high-performance web application designed specifically for **Vercel** with zero-dependency client inference, sub-millisecond predictions, dynamic SHAP waterfall charts, and glassmorphism styling.

### Method A: Deploy via GitHub & Vercel Dashboard (Recommended)

1. Push this repository to your GitHub account:
   ```bash
   git init
   git add .
   git commit -m "feat: AI Career Anxiety System with Vercel Web Deployment"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
2. Go to [vercel.com/new](https://vercel.com/new).
3. Select and import your GitHub repository.
4. Keep the default settings (Framework: **Other**, Root Directory: `./`).
5. Click **Deploy**. Your app will be live globally on `https://<your-project>.vercel.app` in under 20 seconds!

---

### Method B: Deploy via Vercel CLI

Run the following command directly in this directory:
```bash
npx vercel
```
- When asked `Set up and deploy?`, choose `y`.
- Accept all defaults by pressing Enter.
- To deploy directly to production:
  ```bash
  npx vercel --prod
  ```

---

## 💻 Local Preview

To test the Vercel web app locally on your machine:
```bash
# Option 1: Using Python
python3 -m http.server 8080

# Option 2: Using Node.js
npx serve .
```
Open [http://localhost:8080](http://localhost:8080) in your browser.

---

## 🎈 Alternative: Deploy Streamlit to Streamlit Community Cloud

If you also wish to run the original Python Streamlit application (`app.py`):
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your repository, set the Main file path to `app.py`, and click **Deploy**.
4. The Streamlit app will be live on `https://<your-project>.streamlit.app`.

---

## 🏛️ System Architecture

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Vercel Web App** | HTML5, Vanilla CSS, Vanilla JS | Instant global edge delivery on Vercel with responsive interactive UI |
| **Authoritative Backend** | `api/index.py` (FastAPI) | Serverless scikit-learn pipeline inference & SHAP explainability |
| **Serialized Pipelines** | `models/*.joblib` | Authoritative deployment models (Overall/Public/Private GB, Daffodil VotingClassifier) |
| **Client Fallback** | `js/inference.js` | Lightweight client-side heuristic approximation for offline CDN fallback |
| **Research Streamlit App**| Python, Streamlit, SHAP | Local and Streamlit Cloud academic presentation layer |
| **Test Suite** | `pytest` (115 passed) | Automated regression, parity, and security validation testing |

> *Note: Minimum compatible dependency versions are specified in `requirements.txt` and `pyproject.toml`.*

---

## 🔬 Research & Ethical Disclaimer

This application is strictly a **non-clinical, non-diagnostic decision support research prototype** developed for undergraduate capstone defense. The model is trained on a **secondary survey dataset** of 2,036 analytical undergraduate records (following 1,120 first-year exclusions). The target variable is an empirical survey grouping (*Class 0 = No/Low Anxiety*, *Class 1 = Medium/High Anxiety*) and has not been calibrated against clinical psychometric scales (such as GAD-7). Output values represent associative statistical patterns and model-estimated probabilities, not causal determinations or clinical diagnoses.
