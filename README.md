# Basketball Game Strategy Analysis (BGS) — PSPI (Elaboration / Defense-Ready Prototype)

This repository contains the **Potentially Shippable Product Increment (PSPI)** for our capstone: a **decision-support prototype** that helps coaches/analysts identify effective offensive play types against a selected opponent.  

The PSPI is designed to be understandable to **non-basketball stakeholders** by showing a clear workflow:
1) **Explore raw matchup data**  
2) Generate **transparent baseline recommendations** (explainable ranking)  
3) Generate **AI contextual recommendations** (situation-aware ranking)  
4) Review **model testing + evaluation evidence** (metrics + visuals)

---

## What problem does this solve?
In real-world scouting and game planning, staff often review multiple tables/reports under time pressure. This tool consolidates matchup tendencies and produces a ranked shortlist of play types **with evidence**, reducing manual comparison and improving consistency of recommendations.

---

## What’s included (Pages)
- **Home (`/`)**  
  Explains the problem, workflow, and links to the core modules.

- **Data Explorer (`/data-explorer`)**  
  Displays a preview of **raw, non-predicted** matchup data (play-type level) filtered by season/team/opponent.  
  Includes **CSV export** so analysts can use the data externally.

- **Matchup Console — Baseline (`/matchup`)**  
  Produces Top-K ranked play types using a **transparent baseline formula**.  
  Shows the ranking breakdown and a short “why” rationale for each recommendation.

- **Context Simulator — AI (`/context`)**  
  Produces Top-K ranked play types using an **ML-based PPP estimate** combined with game context inputs  
  (e.g., time remaining, score margin).  
  Compares **Baseline vs AI** to show how recommendations change by situation.

- **Model Metrics (`/model-metrics`)**  
  Shows **model evaluation evidence** (e.g., RMSE/MAE/R²) across candidate models and includes at least one
  visualization to support model selection.

- **Glossary (`/glossary`)**  
  Quick definitions for both basketball + ML terms used in the app.

---

## Backend (API + Recommenders)
This project includes a backend service that:
- exposes endpoints for **raw data retrieval and CSV export**
- computes **baseline** and **context-ML** ranked recommendations
- supports **multi-user concurrent usage** via a stateless request design and safe model/data loading at startup

Core backend modules:
- `app.py` — FastAPI service + endpoints  
- `baseline_recommender.py` — baseline scoring + raw matchup data functions  
- `ml_context_recommender.py` — contextual ranking using ML outputs + context adjustments  
- `ml_models.py` — training + evaluation pipeline (candidate model comparisons)

---

## Dataset
The app is intended to be evaluated on a selected historical dataset (play-type performance by team/opponent/season).
The **Model Metrics** page and ML pipeline are used to validate model suitability and show evidence-based model choice.

> Note: The Data Explorer page shows **raw observed values only** (no predicted/calculated values in the preview table).

---

## Tech Stack
- **Frontend:** Next.js (App Router), React  
- **Backend:** FastAPI (Python)  
- **ML / Data:** scikit-learn, pandas (modeling + evaluation)

---

## How to Run (Local)

### Frontend
```bash
npm install
npm run dev
# open http://localhost:3000

# from backend folder (or project root if applicable)
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2

