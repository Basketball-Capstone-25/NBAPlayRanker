# NBA Play Ranker

A decision-support tool for basketball coaches and analysts. Coaches get ranked play-type recommendations for upcoming matchups; analysts explore the underlying data, evaluate model performance, and review shot-level analysis.

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm

### 1. Clone and install

```bash
git clone https://github.com/Basketball-Capstone-25/NBAPlayRanker.git
cd NBAPlayRanker

# Frontend
npm install

# Backend
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
cd ..
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in:

| Variable | Where to find it |
|----------|-----------------|
| `SUPABASE_JWT_SECRET` | Supabase Dashboard > Settings > API > JWT Settings |
| `SUPABASE_URL` | Supabase Dashboard > Settings > API > Project URL |
| `NEXT_PUBLIC_SUPABASE_URL` | Same as `SUPABASE_URL` |
| `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` | Supabase Dashboard > Settings > API > `anon` public key |
| `NEXT_PUBLIC_API_BASE` | `http://localhost:8000` (default for local dev) |

### 3. Run

```bash
# Terminal 1 — Backend
cd backend
python -m uvicorn application.api_coordination.app:app --host 127.0.0.1 --port 8000

# Terminal 2 — Frontend
npm run dev
```

Open http://localhost:3000.

### 4. Run tests

```bash
# Backend (53 tests)
cd backend
python -m pytest tests/ -v

# Frontend RBAC middleware tests
npm run test:rbac
npm run test:rbac:coach
```

---

## Pages

### Public
| Route | Purpose |
|-------|---------|
| `/` | Landing page with workflow overview |
| `/login` | Sign in with Supabase auth |
| `/signup` | Register a new account |
| `/forgot-password` | Request password reset email |
| `/reset-password` | Set new password after email link |
| `/glossary` | Definitions for basketball and ML terms (requires sign-in) |

### Coach
| Route | Purpose |
|-------|---------|
| `/matchup` | Top-K baseline play-type rankings for a chosen matchup |
| `/context` | AI context simulator — re-ranks plays using game situation (score, period, time) |
| `/gameplan` | Visual game plan built from ranked output |

### Analyst
| Route | Purpose |
|-------|---------|
| `/data-explorer` | Browse Synergy play-type data with filtering and CSV export |
| `/statistical-analysis` | ML model evaluation (RMSE, MAE, R²) |
| `/model-metrics` | Cross-validation comparison between Baseline and ML models |
| `/shot-explorer` | Browse NBA play-by-play shot data |
| `/shot-heatmap` | Court heatmap of shot locations by team/player |
| `/shot-plan` | Shot-type ranking by location and context |
| `/shot-model-metrics` | Shot prediction model cross-validation metrics |
| `/shot-statistical-analysis` | Shot model statistical breakdown |

---

## Architecture

### Frontend
Next.js 14 (App Router) with TypeScript. Supabase handles authentication. Middleware enforces role-based routing so coaches and analysts each see only their own pages.

### Backend — 4-Tier Layered Architecture

```
application/          <- API routing, auth, service orchestration
  api_coordination/        FastAPI endpoints and routers
  access_control_services/ JWT validation, RBAC enforcement
  analytics_services/      Statistical analysis orchestration
  recommendation_services/ Ranking orchestration, PDF export

domain/               <- Core business logic (no framework dependencies)
  baseline_recommendation/   Play-type ranking logic
  context_ml_recommendation/ Game-context adjustments (score, time, period)
  shot_analysis/             Shot ETL, aggregation, ML models
  statistical_analysis/      Model evaluation and cross-validation

infrastructure/       <- External integrations and data access
  data_access/             Parquet/CSV loading and caching
  external_integrations/   NLP parsing, Supabase JWT, SportyPy rendering
  model_management/        Ridge regression training, CV pipelines
  visualization_and_export/ PDF generation, PNG court visualizations

data/                 <- Datasets (committed to git, no rebuild needed)
  synergy_playtypes_2019_2025_players.csv
  ml_offense_ppp_predictions.csv
  pbp/                     Play-by-play shot data (parquet)
  etl/                     Data build scripts (only needed to rebuild)
```

Each subsystem exposes a public interface via `__init__.py`. No upward imports between layers.

### Datasets

Two datasets are included in the repository:

1. **Synergy play-type data** (`data/synergy_playtypes_2019_2025_players.csv`) — historical play-type performance by team, opponent, and season. Powers the baseline and context-ML recommendation engines.

2. **NBA play-by-play shots** (`data/pbp/`) — 1.3M+ shot records sourced via hoopR. Powers the shot explorer, heatmaps, shot plans, and shot model analysis.

Both datasets are committed to the repo. No download or rebuild step is required to run the app.

---

## Tests

| File | Tests | What it covers |
|------|-------|----------------|
| `test_baseline.py` | 6 | Baseline recommender output shape and values |
| `test_baseline_api.py` | 3 | `/rank-plays/baseline` endpoint validation |
| `test_ridge_model.py` | 8 | Ridge pipeline structure, fitting, regularization |
| `test_context_ml.py` | 8 | Context factors, time calculations, labeling |
| `test_access_control.py` | 10 | JWT validation, role extraction, session checks |
| `test_access_control_api_bypass.py` | 12 | RBAC enforcement across coach/analyst endpoints |
| `test_access_analyst_workspace_api.py` | 3 | Analyst workspace filtering and limits |
| `middleware.auth-analyst.test.ts` | 3 | Analyst middleware routing |
| `middleware.auth-coach.test.ts` | 3 | Coach middleware routing |

---

## Tech Stack

- **Frontend:** Next.js 14, React 18, TypeScript, Supabase SSR
- **Backend:** FastAPI, Python 3.11
- **ML:** scikit-learn (Ridge regression), pandas, scipy
- **Auth:** Supabase (ES256 JWT via JWKS)
- **Visualization:** SportyPy (court diagrams), Matplotlib (heatmaps), ReportLab (PDF export)
- **Testing:** pytest (backend), Vitest (frontend)

