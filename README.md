# NBA Play Ranker

A decision-support tool for basketball coaches and analysts. Coaches get ranked play-type recommendations for upcoming matchups; analysts explore the underlying data, evaluate model performance, and review shot-level analysis.

The [Iteration 14 elaboration evidence](docs/iteration-14/README.md) links the
assigned stories, test results, deployment records and native model revision 51.
This release adds analyst Top-K uplift JSON/CSV endpoints, a Calibration tab
on Model Metrics, and the coach-owned Gameplan storage schema. Calibration is a
retrospective season-holdout diagnostic; browser Gameplan cloud persistence is
the separate future SCRUM-482 task. The storage API contract is documented in
[gameplan-storage-contract.md](docs/gameplan-storage-contract.md).

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 22 LTS and npm

### 1. Clone and install

```bash
git clone https://github.com/Basketball-Capstone-25/NBAPlayRanker.git
cd NBAPlayRanker

# Frontend
npm ci

# Backend
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install --prefer-binary -r requirements.txt
# Development/test tools
pip install --prefer-binary -r requirements-dev.txt
python -m spacy download en_core_web_sm
cd ..
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in:

| Variable | Where to find it |
|----------|-----------------|
| `SUPABASE_JWT_SECRET` | Only needed for legacy HS256 tokens; use the project's legacy JWT secret |
| `SUPABASE_URL` | Supabase project URL; ES256/RS256 tokens are verified through its public JWKS |
| `NEXT_PUBLIC_SUPABASE_URL` | Same as `SUPABASE_URL` |
| `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` | The project's publishable key (`sb_publishable_…`); the SDK also accepts a legacy `anon` key |
| `NEXT_PUBLIC_API_BASE` | `http://localhost:8000` (default for local dev) |

Both applications read this repository-root `.env`. The backend also accepts
`backend/.env` for backend-only overrides; exported shell variables take
precedence over either file. Backend authentication fails closed when credentials are
missing. For ES256/RS256 projects, configure `SUPABASE_URL`; no private signing
key or legacy secret is needed. Never commit either `.env` file.

For isolated local data experiments only, setting `ALLOW_INSECURE_DEV_AUTH=true`
explicitly disables backend authentication and role checks. Its default is
`false`; keep it false for normal login testing and all hosted environments.

### 3. Run

```bash
# Terminal 1 — Backend
cd backend
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m uvicorn application.api_coordination.app:app --host 127.0.0.1 --port 8000

# Terminal 2 — Frontend
npm run dev
```

Open http://localhost:3000.

The backend health check is http://127.0.0.1:8000/health and its interactive API
documentation is http://127.0.0.1:8000/docs. Login also requires a running
Supabase project with the team's auth/profile schema and matching credentials.
Access is determined by the current protected `profiles.role`. New accounts
remain pending until an administrator assigns their role. Browser CSV/PDF
exports use a same-origin authenticated `/api/exports` route; session tokens
are never placed in download URLs.

To build/run the backend container from the repository root:

```bash
docker build -t nba-playranker-api ./backend
docker run --rm --env-file .env -p 8000:8080 nba-playranker-api
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
| `/model-metrics` | Cross-validation comparison and held-out PPP calibration |
| `/shot-explorer` | Browse NBA play-by-play shot data |
| `/shot-heatmap` | Court heatmap of shot locations by team/player |
| `/shot-plan` | Shot-type ranking by location and context |
| `/shot-model-metrics` | Shot prediction model cross-validation metrics |
| `/shot-statistical-analysis` | Shot model statistical breakdown |

---

### Datasets

Two datasets are included in the repository:

1. **Synergy play-type data** (`backend/data/synergy_playtypes_2019_2025_players.csv`) — historical play-type performance by team, opponent, and season. Powers the baseline and context-ML recommendation engines. Precomputed predictions are in `backend/data/ml_offense_ppp_predictions.csv`.

2. **NBA play-by-play shots** (`backend/data/pbp/`) — shot records sourced via hoopR. The distributed `shots_clean.parquet`, `shots_agg.parquet`, `shots_agg_league.parquet` and `cache/shots_canonical.parquet` power the shot explorer, heatmaps, shot plans, and shot model analysis. The raw download is not needed to serve these generated files.

To rebuild missing/stale aggregates and the canonical cache from the clean data,
run from the repository root with the backend environment activated:

```bash
python backend/data/etl/build_pbp_pipeline.py
```

To rebuild the clean data too, obtain the original
`backend/data/pbp/nba_pbp_2021_present.parquet` using the team's source snapshot or
`backend/data/etl/download_hoopr_pbp.R`, then run the same command with `--force`.
This intentionally requires the raw source and regenerates derived files.

---

## Tests

From the repository root after installing the development requirements:

```bash
backend/.venv/bin/python -m pytest backend/tests -v
npx vitest run --maxWorkers=1 --minWorkers=1
npx tsc --noEmit
npm run build
```

On Windows, use `backend\.venv\Scripts\python` for the Python command. For the
faster backend subset, add `-m "not integration"`. Passing tests do not replace
the authenticated coach/analyst browser walkthrough against the deployment.

| File | What it covers |
|------|----------------|
| `test_baseline.py` | Baseline recommender output shape and values |
| `test_baseline_api.py` | `/rank-plays/baseline` endpoint validation |
| `test_ridge_model.py` | Ridge pipeline structure, fitting, regularization |
| `test_context_ml.py` | Context factors, time calculations, labeling |
| `test_access_control.py` | JWT validation, role extraction, session checks |
| `test_access_control_api_bypass.py` | RBAC enforcement across coach/analyst endpoints |
| `test_supabase_jwt.py` | ES256 signature/issuer/expiry checks without a legacy secret; fail-closed errors |
| `test_access_analyst_workspace_api.py` | Analyst workspace filtering and limits |
| `test_nlp_parser.py`, `test_nlp_integration.py`, `test_nlp_explain.py` | Prompt extraction, defaults, explanations and API integration |
| `test_pbp_cache.py` | Shot cache generation without the raw download and clean-input invalidation |
| `test_topk_uplift.py` | Weighted uplift arithmetic, JSON/CSV parity, validation and analyst access |
| `test_calibration.py` | Temporal folds, train-only preprocessing, calibration metrics and sparse bins |
| `middleware.auth-analyst.test.ts` | Analyst middleware routing |
| `middleware.auth-coach.test.ts` | Coach middleware routing |
| `export-route.test.ts`, `authenticated-download.test.ts` | Session-cookie export proxy, allowed file paths, role failures and download delivery |

---

## Tech Stack

- **Frontend:** Next.js 15, React 19, TypeScript, Supabase SSR
- **Backend:** FastAPI, Python 3.11
- **ML:** scikit-learn (Ridge regression), pandas, scipy
- **Auth:** Supabase
- **Visualization:** SportyPy (court diagrams), Matplotlib (heatmaps), ReportLab (PDF export)
- **Testing:** pytest (backend), Vitest (frontend)
