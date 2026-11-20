# Interactive Portfolio Assessment update

This update turns the existing Streamlit dashboard into one single-screen
**Pension Portfolio Monitor & Early-Warning System**.

## What the user does

1. Selects an anonymised scheme reference and current reporting year.
2. The app retrieves that scheme's immediately preceding annual record from
the authorised local RBA source file.
3. The previous-year positive asset classes are preselected and their amounts
are shown as starting values. The user updates those values to the current
reporting-year amounts, removes classes no longer held, or adds newly held
classes. Unselected asset classes are treated as zero.
4. The app calculates the exact 26 predictors used by the existing model and
sends them to the existing FastAPI `/predict` endpoint.
5. One dashboard view displays previous-versus-current portfolio amounts and
shares, HHI/diversification comparisons, and the calibrated model probability,
classification and monitoring message.

No new model is trained. No SHAP explanation is displayed. The existing API,
model artifacts, model threshold and training pipeline are retained.

## Files in this update

- `dashboard/app.py` — replacement single-screen Streamlit application.
- `dashboard/assessment.py` — feature calculation for current user-entered amounts.
- `Dockerfile` — updated to include the existing `src/` feature-engineering package.
- `docker-compose.yml` — mounts the authorised source CSV into the dashboard
  container as a read-only runtime file; it is not copied into the image.
- `tests/test_interactive_assessment.py` — tests for the 26-feature construction.

## Apply the update

From the extracted `interactive_dashboard_update` folder, copy/merge these files
into the existing `AML CAPSTONE PROJECT` folder, preserving all other project
files:

- Replace `dashboard/app.py`.
- Add `dashboard/assessment.py` and `dashboard/__init__.py`.
- Replace the root `Dockerfile`.
- Replace the root `docker-compose.yml`.
- Copy `tests/test_interactive_assessment.py` into the existing `tests/` folder.

Do not replace `api/`, `models/`, `src/`, `scripts/run_training.py`,
`scripts/predict.py`, or the model artifacts.

## Private source-file requirement

The project root must contain the authorised file named exactly:

`COPY 1 FINAL.csv`

Compose mounts it into the dashboard container at `/private_data/rba_source.csv`
with read-only access. The Dockerfile does not copy this CSV into the image.
Keep the raw CSV out of public GitHub. Add the following to `.gitignore` if it
is not already ignored:

```gitignore
COPY 1 FINAL.csv
```

## Rebuild and run

Run from the project root:

```powershell
docker compose down
docker compose up --build -d
```

Check both services:

```powershell
docker compose ps
```

Open the single dashboard at `http://localhost:8501`; the API documentation
remains at `http://localhost:8000/docs`.

## Important interpretation notes

- The model estimates whether at least one asset-class share changes by an
  absolute 5 percentage points or more between consecutive reporting years.
- It does not identify which asset class will change, the direction of change,
  or whether a change is harmful or non-compliant.
- If no valid consecutive previous-year record exists, the app cannot produce
  a model assessment for that scheme/year.
- For an input year of 2025, the app predicts 2026. The model's held-out test
  outcomes end in 2025, so a 2026 prediction is a prospective prototype signal
  beyond the independent test window and must be interpreted cautiously.
- Scheme-level records are read in memory from the private, read-only mount.
  They are not exported to the aggregate dashboard files or saved by the app.
