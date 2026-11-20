# Streamlit dashboard add-on

This add-on provides the Streamlit research-prototype dashboard specified in the dissertation proposal. It is designed to sit alongside the already-working FastAPI prediction service.

## Dashboard contents

- Overview of the period covered and aggregate portfolio metrics
- Scheme-average asset allocation trends by asset class
- Average HHI, diversification index and active asset-class count
- Observed consecutive-year transitions and material-change rates
- Aggregate held-out model predictions for the configured 2024–2025 test period
- Global SHAP feature importance and predictor-group summaries
- Aggregate data-quality counts
- FastAPI health status, to show whether the prediction service is online

All displayed research outputs are aggregate summaries. The export script deliberately excludes `Ref` and other scheme-level identifiers and does not write row-level scheme records to `dashboard_data/`.

## Add these files to the existing project

Copy/merge the following into the root of the existing `AML CAPSTONE PROJECT` folder:

- `dashboard/` (new folder)
- `scripts/build_dashboard_data.py` (new script alongside the existing scripts)
- `requirements-dashboard.txt`
- `Dockerfile` (replace the earlier API-only Dockerfile with this combined API + dashboard Dockerfile)
- `docker-compose.yml` (replace the API-only compose file with this two-service compose file)
- `.dockerignore` (replace the earlier Docker ignore file)
- `README_DASHBOARD.md`

Do not replace `api/`, `src/`, `models/`, `scripts/run_training.py`, or `scripts/predict.py`.

## Generate aggregate dashboard data

First ensure the training pipeline has completed and these files exist:

- `models/material_allocation_change_model.joblib`
- `models/model_metadata.json`

Then, from the project root, run:

```powershell
python scripts/build_dashboard_data.py --data "COPY 1 FINAL.csv"
```

This creates `dashboard_data/` containing only aggregate CSV summaries and dashboard metadata. It reads the authorised raw file locally; the raw file is not copied into the dashboard folder or Docker image.

## Run both services with Docker Compose

From the project root:

```powershell
docker compose up --build
```

When both services are healthy:

- Dashboard: <http://localhost:8501>
- FastAPI health: <http://localhost:8000/health>
- FastAPI documentation: <http://localhost:8000/docs>

Leave the terminal running while you use the dashboard. Stop the services with `Ctrl+C`, then run `docker compose down` in another terminal if needed.

## Run the dashboard without Docker (optional local development)

Install the dashboard-only dependencies into the project's existing Python environment:

```powershell
python -m pip install -r requirements-dashboard.txt
```

Start the API separately, then run:

```powershell
streamlit run dashboard/app.py
```

The local dashboard defaults to `http://127.0.0.1:8000` for its API health check. Set `API_BASE_URL` if your API runs at a different address.

## Important interpretation and data-use notes

- Asset-class percentages are unweighted averages of scheme-level shares within each year, not industry-wide asset-weighted totals.
- A positive class prediction is a monitoring signal, not evidence of harm, imprudence or non-compliance.
- SHAP values explain the underlying base classifier before the separate sigmoid calibration layer; they are not causal effects or percentage-point changes in calibrated probability.
- The dashboard contains aggregate outputs only. Do not add raw RBA data, scheme identifiers or row-level scheme observations to the public repository.
- Confirm that the data owner and university permit public distribution of the aggregate extracts and serialized model artifacts before publishing them.
- This is a research prototype, not a live regulatory decision system.
