# Docker setup for the Pension Asset-Allocation Change Prediction API

Copy these four files into the **root of your existing AML CAPSTONE PROJECT folder**:

- `Dockerfile`
- `docker-compose.yml`
- `requirements-docker.txt`
- `.dockerignore`

Do not create a new project folder for them. The project root should already contain:

- `api/` (including `main.py` and `schemas.py`)
- `models/` (including `material_allocation_change_model.joblib` and `model_metadata.json`)

The model files must be generated first by running `python scripts/run_training.py` from the project root. The Docker image does not train the model; it packages the trained model and serves predictions through the API.

## Build and start

Open a PowerShell terminal in the project root and run:

```powershell
docker compose up --build
```

Keep that terminal open while the service is running. In your browser, check:

- `http://localhost:8000/health` — should return a healthy status and `model_loaded: true`.
- `http://localhost:8000/docs` — interactive API documentation, where you can test `POST /predict`.

To stop the service, press `Ctrl+C` in the terminal, then run:

```powershell
docker compose down
```

## Important

- The Docker build intentionally excludes CSV files, Excel files, notebooks, PDFs and development folders. Raw administrative data is not needed inside the running API container.
- Do not delete or rename the two model artifacts in `models/`.
- If port 8000 is already occupied by the local non-Docker API, stop that local Uvicorn process before starting the container.
