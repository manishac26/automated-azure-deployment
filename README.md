# Network Intrusion Detection Using AutoML

Comparative evaluation of Random Forest baselines, hyperparameter optimization (HPO), and AutoML (FLAML) on CIC-IDS2017.

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Data

Place the extracted CIC-IDS2017 CSV files in:

`MachineLearningCSV/MachineLearningCVE/`

The raw dataset is not included in this repository.

## Run the pipeline

Run these commands from the project root:

```bash
python -m src.audit_dataset
python -m src.preprocessing
python -m src.split_data
```

## Experiments

```bash
python -m src.run_baseline
python -m src.run_baseline_100k
python -m src.run_grid_search
python -m src.run_random_search
python -m src.run_optuna_search
python -m src.run_flaml
```

## API backend

Start the FastAPI backend from the project root:

```bash
uvicorn deployment.app:app --host 0.0.0.0 --port 8000
```

Health and metadata endpoints:

- `GET /health`
- `GET /metadata`
- `POST /predict`

See `deployment/README.md` and `deployment/example_request.json` for details.

## Frontend

The frontend uses React, TypeScript, and Vite.

```bash
cd frontend
npm install
```

Create a local `.env` file based on `.env.example` and set `VITE_API_BASE_URL` to your backend URL. Then run:

```bash
npm run dev
```

The default frontend URL is `http://localhost:5173`. The API base URL in `.env.example` is configurable.

To build the frontend:

```bash
npm run build
```

## Tests

From the project root:

```bash
pytest tests/ -v
```

## Results and reports

- `results/` contains experiment metrics and figures.
- `reports/` contains the dataset audit, experiment design, final report, model card, interpretability notes, reliability analysis, and presentation outline.

## Project scope

This project compares a Random Forest baseline, Grid Search, Random Search, Optuna, and FLAML AutoML for binary network intrusion detection using CIC-IDS2017.
