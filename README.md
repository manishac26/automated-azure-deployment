# Network Intrusion Detection Using AutoML

Comparative evaluation of RF baselines/HPO and AutoML (FLAML) on CIC-IDS2017.

## Setup
```bash
pip install -r requirements.txt
```

## Data
Place extracted CSVs in MachineLearningCSV/MachineLearningCVE/.

## Run pipeline
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

## API (Backend)
```bash
# from project root
uvicorn deployment.app:app --host 0.0.0.0 --port 8000
```

## Frontend (React + TypeScript)
```bash
cd frontend
cp .env.example .env  # set VITE_API_BASE_URL if needed
npm install
npm run dev
```
- Local URL: http://localhost:5173
- API endpoint used: POST {VITE_API_BASE_URL}/predict (default http://localhost:8000/predict)
- Metadata/health: GET {VITE_API_BASE_URL}/metadata, GET {VITE_API_BASE_URL}/health

## Tests
```bash
pytest tests/ -v
```

## Build
```bash
cd frontend && npm run build
```

## Results
See results/ and reports/.
