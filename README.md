<<<<<<< HEAD
# ResQ AI — QUANTEX

Starter repository for the ResQ AI disaster-response project.

## Current starter scope
- Optionally reads EM-DAT and HumAID CSV files from `data/raw/`.
- Fetches recent earthquake events from the USGS GeoJSON feed.
- Writes normalized source data to `data/processed/`.
- Does not generate severity or NLP predictions; those belong to later, validated model stages.

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.pipeline
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again.

## Optional input files
- `data/raw/emdat_india.csv` — EM-DAT export. The pipeline keeps the available columns and attempts to normalize common fields.
- `data/raw/humAID.csv` — HumAID dataset export. The pipeline saves a cleaned copy if present.

Check dataset licenses, access conditions, and privacy requirements before committing any raw data to GitHub. Keep raw datasets out of Git by default.

## Outputs
CSV files are written to `data/processed/`. The USGS feed is public and refreshed when the pipeline runs.
=======
# QUANTEX-ResQ-AI
AI-powered disaster response optimization using multi-source data integration, severity classification, and humanitarian intelligence
>>>>>>> ae625519db694bc4fa0ecc480bedf6aca2afdde9
