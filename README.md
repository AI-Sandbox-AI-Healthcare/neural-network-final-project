# neural-network-final-project

Final project for the chronic-pain-prediction benchmark: a starter script to
beat a Random Forest baseline, a results explorer UI, and the raw Synthea
patient data both scripts read from.

## Layout

```
neural-network-final-project/
├── data/                          Synthea patient data (synthetic, no real PHI)
│   ├── patients.csv               6 raw CSVs, as exported from Synthea
│   ├── conditions.csv
│   ├── encounters.csv
│   ├── medications.csv
│   ├── procedures.csv
│   ├── observations.csv
│   └── synthea_enriched_features.csv   generated the first time you run
│                                        run_explorer.py or build_docs.py
│                                        (see "Data loading" below)
├── my_model_over_30_iterations.csv     your model's metrics, read by the explorer
├── final_project_starter.py            the assignment -- edit build_features()/build_model()
├── run_explorer.py                     Flask app, results explorer UI (localhost:3002)
├── start.bat                           Windows shortcut: launches run_explorer.py
├── build_docs.py                       renders the explorer to docs/index.html for GitHub Pages
└── docs/                               static build output, served by GitHub Pages
```

## Data loading

`data/synthea_enriched_features.csv` (45 pre-engineered features + `binary_label`)
is the real dataset the evaluation harness and explorer's dataset stats are
meant to run against, but it isn't part of this repo -- only the 6 raw
Synthea CSVs are.

Both `run_explorer.py` and `final_project_starter.py` follow the same rule:
**if `data/synthea_enriched_features.csv` exists, load it and use the real
data.** If it doesn't, `run_explorer.py` builds a small 20-row stand-in from
the real `patients.csv`/`conditions.csv` (age, gender, race, ethnicity,
healthcare expenses/coverage, and a keyword-match chronic-pain label) and
**saves it** to that path, so the explorer's dataset stats always reflect
real (if limited) data instead of a hardcoded placeholder.

That 20-row stand-in only has enough columns for the explorer's aggregate
stats -- it does **not** have the 24 baseline features (`num_conditions`,
`pain_severity`, vitals, etc.) that `final_project_starter.py`'s
`build_features()` requires. Running the starter script against it will
raise a clear `KeyError` naming the missing columns; that's expected and
correct -- it means you need to drop the real, fully engineered
`synthea_enriched_features.csv` into `data/` before running the actual
evaluation.

## Running the explorer locally

```bash
python run_explorer.py
```

On Windows, double-click `start.bat` to do the same thing.

Opens at `http://localhost:3002`. Every request re-reads
`my_model_over_30_iterations.csv`, so re-running your model and refreshing
the page shows updated metrics without restarting the server.

## Running the starter script

```bash
python final_project_starter.py
```

## GitHub Pages

https://ai-sandbox-ai-healthcare.github.io/neural-network-final-project/
