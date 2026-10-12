# neural-network-final-project

Final project for the chronic pain prediction benchmark. This project includes a starter script, a results explorer UI, and synthetic [Synthea](https://synthea.mitre.org/downloads) patient data. Students will work with a baseline model, complete parts of the modeling pipeline, run experiments, compare CPU and GPU model results, and reflect on reproducibility practices.

## Final Project Overview

This final project is part of the AI Sandbox learning environment. The goal is to give students a hands-on experience with a realistic AI workflow using synthetic healthcare data. Students will start with a provided baseline model and then try to improve its performance through feature engineering, model selection, parameter tuning, error analysis, and evaluation.

The project includes both code-based and UI-based components. Students will complete the starter code, run their model, generate metrics across multiple iterations, and use the explorer UI to compare their results with benchmark models. The explorer helps students inspect model performance, compare tradeoffs, and understand why reproducibility matters in AI experiments.

Students’ goal is to beat the provided baseline performance and earn a place on the project leaderboard. Leaderboard position and beating the baseline will be considered as part of the final project grade, along with code completion, model comparison, reproducibility reflection, and explanation of design choices.

Students are encouraged to be ambitious when building their models. They may use any neural network architecture or modeling approach they want, as long as it can beat the baseline and be explained clearly. The goal is not only to get a higher score, but also to understand why the model performs better and whether the result is reproducible.

In the final report, students will explain their design choices, summarize their findings, compare their model with the provided benchmarks, and answer directed research questions about model performance, reproducibility, computational cost, and tradeoffs.

This project is designed to make the AI modeling process visible. Students do not only train a model. They compare results, examine variation across iterations, think about CPU and GPU tradeoffs, and explain their choices. The use of synthetic Synthea data also allows students to work with realistic healthcare-style data without using real patient information. They can also compare model performance with computational cost and reflect on what makes an AI experiment reproducible.

## Explorer Link

The current project explorer is available here:

https://ai-sandbox-ai-healthcare.github.io/neural-network-final-project/

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
├── my_model.csv                       your model's results, one row per run (5 runs), read by the explorer
├── final_project_starter.py            the assignment -- edit build_features()/build_model()
├── run_explorer.py                     Flask app, results explorer UI (localhost:3002)
├── start.bat                           Windows shortcut: launches run_explorer.py
├── build_docs.py                       renders the explorer to docs/index.html for GitHub Pages
└── docs/                               static build output, served by GitHub Pages
```

## Running the explorer locally

```bash
python run_explorer.py
```

On Windows, double-click start.bat to do the same thing.

The explorer opens at:
http://localhost:3002

## TODO

Jetstream2 infrastructure development is in progress.
