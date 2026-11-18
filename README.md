# EVI Vegetation Forecasting — Kurdistan Region of Iraq

Deep learning-based forecasting of Enhanced Vegetation Index (EVI) for vegetation
health monitoring across the four governorates of the Kurdistan Region of Iraq
(Erbil, Duhok, Sulaymaniyah, Halabja), using Sentinel-2 imagery and TerraClimate
climate data (2016–2024).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20276859.svg)](https://doi.org/10.5281/zenodo.20276859)

This repository accompanies a manuscript that is currently under peer review. It contains the complete pipeline from data extraction through
model training, uncertainty quantification, and future forecasting.

> ## ⚠️ Correction (November 2026): leakage-free re-run
>
> During the IJRS minor revision we found that the original notebook let the **test set leak into training**:
> the test windows were used as `validation_data` for `EarlyStopping(restore_best_weights=True)` and
> `ReduceLROnPlateau`, and the scalers and gap-filling climatology were fitted on all 108 months, including the
> test period. All experiments were re-run with a leakage-free protocol (see below), and **the results reported in
> the revised manuscript come from [`leakage_free/`](leakage_free/)**. The original notebooks and outputs are kept,
> clearly marked, in [`archive/original_submission/`](archive/original_submission/) and should not be used.
>
> Effect of the fix (mean test R² over 4 governorates × 5 seeds, 17 test months Aug 2023 – Dec 2024):
>
> | | Original protocol | Leakage-free |
> |---|---|---|
> | BiLSTM (previous best model) | 0.945 (published) | 0.766 (early-stopping rule) / 0.911 (fixed 100 epochs) |
> | Best models | — | CNN-BiLSTM 0.923, CNN-BiLSTM-GRU 0.922, CNN-GRU 0.915 |
> | Climatological mean baseline | — | 0.778 |
>
> The CNN-recurrent and recurrent models keep clear skill over the seasonal cycle, but the accuracies are lower than
> first reported and differences among the leading models are within seed-to-seed variability.
> **CNN-BiLSTM-GRU**, ranked in the top two under both training rules, is now the reference model for the
> uncertainty analysis and the scenario forecasts.

---

## Leakage-free protocol (used for all reported results)

- **Split:** chronological. 24-month input windows; the last **17** targets (Aug 2023 – Dec 2024) form the test set,
  the preceding 67 windows the training set.
- **Preprocessing:** z-score scalers and the climatology used for gap-filling are fitted on **training months only**.
- **Training rule A (main tables):** train on the first 55 training windows, early-stop on the last 12
  (Aug 2022 – Jul 2023), then refit on all 67 training windows for the selected number of epochs.
- **Training rule B (robustness):** fixed 100 epochs on all 67 training windows, no callbacks.
- **Evaluation:** the test set is used once, for scoring only. 5 random seeds per model and governorate;
  results are reported as mean ± SD. Skill is also reported relative to a training-period climatology
  (MSE skill score, anomaly correlation), because R² is dominated by the seasonal cycle.
- **Uncertainty:** MC Dropout (dropout active, BatchNorm in inference mode), calibrated post hoc with a variance
  factor *k* fitted on the last 12 training windows (k ≈ 3.0 for CNN-BiLSTM-GRU); calibrated coverage 96–98%.
- **Additional analyses:** leave-one-governorate-out spatial transfer, input ablation, gap-filling sensitivity,
  and 2025–2030 climate-scenario forecasts (stationary, P −20%, T +2 °C, hot-dry, recurring 2022 drought).

See [`leakage_free/README.md`](leakage_free/README.md) for the scripts, outputs and how to re-run them.

---

## Repository structure

```
├── leakage_free/              # CORRECTED code and results used in the revised manuscript
│   ├── src/                   #   scripts (benchmark, baselines, transfer, ablation, MC dropout, scenarios)
│   ├── results/raw/           #   per-governorate outputs (all seeds, all test-month predictions)
│   ├── results/summary/       #   manuscript tables (Table 2, Table 4, Tables S2/S4, skill scores)
│   └── run_all.sh
├── archive/original_submission/  # SUPERSEDED notebooks and outputs (contain the test-set leakage)
├── data/          # Input datasets: monthly EVI + climate per governorate (2016–2024)
├── shapefiles/    # Governorate boundary shapefiles (Erbil, Duhok, Sulaymaniyah, Halabja)
├── requirements.txt           # environment of the original notebook (GEE extraction, maps)
├── LICENSE
└── README.md
```

The **data extraction** (Google Earth Engine), study-area maps and descriptive statistics are still done with the
original notebook (stages 1–4 below); those stages are not affected by the leakage. **All modelling, evaluation,
uncertainty and forecasting results must be taken from `leakage_free/`.**

## Requirements

- Python 3.10+
- A Google Earth Engine (GEE) account with an active project (free for research/education — [sign up here](https://earthengine.google.com))
- Recommended: run in [Google Colab](https://colab.research.google.com) (the notebook is written for and tested on Colab; local execution is possible but requires adjusting file paths — see below)

Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note on versions:** the pinned versions in `requirements.txt` reflect the
> environment used to produce the results reported in the manuscript. If you
> run into a dependency conflict, `pip install -r requirements.txt --no-deps`
> followed by resolving conflicts individually is usually the fastest fix.

## Setup

1. **Clone the repository** (or open `archive/original_submission/notebooks/EVI_Vegetation_Health_Forecasting_REVISED_01_08_2026.ipynb` in Google Colab for the GEE extraction stages).

2. **Google Earth Engine authentication.** The notebook's GEE initialization cell handles this automatically depending on environment (Colab auth, saved credentials, or service account), but for a first-time run:
   ```python
   import ee
   ee.Authenticate()   # opens a one-time browser login
   ee.Initialize(project="your-gee-project-id")
   ```
   Replace `GEE_PROJECT` in the notebook's GEE setup cell with your own GEE project ID.

3. **Shapefiles.** If running in Colab, upload the four governorate shapefiles from `shapefiles/` to `/content/` (the default `SHP_DIR` in the notebook), or update `SHP_DIR` to point to wherever you've placed them. All four `.shp` (plus accompanying `.dbf`/`.shx`/`.prj`) files must be present.

4. **Pre-extracted data (optional, faster start).** If you want to skip the GEE extraction step (which can take significant time depending on GEE quota) and go straight to modeling, the already-extracted `evi_climate_kurdistan_2016_2024.csv` is provided in `data/` — place it in your working directory and skip to Run Order step 3 below.

## Run order

### Corrected modelling pipeline (leakage-free)

```bash
pip install -r leakage_free/requirements.txt
bash leakage_free/run_all.sh          # all experiments (CPU, several hours)
python leakage_free/src/summarize.py  # rebuild the summary tables from the raw outputs only (seconds)
```

### Original notebook (superseded for stages 5–10)

> Stages 5–10 of the original notebook contain the leakage described above and are kept for reference only.

The notebook is organized into numbered cells; run them top to bottom. Approximate runtime is noted for the slower stages (on a standard Colab GPU runtime):

| Stage | Cells | Description | Approx. runtime |
|---|---|---|---|
| 1 | Install & imports | Package installation, global settings, seed fixing | < 1 min |
| 2 | Study area | Load governorate shapefiles, plot study area map | < 1 min |
| 3 | GEE extraction | Authenticate GEE, extract monthly Sentinel-2 EVI + TerraClimate precip/temp per governorate (2016–2024) | 20–60 min (skip if using provided CSV) |
| 4 | QC & gap-filling | Quality control, climatological gap-filling, wide-format reshaping, data quality report | < 2 min |
| 5 | Feature engineering | Z-score normalization, cyclic month encoding, sliding-window sequence creation (24-month windows), train/test split | < 1 min |
| 6 | Model training | Define and train all 9 architectures (BiLSTM, GRU, CNN-GRU, CNN-BiLSTM, BiLSTM-GRU, CNN-BiLSTM-GRU, TCN, Transformer, CNN-Transformer) across all 4 governorates | 30–90 min total (GPU strongly recommended) |
| 7 | Evaluation | Results table, comparison figures, loss curves, prediction plots | < 5 min |
| 8 | Uncertainty | MC-Dropout uncertainty estimation (100 forward passes), rolling cross-validation | 10–20 min |
| 9 | Forecasting | 5-year and 10-year stationary-climatology EVI forecasts, per-model and cross-model comparison plots | < 5 min |
| 10 | Save outputs | Export all CSVs and figures; optional Google Drive export cell (Colab only) | < 1 min |

Total end-to-end runtime (excluding GEE extraction): **approximately 1–2 hours** on a Colab GPU runtime (T4 or better recommended).

## Outputs

Corrected results (`leakage_free/results/summary/`):
- `Table2_leakfree_mean.csv`, `Table_by_model_region_leakfree.csv` — 9 models, rule A, mean ± SD over 5 seeds
- `Table2_fixed100_mean.csv`, `neural_baselines_fixed100.csv`, `baselines_17m.csv` — rule B and baselines
- `skill_vs_climatology.csv` — MSE skill score and anomaly R²/r against the training-period climatology
- `TableS2_picp_calibrated.csv`, `TableS4_logo_transfer.csv`, `ablation_summary.csv`, `gapfill_summary.csv`
- `scenario_forecasts_2025_2030.csv`, `Table4_scenarios.csv` — CNN-BiLSTM-GRU scenario forecasts with calibrated bands

Running the original notebook produces, among others (superseded where marked):
- `evi_model_metrics.csv` — MSE/MAE/R²/Accuracy for all 9 models × 4 governorates (**superseded**, leaky protocol; now in `archive/original_submission/data/`)
- `evi_historical_2016_2024.csv`, `precip_historical_2016_2024.csv`, `temp_historical_2016_2024.csv` — processed historical records
- `evi_forecasts_5year_2026_2030.csv`, `evi_forecasts_10year_2026_2035.csv` — stationary-climatology forecast outputs (**superseded**; replaced by `leakage_free/results/summary/scenario_forecasts_2025_2030.csv`)
- Figures: study area map, EVI/climate time series, annual statistics, model comparison, loss curves, prediction-vs-truth plots, MC-Dropout uncertainty bands, forecast plots

All outputs used in the revised manuscript's tables and figures are provided pre-computed in `leakage_free/results/` for readers who do not wish to re-run the full pipeline.

## Reproducibility notes

- Random seeds are fixed (`np.random.seed(42)`, `tf.random.set_seed(42)`) at notebook start; however, exact bitwise reproducibility of deep-learning training is not guaranteed across different hardware/driver/TensorFlow-version combinations, which is expected behavior for GPU-accelerated training.
- The leakage-free scripts fix seeds 0–4 with `tf.keras.utils.set_random_seed` and run single-threaded on CPU (TensorFlow 2.21).
- `leakage_free/src/leakage_check.py` reproduces the original (leaky) and corrected protocols side by side for BiLSTM.
- `leakage_free/src/summarize.py` regenerates every summary table from the raw outputs; it was checked to reproduce the manuscript numbers exactly.
- The original notebook was verified to run end-to-end in a clean environment as of 01/08/2026.

## Citation

If you use this code or data, please cite:

> Until the article is published, please cite the repository archive below. The article reference will be added here once it is available.

Repository archive: [https://doi.org/10.5281/zenodo.20276859](https://doi.org/10.5281/zenodo.20276859) (v1.0.0, original submission). The corrected, leakage-free code is archived from release **v2.0.0** onwards; please cite that version's DOI (listed under "Versions" on the Zenodo record).

## License

MIT License — see `LICENSE` for details.

## Contact

Dr. Azad Rasul — Salahaddin University-Erbil, Department of Forestry
azad.rasul@soran.edu.iq
