# EVI Vegetation Forecasting — Kurdistan Region of Iraq

Deep learning-based forecasting of Enhanced Vegetation Index (EVI) for vegetation
health monitoring across the four governorates of the Kurdistan Region of Iraq
(Erbil, Duhok, Sulaymaniyah, Halabja), using Sentinel-2 imagery and TerraClimate
climate data (2016–2024).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20276859.svg)](https://doi.org/10.5281/zenodo.20276859)

This repository accompanies the manuscript *"[manuscript title]"* (submitted to
*[journal name]*). It contains the complete pipeline from data extraction through
model training, uncertainty quantification, and future forecasting.

---

## Repository structure

```
├── notebooks/     # Main analysis notebook (Google Colab)
├── data/          # Processed CSV datasets (EVI, climate, model outputs, forecasts)
├── shapefiles/    # Governorate boundary shapefiles (Erbil, Duhok, Sulaymaniyah, Halabja)
├── requirements.txt
├── LICENSE
└── README.md
```

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

1. **Clone the repository** (or open `notebooks/EVI_Vegetation_Health_Forecasting.ipynb` directly in Google Colab via GitHub → Colab integration).

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

Running the full notebook produces, among others:
- `evi_model_metrics.csv` — MSE/MAE/R²/Accuracy for all 9 models × 4 governorates
- `evi_historical_2016_2024.csv`, `precip_historical_2016_2024.csv`, `temp_historical_2016_2024.csv` — processed historical records
- `evi_forecasts_5year_2026_2030.csv`, `evi_forecasts_10year_2026_2035.csv` — stationary-climatology forecast outputs
- Figures: study area map, EVI/climate time series, annual statistics, model comparison, loss curves, prediction-vs-truth plots, MC-Dropout uncertainty bands, forecast plots

All outputs used in the manuscript's tables and figures are also provided pre-computed in `data/` for readers who do not wish to re-run the full pipeline.

## Reproducibility notes

- Random seeds are fixed (`np.random.seed(42)`, `tf.random.set_seed(42)`) at notebook start; however, exact bitwise reproducibility of deep-learning training is not guaranteed across different hardware/driver/TensorFlow-version combinations, which is expected behavior for GPU-accelerated training.
- This repository and its contents were verified to run end-to-end in a clean environment as of 01/08/2026.

## Citation

If you use this code or data, please cite:

> [Author names]. ([Year]). *[Manuscript title]*. [Journal name]. [DOI once assigned]

Repository archive: [https://doi.org/10.5281/zenodo.20276859](https://doi.org/10.5281/zenodo.20276859)

## License

MIT License — see `LICENSE` for details.

## Contact

Dr. Azad Rasul — Salahaddin University-Erbil, Department of Forestry
azad.rasul@soran.edu.iq
