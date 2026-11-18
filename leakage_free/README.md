# Leakage-free re-run (IJRS revision, November 2026)

This folder holds the corrected code and every result reported in the revised manuscript. It replaces the
modelling stages (5–10) of the original notebook, which are archived in
[`../archive/original_submission/`](../archive/original_submission/).

## What was wrong and what changed

| | Original notebook | Leakage-free code (this folder) |
|---|---|---|
| Early stopping / LR schedule | monitored on the **test** windows (`validation_data=(X_test, y_test)`, `restore_best_weights=True`) | monitored on the last 12 **training** windows (Aug 2022 – Jul 2023); model then refit on all 67 training windows |
| Scalers, gap-fill climatology | fitted on all 108 months | fitted on training months only |
| Test set | described as 20 months from May 2023 | 17 windows, Aug 2023 – Dec 2024 (as actually evaluated) |
| Seeds | single seed (42) | 5 seeds (0–4), mean ± SD reported |
| MC Dropout | BatchNorm also in training mode; raw intervals | BatchNorm in inference mode; post-hoc variance calibration (k fitted on training data) |
| Reference model | BiLSTM | CNN-BiLSTM-GRU (top two under both training rules) |

## Scripts (`src/`)

| Script | Purpose | Raw output (`results/raw/`) |
|---|---|---|
| `common.py` | data loading, QC, leakage-free gap-filling/scaling, windowing | – |
| `models.py` | 9 architectures + LSTM/CNN baselines (MC-dropout switch) | – |
| `leakage_check.py` | BiLSTM under the original vs. clean protocol | `leakage_check_bilstm.csv` |
| `bench.py REGION` | rule A: early stopping on last 12 training windows, refit | `bench_*.csv`, `preds_*.csv` |
| `bench.py REGION --fixed 100` | rule B: fixed 100 epochs | `benchfix100_*.csv`, `predsfix100_*.csv` |
| `bench.py REGION --fixed 100 --baselines` | LSTM/CNN baselines | `basefix100_*.csv`, `predsbase100_*.csv` |
| `stat_baselines.py` | climatology, seasonal and naive persistence on the 17 test months | `summary/baselines_17m.csv` |
| `logo.py REGION` | leave-one-governorate-out transfer | `logo_*.csv` |
| `ablation.py REGION` | input ablation (no P, no T, no season, EVI only, …) | `ablation_*.csv` |
| `gapfill.py REGION` | gap-filling sensitivity with matched random-drop control | `gapfill_*.csv` |
| `mc.py REGION` | MC Dropout + calibration data | `mc_*.csv` |
| `scenarios.py REGION` | 2025–2030 climate-scenario forecasts, CNN-BiLSTM-GRU | `scen_raw_*.csv` |
| `summarize.py` | builds all tables in `results/summary/` from the raw files | `results/summary/*.csv` |

`REGION` is one of `Erbil`, `Duhok`, `Sulaymaniyah`, `Halabja`. Input data default to
`../data/evi_climate_kurdistan_2016_2024.csv`; set `EVI_CSV` / `EVI_OUT` to change the input file or output folder.

## Re-running

```bash
pip install -r requirements.txt
bash run_all.sh                 # everything; CPU, several hours
python src/summarize.py         # tables only, from the provided raw outputs (seconds, no TensorFlow needed)
```

The raw outputs in `results/raw/` were produced with TensorFlow 2.21 on CPU, one thread per process. Neural-network
training is not bit-for-bit reproducible across hardware and library versions, so a re-run elsewhere may give slightly
different numbers per seed; the 5-seed means should agree closely. (With TensorFlow 2.21 on CPU, a spot check
reproduced the stored CNN-BiLSTM-GRU and LSTM-baseline results for Erbil, seed 0, exactly.)

## Key results (mean over 4 governorates; SD across 5 seeds)

| Model | Test R², rule A | Test R², rule B (100 epochs) |
|---|---|---|
| CNN-BiLSTM | 0.923 ± 0.010 | 0.909 ± 0.019 |
| CNN-BiLSTM-GRU (reference) | 0.922 ± 0.007 | 0.923 ± 0.013 |
| CNN-GRU | 0.915 ± 0.013 | 0.909 ± 0.016 |
| GRU | 0.845 ± 0.042 | 0.881 ± 0.010 |
| TCN | 0.796 ± 0.048 | 0.854 ± 0.018 |
| BiLSTM-GRU | 0.769 ± 0.036 | 0.923 ± 0.008 |
| BiLSTM | 0.766 ± 0.027 | 0.911 ± 0.015 |
| CNN-Transformer | 0.340 ± 0.276 | 0.765 ± 0.070 |
| Transformer | −0.054 ± 0.079 | −0.073 ± 0.300 |
| *Climatological mean (statistical baseline)* | *0.778* | |

- Calibrated MC-Dropout coverage (nominal 95.4%): CNN-BiLSTM-GRU 96.8% (k = 3.0), BiLSTM-GRU 98.2%,
  CNN-Transformer 95.9%; raw coverage was only 49–69% (`TableS2_picp_calibrated.csv`).
- Leave-one-governorate-out transfer R²: BiLSTM-GRU 0.898, CNN-BiLSTM-GRU 0.890, CNN-GRU 0.866 (`TableS4_logo_transfer.csv`).
