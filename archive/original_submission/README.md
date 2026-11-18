# Original-submission material (superseded)

These notebooks and outputs are kept **only for transparency**. They were used for the first
submission and earlier revisions and **must not be used to reproduce the reported results**.

They contain a test-set leakage problem identified during the IJRS minor revision (November 2026):

1. `train_model()` passed the **test windows** as `validation_data` to `EarlyStopping(restore_best_weights=True)`
   and `ReduceLROnPlateau`, so the test set influenced both when training stopped and which weights were kept.
2. The z-score scalers and the climatological gap-filling were fitted on **all 108 months**, including the test period.
3. The test period was described as 20 months starting May 2023; the code actually evaluated **17 windows
   (Aug 2023 - Dec 2024)**.

As a result, `data/evi_model_metrics.csv` (e.g. BiLSTM R² ≈ 0.945) overstates test skill, and the forecasts in
`data/evi_forecasts_*` come from models selected with the leaky protocol.

The corrected code and results are in [`../../leakage_free/`](../../leakage_free/).
`leakage_free/src/leakage_check.py` reproduces the old numbers with the old protocol and shows the effect of the fix.
