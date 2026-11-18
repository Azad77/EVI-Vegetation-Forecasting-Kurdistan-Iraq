"""Shared data loading and leakage-free preprocessing.

Leakage-free rules (applied everywhere unless a script states otherwise):
  * the last NTEST=17 monthly targets (Aug 2023 - Dec 2024) form the test set;
  * gap-filling climatology and z-score scalers are fitted on training months only;
  * the test windows are never used for early stopping, learning-rate schedules,
    calibration or model selection.

Paths can be overridden with the environment variables EVI_CSV and EVI_OUT.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
CSV = os.environ.get('EVI_CSV', os.path.join(REPO, 'data', 'evi_climate_kurdistan_2016_2024.csv'))
OUT = os.environ.get('EVI_OUT', os.path.join(REPO, 'leakage_free', 'results', 'raw'))
os.makedirs(OUT, exist_ok=True)

REGIONS = ['Erbil', 'Duhok', 'Sulaymaniyah', 'Halabja']
SEQ = 24     # input window length (months)
NTEST = 17   # test targets: Aug 2023 - Dec 2024
NV = 12      # last 12 training windows (Aug 2022 - Jul 2023) used for early stopping / calibration
NSEED = 5


def single_thread():
    """Deterministic-ish CPU execution (the reported runs used one thread per process)."""
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)


def load():
    """Monthly EVI/precipitation/temperature per governorate, with QC (valid_pixels >= 50, -0.1 <= EVI <= 0.8)."""
    df = pd.read_csv(CSV); df['date'] = pd.to_datetime(df['date'])
    out = {}
    for r in REGIONS:
        s = df[df.region == r].sort_values('date').set_index('date')
        e = s.evi_median.copy(); e[(s.valid_pixels < 50) | (e < -0.1) | (e > 0.8)] = np.nan
        out[r] = pd.DataFrame({'evi': e,
                               'precip': s.precip_mm.interpolate(limit_direction='both'),
                               'temp': s.temp_c.interpolate(limit_direction='both')})
    return out


def prep(d, strict=True):
    """Gap-fill, scale and window one governorate.

    strict=True  -> climatology and scalers from the training months only (leakage-free evaluation).
    strict=False -> use the whole 2016-2024 record (only for the final forecasting model, which has no test set).
    Returns X (n_windows, SEQ, 5), y (n_windows,), scaler stats {var: (mean, sd)}, target dates.
    Features: EVI, precipitation, temperature, sin(month), cos(month).
    """
    d = d.copy(); n = len(d); ntr_months = n - NTEST
    tr_mask = np.arange(n) < ntr_months if strict else np.ones(n, bool)
    clim = d.evi[tr_mask].groupby(d.index.month[tr_mask]).mean()
    miss = d.evi.isna()
    d.loc[miss, 'evi'] = [clim.get(m, np.nan) for m in d.index.month[miss]]
    d['evi'] = d.evi.interpolate(limit_direction='both')
    stats = {}
    for c in ['evi', 'precip', 'temp']:
        mu, sd = d[c][tr_mask].mean(), d[c][tr_mask].std(ddof=0)
        stats[c] = (mu, sd); d[c] = (d[c] - mu) / sd
    m = d.index.month.values
    F = np.column_stack([d.evi, d.precip, d.temp, np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)])
    X = np.array([F[i:i + SEQ] for i in range(n - SEQ)]); y = np.array([F[i + SEQ, 0] for i in range(n - SEQ)])
    return X, y, stats, d.index[SEQ:]


def split(X, y):
    return X[:-NTEST], y[:-NTEST], X[-NTEST:], y[-NTEST:]


def metrics(t, p):
    return dict(R2=r2_score(t, p), MAE=mean_absolute_error(t, p), MSE=mean_squared_error(t, p),
                Acc=float(np.mean(np.abs(p - t) <= 0.05)))


def out(name):
    return os.path.join(OUT, name)
