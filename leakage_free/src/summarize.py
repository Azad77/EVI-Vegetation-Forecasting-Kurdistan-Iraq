"""Builds the summary tables in ../results/summary from the raw per-region outputs in ../results/raw.
Pure pandas/numpy (no TensorFlow needed).

  Table2_leakfree_mean.csv            rule A (early stopping on last 12 training windows): mean over governorates, SD across seeds
  Table_by_model_region_leakfree.csv  rule A per model x governorate (mean over 5 seeds; epochs = median)
  Table2_fixed100_mean.csv            rule B (fixed 100 epochs) robustness check
  neural_baselines_fixed100.csv       LSTM/CNN baselines, fixed 100 epochs
  skill_vs_climatology.csv            MSESS, anomaly R2, anomaly r per model/region/seed (rule B predictions)
  TableS4_logo_transfer.csv           leave-one-governorate-out transfer R2
  TableS2_picp_calibrated.csv         raw / ensemble / calibrated MC-dropout coverage
  ablation_summary.csv, gapfill_summary.csv
  scenario_forecasts_2025_2030.csv    seed-pooled scenario forecasts with calibrated bands (k from TableS2)
  Table4_scenarios.csv                2026-2030 mean and April-mean EVI per governorate and scenario
"""
import os, numpy as np, pandas as pd
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
RAW = os.path.join(REPO, 'leakage_free', 'results', 'raw')
SUM = os.path.join(REPO, 'leakage_free', 'results', 'summary'); os.makedirs(SUM, exist_ok=True)
CSV = os.environ.get('EVI_CSV', os.path.join(REPO, 'data', 'evi_climate_kurdistan_2016_2024.csv'))
R = ['Erbil', 'Duhok', 'Sulaymaniyah', 'Halabja']; NTEST = 17
rd = lambda p: pd.concat(pd.read_csv(os.path.join(RAW, f'{p}_{r}.csv')) for r in R)
save = lambda df, n, **kw: (df.to_csv(os.path.join(SUM, n), **kw), print('wrote', n))

def model_table(B):
    by = B.groupby(['model', 'region']).agg(R2_mean=('R2', 'mean'), R2_sd=('R2', 'std'), MAE=('MAE', 'mean'), MSE=('MSE', 'mean'),
                                            Acc=('Acc', 'mean'), epochs=('epochs', 'median'), params=('params', 'first')).round(4).reset_index()
    # mean over governorates; SD across the 5 seeds of the governorate-averaged score
    reg = B.groupby(['model', 'seed'])[['R2', 'MAE', 'MSE', 'Acc']].mean().groupby('model').agg(['mean', 'std'])
    reg.columns = [f'{a}_{b}' for a, b in reg.columns]
    return by, reg.round(4).sort_values('R2_mean', ascending=False).reset_index()

# 1. benchmark tables
by, mean = model_table(rd('bench')); save(by, 'Table_by_model_region_leakfree.csv', index=False); save(mean, 'Table2_leakfree_mean.csv', index=False)
_, mean_fix = model_table(rd('benchfix100')); save(mean_fix, 'Table2_fixed100_mean.csv', index=False)
_, base = model_table(rd('basefix100')); save(base, 'neural_baselines_fixed100.csv', index=False)

# 2. skill beyond the seasonal cycle (training-period climatology)
D = pd.read_csv(CSV); D['date'] = pd.to_datetime(D['date'])
def clim(r):
    s = D[D.region == r].sort_values('date').set_index('date'); e = s.evi_median.copy()
    e[(s.valid_pixels < 50) | (e < -0.1) | (e > 0.8)] = np.nan; tr = e.iloc[:-NTEST]
    return tr.groupby(tr.index.month).mean()
C = {r: clim(r) for r in R}
P = rd('predsfix100'); P['date'] = pd.to_datetime(P.date)
rows = []
for (m, r, s), g in P.groupby(['model', 'region', 'seed']):
    t, p = g.obs.values, g.pred.values; c = np.array([C[r][d.month] for d in g.date])
    rows.append(dict(model=m, region=r, seed=s, MSESS=1 - ((t - p) ** 2).mean() / ((t - c) ** 2).mean(),
                     anomR2=r2_score(t - c, p - c), anomCorr=np.corrcoef(t - c, p - c)[0, 1]))
save(pd.DataFrame(rows), 'skill_vs_climatology.csv', index=False)

# 3. spatial transfer, ablation, gap-filling
L = rd('logo'); save(L.groupby(['model', 'target']).R2.agg(['mean', 'std']).round(3).unstack('target'), 'TableS4_logo_transfer.csv')
A = rd('ablation'); save(A.groupby(['model', 'variant'])[['R2', 'MAE', 'MSESS', 'anom_r']].mean().round(3), 'ablation_summary.csv')
G = rd('gapfill'); save(G.groupby(['region', 'variant']).R2.agg(['mean', 'std']).round(3), 'gapfill_summary.csv')

# 4. MC dropout coverage + variance calibration (k pooled over the 12 calibration months x 4 governorates x 5 seeds)
M = rd('mc'); M['sd'] = M.sd.clip(lower=1e-6); rows = []; K = {}
for mod, g in M.groupby('model'):
    a = g[(g.fit == 'A') & (g.set == 'test')]
    e = a.groupby(['region', 'date']).agg(obs=('obs', 'first'), m=('mean', 'mean'), v=('sd', lambda s: (s ** 2).mean()),
                                          vm=('mean', lambda s: s.var(ddof=0))).reset_index()
    es = np.sqrt(e.v + e.vm)
    cal = g[(g.fit == 'B') & (g.set == 'cal')]; k = np.sqrt((((cal.obs - cal['mean']) / cal.sd) ** 2).mean()); K[mod] = k
    b = g[(g.fit == 'B') & (g.set == 'test')]
    rows.append(dict(model=mod, raw_PICP=(abs(a.obs - a['mean']) <= 2 * a.sd).mean(), raw_width=(4 * a.sd).mean(),
                     ens_PICP=(abs(e.obs - e.m) <= 2 * es).mean(), ens_width=(4 * es).mean(), k=k,
                     cal_PICP=(abs(b.obs - b['mean']) <= 2 * k * b.sd).mean(), cal_width=(4 * k * b.sd).mean()))
save(pd.DataFrame(rows).round(3), 'TableS2_picp_calibrated.csv', index=False)

# 5. scenario forecasts: pool 5 seeds x 100 MC trajectories; band = mean +/- 2*k*sd with k for CNN-BiLSTM-GRU rounded to 3.0
S = rd('scen_raw'); kf = round(K['CNN_BiLSTM_GRU'], 1)
F = S.groupby(['region', 'scenario', 'date']).apply(lambda h: pd.Series(dict(
    mean=h['mean'].mean(), sd=np.sqrt((h.sd ** 2).mean() + h['mean'].var(ddof=0))))).reset_index()
F['lo'] = F['mean'] - 2 * kf * F.sd; F['hi'] = F['mean'] + 2 * kf * F.sd
save(F, 'scenario_forecasts_2025_2030.csv', index=False)
F['date'] = pd.to_datetime(F.date); f = F[F.date.dt.year >= 2026]
T4 = pd.concat({'mean': f.groupby(['region', 'scenario'])['mean'].mean().unstack(),
                'april': f[f.date.dt.month == 4].groupby(['region', 'scenario'])['mean'].mean().unstack()}, axis=1).round(3)
save(T4, 'Table4_scenarios.csv')
