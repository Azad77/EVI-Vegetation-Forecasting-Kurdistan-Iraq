"""Statistical baselines on the same 17 test months (no training needed).
Climatology = training-period monthly mean EVI; seasonal persistence = EVI 12 months earlier;
naive persistence = last training-period EVI (Jul 2023) held constant over the test period
(multi-step horizon, no test observations used). Series after leakage-free gap-filling.
Output: ../results/summary/baselines_17m.csv
"""
import os
from common import *
data = load(); rows = []
for r in REGIONS:
    X, y, st, dates = prep(data[r], strict=True); mu, sd = st['evi']
    evi = pd.Series(np.concatenate([X[0, :, 0], y]) * sd + mu, index=pd.date_range(dates[0] - pd.DateOffset(months=SEQ), periods=len(y) + SEQ, freq='MS'))
    te = evi.iloc[-NTEST:]; tr = data[r].iloc[:-NTEST]; clim = tr.evi.groupby(tr.index.month).mean()
    for name, p in [('Climatology', np.array([clim[d.month] for d in te.index])),
                    ('SeasonalPersistence', evi.shift(12).iloc[-NTEST:].values),
                    ('NaivePersistence', np.full(NTEST, evi.iloc[-NTEST - 1]))]:
        mt = metrics(te.values, p); rows.append(dict(model=name, region=r, R2=mt['R2'], MAE=mt['MAE'], Acc=mt['Acc']))
o = pd.DataFrame(rows); print(o.round(3))
o.to_csv(os.path.join(REPO, 'leakage_free', 'results', 'summary', 'baselines_17m.csv'), index=False)
