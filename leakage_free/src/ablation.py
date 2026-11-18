"""Input ablation (Referee 2, Point 7): retrain CNN-BiLSTM-GRU and BiLSTM-GRU with feature subsets.
Leakage-free, fixed 100 epochs, 5 seeds. Usage: python ablation.py REGION  -> ablation_REGION.csv"""
import sys
import models
from common import *
single_thread()
region = sys.argv[1]; data = load(); X, y, st, dates = prep(data[region], strict=True); mu, sd = st['evi']
# feature columns: 0 EVI, 1 precipitation, 2 temperature, 3 sin(month), 4 cos(month)
V = {'Full (EVI+P+T+season)': [0, 1, 2, 3, 4], 'No precipitation': [0, 2, 3, 4], 'No temperature': [0, 1, 3, 4],
     'No climate (EVI+season)': [0, 3, 4], 'No seasonal encoding': [0, 1, 2], 'EVI history only': [0],
     'No EVI history (P+T+season)': [1, 2, 3, 4]}
tr = data[region].iloc[:-NTEST]; clim = tr.evi.groupby(tr.index.month).mean()
c = np.array([clim[d.month] for d in dates[-NTEST:]])
rows = []
for name in ['CNN_BiLSTM_GRU', 'BiLSTM_GRU']:
    b, lr = models.REG[name]
    for vn, cols in V.items():
        models.NF = len(cols); Xtr, ytr, Xte, yte = split(X[:, :, cols], y); t = yte * sd + mu
        for seed in range(NSEED):
            tf.keras.utils.set_random_seed(seed); m = b(); m.compile(keras.optimizers.Adam(lr), 'mse')
            m.fit(Xtr, ytr, epochs=100, batch_size=16, verbose=0, shuffle=True)
            p = m.predict(Xte, verbose=0).ravel() * sd + mu
            rows.append(dict(model=name, variant=vn, region=region, seed=seed, R2=r2_score(t, p), MAE=mean_absolute_error(t, p),
                             MSESS=1 - ((t - p) ** 2).mean() / ((t - c) ** 2).mean(), anom_r=np.corrcoef(t - c, p - c)[0, 1]))
        print(name, vn, flush=True)
pd.DataFrame(rows).to_csv(out(f'ablation_{region}.csv'), index=False); print('DONE')
