"""Climate-scenario forecasts 2025-2030 (Referee 2, Point 5) with CNN-BiLSTM-GRU.
The forecasting model is trained on all 84 windows of 2016-2024 (no test set here, so strict=False), 5 seeds,
100 epochs. Future climate = 2016-2024 monthly climatology with delta changes. Recursive 72-month forecasts with
100 MC-dropout trajectories per seed. Bands (summarize.py) = mean +/- 2*k*sd with k = 3.0 from the calibration step.
Usage: python scenarios.py REGION -> scen_raw_REGION.csv"""
import sys
import models
models.MC_DROPOUT = True
from common import *
single_thread()
region = sys.argv[1]; NMC = 100; H = 72
data = load(); d = data[region].copy()
X, y, st, dates = prep(d, strict=False)
clim = d.copy(); clim['evi'] = clim.evi.fillna(clim.evi.groupby(clim.index.month).transform('mean')).interpolate(limit_direction='both')
F = np.column_stack([(clim[c] - st[c][0]) / st[c][1] for c in ['evi', 'precip', 'temp']])
m_ = clim.index.month.values; F = np.column_stack([F, np.sin(2 * np.pi * m_ / 12), np.cos(2 * np.pi * m_ / 12)])
cp = d.precip.groupby(d.index.month).mean(); ct = d.temp.groupby(d.index.month).mean()
p22 = d.precip['2022']; t22 = d.temp['2022']
fdates = pd.date_range('2025-01-01', periods=H, freq='MS')
def climate(scn, date):
    mo = date.month; P, T = cp[mo], ct[mo]
    if scn == 'P-20%': P *= 0.8
    if scn == 'T+2C': T += 2
    if scn == 'Hot-dry (P-20%, T+2C)': P *= 0.8; T += 2
    if scn == 'Drought 2022 recurring 2027' and date.year == 2027: P, T = p22.iloc[mo - 1], t22.iloc[mo - 1]
    return (P - st['precip'][0]) / st['precip'][1], (T - st['temp'][0]) / st['temp'][1]
SC = ['Stationary climatology', 'P-20%', 'T+2C', 'Hot-dry (P-20%, T+2C)', 'Drought 2022 recurring 2027']
b, lr = models.REG['CNN_BiLSTM_GRU']; rows = []
for seed in range(NSEED):
    tf.keras.utils.set_random_seed(seed); m = b(); m.compile(keras.optimizers.Adam(lr), 'mse')
    m.fit(X, y, epochs=100, batch_size=16, verbose=0, shuffle=True)
    for scn in SC:
        W = np.repeat(F[-SEQ:][None], NMC, 0); traj = []
        for dt in fdates:
            yh = m(W[:, -SEQ:], training=False).numpy().ravel()
            pz, tz = climate(scn, dt); mo = dt.month
            nxt = np.column_stack([yh, np.full(NMC, pz), np.full(NMC, tz), np.full(NMC, np.sin(2 * np.pi * mo / 12)), np.full(NMC, np.cos(2 * np.pi * mo / 12))])
            W = np.concatenate([W, nxt[:, None, :]], 1); traj.append(yh * st['evi'][1] + st['evi'][0])
        traj = np.array(traj).T
        for j, dt in enumerate(fdates):
            rows.append(dict(region=region, seed=seed, scenario=scn, date=dt, mean=traj[:, j].mean(), sd=traj[:, j].std()))
    print(seed, flush=True)
pd.DataFrame(rows).to_csv(out(f'scen_raw_{region}.csv'), index=False); print('DONE')
