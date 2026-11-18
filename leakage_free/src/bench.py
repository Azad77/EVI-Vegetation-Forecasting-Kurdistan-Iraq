"""Leakage-free benchmark of all models for one governorate (5 seeds each).

Usage:
  python bench.py REGION                    # rule A: early stopping on last 12 training windows, then refit
  python bench.py REGION --fixed 100        # rule B: fixed 100 epochs on all 67 training windows
  python bench.py REGION --fixed 100 --baselines   # neural baselines (LSTM, CNN), fixed 100 epochs

Rule A, per model/seed:
  stage 1: train on the first 55 training windows, early-stop on the last 12 (chronological) -> best epoch E;
  stage 2: refit on all 67 training windows for E epochs (no callbacks); evaluate once on the 17 test windows.
Outputs (in EVI_OUT): bench_REGION.csv + preds_REGION.csv (rule A);
  benchfix100_REGION.csv + predsfix100_REGION.csv (rule B); basefix100_REGION.csv + predsbase100_REGION.csv (baselines).
"""
import argparse
import models
from common import *

ap = argparse.ArgumentParser(); ap.add_argument('region', choices=REGIONS)
ap.add_argument('--fixed', type=int, default=0, help='fixed number of epochs (0 = early-stopping rule)')
ap.add_argument('--baselines', action='store_true', help='run the neural baselines instead of the 9 primary models')
a = ap.parse_args(); single_thread()

data = load(); X, y, st, dates = prep(data[a.region], strict=True); mu, sd = st['evi']
Xtr, ytr, Xte, yte = split(X, y); t = yte * sd + mu
reg = models.REG_BASE if a.baselines else models.REG
rows, preds = [], []
for name, (b, lr) in reg.items():
    for seed in range(NSEED):
        if a.fixed:
            E = a.fixed
        else:
            tf.keras.utils.set_random_seed(seed)
            m = b(); m.compile(keras.optimizers.Adam(lr), 'mse')
            h = m.fit(Xtr[:-NV], ytr[:-NV], validation_data=(Xtr[-NV:], ytr[-NV:]), epochs=150, batch_size=16, verbose=0,
                      shuffle=True, callbacks=[keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
            E = int(np.argmin(h.history['val_loss'])) + 1
        tf.keras.utils.set_random_seed(seed)
        m = b(); m.compile(keras.optimizers.Adam(lr), 'mse'); m.fit(Xtr, ytr, epochs=E, batch_size=16, verbose=0, shuffle=True)
        p = m.predict(Xte, verbose=0).ravel() * sd + mu
        rows.append(dict(model=name, region=a.region, seed=seed, epochs=E, params=m.count_params(), **metrics(t, p)))
        preds.append(pd.DataFrame({'model': name, 'region': a.region, 'seed': seed, 'date': dates[-NTEST:], 'obs': t, 'pred': p}))
        print(rows[-1], flush=True)

tag = ('basefix%d' % a.fixed, 'predsbase%d' % a.fixed) if a.baselines else \
      (('benchfix%d' % a.fixed, 'predsfix%d' % a.fixed) if a.fixed else ('bench', 'preds'))
pd.DataFrame(rows).to_csv(out(f'{tag[0]}_{a.region}.csv'), index=False)
pd.concat(preds).to_csv(out(f'{tag[1]}_{a.region}.csv'), index=False)
print('DONE', a.region)
