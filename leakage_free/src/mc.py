"""MC Dropout + post-hoc variance calibration (Referee 2, Point 4 / Table S2).
Dropout active at inference, BatchNorm in inference mode, 50 passes, fixed 100 epochs, 5 seeds.
  fit A: model on all 67 training windows -> raw PICP on the test set (and 5-seed deep ensemble).
  fit B: model on the first 55 windows -> MC on the last 12 training windows ('cal') and on the test set;
         k = sqrt(mean(z^2)), z = (obs - mean)/sd on 'cal'; calibrated interval = mean +/- 2*k*sd (see summarize.py).
Usage: python mc.py REGION -> mc_REGION.csv"""
import sys
import models
models.MC_DROPOUT = True
from common import *
single_thread()
region = sys.argv[1]; NMC = 50; EP = 100
data = load(); X, y, st, dates = prep(data[region], strict=True); mu, sd = st['evi']
Xtr, ytr, Xte, yte = split(X, y)
def mc(m, Xin): return np.stack([m(Xin, training=False).numpy().ravel() for _ in range(NMC)]) * sd + mu
rows = []
for name in ['BiLSTM_GRU', 'CNN_BiLSTM_GRU', 'CNN_Transformer']:
    b, lr = models.REG[name]
    for seed in range(NSEED):
        for fit in ['A', 'B']:
            tf.keras.utils.set_random_seed(seed); m = b(); m.compile(keras.optimizers.Adam(lr), 'mse')
            n = len(Xtr) if fit == 'A' else len(Xtr) - NV
            m.fit(Xtr[:n], ytr[:n], epochs=EP, batch_size=16, verbose=0, shuffle=True)
            sets = [('test', Xte, yte, dates[-NTEST:])] + ([('cal', Xtr[-NV:], ytr[-NV:], dates[-NTEST - NV:-NTEST])] if fit == 'B' else [])
            for nm, Xs, ys, ds in sets:
                P = mc(m, Xs)
                for j in range(len(ys)):
                    rows.append(dict(model=name, region=region, seed=seed, fit=fit, set=nm, date=ds[j], obs=ys[j] * sd + mu,
                                     mean=P[:, j].mean(), sd=P[:, j].std()))
        print(name, seed, flush=True)
pd.DataFrame(rows).to_csv(out(f'mc_{region}.csv'), index=False); print('DONE')
