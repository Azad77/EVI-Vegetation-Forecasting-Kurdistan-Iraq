"""Gap-filling sensitivity (Table S1), leakage-free: CNN-BiLSTM-GRU, fixed 100 epochs, 5 seeds.
Full training set vs. excluding windows that touch a gap-filled month vs. dropping the same number of random windows.
Usage: python gapfill.py REGION -> gapfill_REGION.csv"""
import sys
import models
from common import *
single_thread()
region = sys.argv[1]; data = load(); X, y, st, dates = prep(data[region], strict=True); mu, sd = st['evi']
gap = data[region].evi.isna().values; n = len(gap)
flag = np.array([gap[i:i + SEQ + 1].any() for i in range(n - SEQ)])
Xtr, ytr, Xte, yte = split(X, y); ftr = flag[:-NTEST]; t = yte * sd + mu
assert not flag[-NTEST:].any()
b, lr = models.REG['CNN_BiLSTM_GRU']
def run(idx, seed):
    tf.keras.utils.set_random_seed(seed); m = b(); m.compile(keras.optimizers.Adam(lr), 'mse')
    m.fit(Xtr[idx], ytr[idx], epochs=100, batch_size=16, verbose=0, shuffle=True)
    return r2_score(t, m.predict(Xte, verbose=0).ravel() * sd + mu)
allidx = np.arange(len(Xtr)); keep = allidx[~ftr]; nd = int(ftr.sum()); rng = np.random.default_rng(100)
rows = []
for s in range(NSEED):
    rows.append(('full', s, run(allidx, s))); rows.append(('excl_gapfilled', s, run(keep, s)))
    drop = rng.choice(keep, nd, replace=False); rows.append(('random_drop', s, run(np.setdiff1d(allidx, drop), s)))
    print(rows[-3:], flush=True)
o = pd.DataFrame(rows, columns=['variant', 'seed', 'R2']); o['region'] = region; o['n_dropped'] = nd; o['n_gap_months'] = int(gap.sum())
o.to_csv(out(f'gapfill_{region}.csv'), index=False); print('DONE')
