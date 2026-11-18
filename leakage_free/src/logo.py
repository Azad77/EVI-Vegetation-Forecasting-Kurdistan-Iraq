"""Leave-one-governorate-out spatial transfer (Referee 2, Point 3 / Table S4).
Train on the training windows (targets <= Jul 2023) of the other 3 governorates; test zero-shot on the held-out
governorate's 17 test windows. Fixed 100 epochs, 5 seeds. No EVI of the target governorate is used for training.
Usage: python logo.py TARGET_REGION -> logo_TARGET.csv"""
import sys
import models
from common import *
single_thread()
target = sys.argv[1]; data = load(); P = {r: prep(data[r], strict=True) for r in REGIONS}
src = [r for r in REGIONS if r != target]
Xs = np.concatenate([P[r][0][:-NTEST] for r in src]); ys = np.concatenate([P[r][1][:-NTEST] for r in src])
X, y, st, dates = P[target]; mu, sd = st['evi']; Xte, yte = X[-NTEST:], y[-NTEST:]; t = yte * sd + mu
rows = []
for name in ['CNN_BiLSTM_GRU', 'BiLSTM_GRU', 'CNN_GRU']:
    b, lr = models.REG[name]
    for seed in range(NSEED):
        tf.keras.utils.set_random_seed(seed); m = b(); m.compile(keras.optimizers.Adam(lr), 'mse')
        m.fit(Xs, ys, epochs=100, batch_size=16, verbose=0, shuffle=True)
        p = m.predict(Xte, verbose=0).ravel() * sd + mu
        mt = metrics(t, p); rows.append(dict(model=name, target=target, seed=seed, R2=mt['R2'], MAE=mt['MAE'], Acc=mt['Acc']))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(out(f'logo_{target}.csv'), index=False); print('DONE')
