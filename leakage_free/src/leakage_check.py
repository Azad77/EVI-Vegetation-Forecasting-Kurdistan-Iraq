"""Reproduces the leakage diagnosis for BiLSTM (5 seeds x 4 governorates).

'original' = protocol of the first submission: scalers/gap-fill climatology fitted on all 108 months and the
             TEST windows passed as validation_data to EarlyStopping(restore_best_weights) and ReduceLROnPlateau.
'clean'    = same training settings, but scalers/climatology from training months only and early stopping on
             the last 12 training windows.
Output: leakage_check_bilstm.csv
"""
import models
from common import *
single_thread()
data = load(); rows = []
for r in REGIONS:
    for proto in ['original', 'clean']:
        X, y, st, dates = prep(data[r], strict=(proto == 'clean')); mu, sd = st['evi']
        Xtr, ytr, Xte, yte = split(X, y)
        for seed in range(NSEED):
            tf.keras.utils.set_random_seed(seed)
            m = models.build_bilstm(); m.compile(keras.optimizers.Adam(5e-4), 'mse')
            cb = [keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
                  keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)]
            if proto == 'original':   # LEAKY: test set used for early stopping (reproduces the old numbers)
                m.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=150, batch_size=16, verbose=0, callbacks=cb)
            else:
                m.fit(Xtr[:-NV], ytr[:-NV], validation_data=(Xtr[-NV:], ytr[-NV:]), epochs=150, batch_size=16, verbose=0, callbacks=cb)
            p = m.predict(Xte, verbose=0).ravel()
            rows.append((r, proto, seed, r2_score(yte * sd + mu, p * sd + mu))); print(rows[-1], flush=True)
df = pd.DataFrame(rows, columns=['region', 'proto', 'seed', 'r2']); df.to_csv(out('leakage_check_bilstm.csv'), index=False)
print(df.groupby(['region', 'proto']).r2.agg(['mean', 'std', 'min', 'max']).round(3))
