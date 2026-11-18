"""Model zoo: 9 primary architectures + 2 neural baselines (identical to the manuscript notebook).

Set models.MC_DROPOUT = True *before* calling a builder to get dropout layers that stay active at
inference (Monte Carlo Dropout). BatchNormalization always stays in inference mode at prediction time.
"""
from common import layers, Model, SEQ

NF = 5            # number of input features (changed by ablation.py)
MC_DROPOUT = False


class MCDropout(layers.Dropout):
    def call(self, x, training=None):
        return super().call(x, training=True)


def Drop(rate):
    return MCDropout(rate) if MC_DROPOUT else layers.Dropout(rate)


def _inp():
    return layers.Input((SEQ, NF))


def build_bilstm(units=64, dropout=0.3):
    inp = _inp(); x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(inp); x = Drop(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(units // 2, return_sequences=True))(x); x = Drop(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(units // 4))(x); return Model(inp, layers.Dense(1)(x))

def build_gru(units=64, dropout=0.3):
    inp = _inp(); x = layers.GRU(units, return_sequences=True)(inp); x = Drop(dropout)(x)
    x = layers.GRU(units)(x); x = Drop(dropout)(x); return Model(inp, layers.Dense(1)(x))

def build_cnn_gru(filters=64, units=64, dropout=0.3):
    inp = _inp(); x = layers.Conv1D(filters, 3, activation='relu', padding='same')(inp); x = layers.BatchNormalization()(x)
    x = layers.GRU(units)(x); x = Drop(dropout)(x); return Model(inp, layers.Dense(1)(x))

def build_cnn_bilstm(filters=64, units=64, dropout=0.3):
    inp = _inp(); x = layers.Conv1D(filters, 3, activation='relu', padding='same')(inp); x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(units))(x); x = Drop(dropout)(x); return Model(inp, layers.Dense(1)(x))

def build_bilstm_gru(units=64, dropout=0.3):
    inp = _inp(); x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(inp); x = Drop(dropout)(x)
    x = layers.GRU(units)(x); x = Drop(dropout)(x); return Model(inp, layers.Dense(1)(x))

def build_cnn_bilstm_gru(filters=64, units=64, dropout=0.3):
    inp = _inp(); x = layers.Conv1D(filters, 3, activation='relu', padding='same')(inp); x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x); x = Drop(dropout)(x)
    x = layers.GRU(units // 2)(x); x = Drop(dropout)(x); return Model(inp, layers.Dense(1)(x))

def build_tcn(filters=64, dropout=0.2):
    inp = _inp(); x = inp
    for d in (1, 2, 4, 8):
        x = layers.Conv1D(filters, 3, padding='causal', dilation_rate=d, activation='relu')(x)
    x = Drop(dropout)(x); x = layers.GlobalAveragePooling1D()(x); return Model(inp, layers.Dense(1)(x))

def build_transformer(d_model=64, n_heads=4, dropout=0.1):
    inp = _inp(); x = layers.Dense(d_model)(inp)
    a = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)(x, x); x = layers.LayerNormalization(epsilon=1e-6)(x + a)
    ff = layers.Dense(d_model * 2, activation='relu')(x); ff = layers.Dense(d_model)(ff); x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    x = Drop(dropout)(x); x = layers.GlobalAveragePooling1D()(x); return Model(inp, layers.Dense(1)(x))

def build_cnn_transformer(filters=64, d_model=64, n_heads=4, dropout=0.2):
    inp = _inp(); x = layers.Conv1D(filters, 3, padding='same', activation='relu')(inp); x = layers.BatchNormalization()(x)
    x = layers.Dense(d_model)(x); a = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + a); x = Drop(dropout)(x); x = layers.GlobalAveragePooling1D()(x)
    return Model(inp, layers.Dense(1)(x))

# neural baselines
def build_lstm(units=64, dropout=0.3):
    inp = _inp(); x = layers.LSTM(units)(inp); x = Drop(dropout)(x); return Model(inp, layers.Dense(1)(x))

def build_cnn(filters=64, dropout=0.3):
    inp = _inp(); x = layers.Conv1D(filters, 3, activation='relu', padding='same')(inp)
    x = layers.Conv1D(filters // 2, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x); x = Drop(dropout)(x); return Model(inp, layers.Dense(1)(x))


# name -> (builder, learning rate)
REG = {'BiLSTM': (build_bilstm, 5e-4), 'GRU': (build_gru, 5e-4), 'CNN_GRU': (build_cnn_gru, 1e-3),
       'CNN_BiLSTM': (build_cnn_bilstm, 5e-4), 'BiLSTM_GRU': (build_bilstm_gru, 5e-4),
       'CNN_BiLSTM_GRU': (build_cnn_bilstm_gru, 5e-4), 'TCN': (build_tcn, 1e-3),
       'Transformer': (build_transformer, 5e-4), 'CNN_Transformer': (build_cnn_transformer, 5e-4)}
REG_BASE = {'LSTM_baseline': (build_lstm, 5e-4), 'CNN_baseline': (build_cnn, 5e-4)}
