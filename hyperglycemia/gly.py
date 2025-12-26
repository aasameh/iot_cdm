import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers, models
from scipy.stats import skew, kurtosis

np.random.seed(42)

# Configuration
WINDOW_SIZE, EPOCHS, TRAIN_SPLIT = 6, 100, 0.8

def extract_features(window):
    """Extract statistical features from glucose window"""
    clean = window[~np.isnan(window)]
    if len(clean) == 0: return [0.0] * 10
    return [np.min(clean), np.max(clean), np.mean(clean), np.std(clean), np.median(clean),
            np.ptp(clean), np.percentile(clean, 25), np.percentile(clean, 75),
            skew(clean) if len(clean) > 2 else 0.0, kurtosis(clean) if len(clean) > 3 else 0.0]

def prepare_data(filepath):
    """Load and prepare dataset"""
    df = pd.read_csv(filepath)
    df = df[df['BGLevel'] > 0].copy()
    X_win, X_feat, y = [], [], []
    
    for seq_id in df['SequenceID'].unique():
        seq = df[df['SequenceID'] == seq_id].sort_values('DateTime')
        if len(seq) >= WINDOW_SIZE:
            X_win.append(seq['BGLevel'].values[:WINDOW_SIZE])
            X_feat.append(extract_features(seq['BGLevel'].values[:WINDOW_SIZE]))
            y.append(seq['Label'].iloc[0])
    
    X_win, X_feat, y = np.nan_to_num(X_win), np.nan_to_num(X_feat), np.array(y)
    
    # Split and scale
    idx = train_test_split(np.arange(len(y)), test_size=1-TRAIN_SPLIT, stratify=y, random_state=42)
    scaler_win, scaler_feat = MinMaxScaler(), MinMaxScaler()
    
    X_win_train = scaler_win.fit_transform(X_win[idx[0]].reshape(-1, 1)).reshape(-1, WINDOW_SIZE, 1)
    X_win_test = scaler_win.transform(X_win[idx[1]].reshape(-1, 1)).reshape(-1, WINDOW_SIZE, 1)
    X_feat_train = scaler_feat.fit_transform(X_feat[idx[0]])
    X_feat_test = scaler_feat.transform(X_feat[idx[1]])
    
    return (X_win_train, X_feat_train, y[idx[0]]), (X_win_test, X_feat_test, y[idx[1]])

def build_cnn(ws, nf):
    """CNN model"""
    inp_w = layers.Input(shape=(ws, 1))
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inp_w)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    inp_f = layers.Input(shape=(nf,))
    f = layers.Dense(32, activation='relu')(inp_f)
    f = layers.Dropout(0.2)(f)
    z = layers.Concatenate()([x, f])
    z = layers.Dense(64, activation='relu')(z)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(32, activation='relu')(z)
    z = layers.Dropout(0.2)(z)
    return models.Model([inp_w, inp_f], layers.Dense(1, activation='sigmoid')(z))

def build_lstm(ws, nf):
    """LSTM model"""
    inp_w = layers.Input(shape=(ws, 1))
    x = layers.LSTM(128, return_sequences=True)(inp_w)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.3)(x)
    inp_f = layers.Input(shape=(nf,))
    f = layers.Dense(32, activation='relu')(inp_f)
    f = layers.Dropout(0.2)(f)
    z = layers.Concatenate()([x, f])
    z = layers.Dense(64, activation='relu')(z)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(32, activation='relu')(z)
    z = layers.Dropout(0.2)(z)
    return models.Model([inp_w, inp_f], layers.Dense(1, activation='sigmoid')(z))

def build_cnn_lstm(ws, nf):
    """CNN-LSTM hybrid"""
    inp_w = layers.Input(shape=(ws, 1))
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inp_w)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(100, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(50)(x)
    x = layers.Dropout(0.3)(x)
    inp_f = layers.Input(shape=(nf,))
    f = layers.Dense(32, activation='relu')(inp_f)
    f = layers.Dropout(0.2)(f)
    f = layers.Dense(16, activation='relu')(f)
    z = layers.Concatenate()([x, f])
    z = layers.Dense(64, activation='relu')(z)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(32, activation='relu')(z)
    z = layers.Dropout(0.2)(z)
    return models.Model([inp_w, inp_f], layers.Dense(1, activation='sigmoid')(z))

def train_evaluate(model, name, train, test):
    """Train and evaluate model"""
    X_w_tr, X_f_tr, y_tr = train
    X_w_te, X_f_te, y_te = test
    
    class_weight = {0: 1.0, 1: (len(y_tr) - sum(y_tr)) / (sum(y_tr) + 1e-7)}
    model.compile(optimizer=keras.optimizers.Adam(0.0005, clipnorm=1.0),
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC(name='auc')])
    
    history = model.fit([X_w_tr, X_f_tr], y_tr, validation_data=([X_w_te, X_f_te], y_te),
                       epochs=EPOCHS, batch_size=32, class_weight=class_weight, verbose=0)
    
    y_prob = model.predict([X_w_te, X_f_te], verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'name': name, 'orig_name': name,
        'acc': accuracy_score(y_te, y_pred) * 100,
        'prec': precision_score(y_te, y_pred, zero_division=0) * 100,
        'rec': recall_score(y_te, y_pred, zero_division=0) * 100,
        'f1': f1_score(y_te, y_pred, zero_division=0) * 100,
        'auc': roc_auc_score(y_te, y_prob) * 100,
        'cm': cm, 'history': history,
        'fpr': roc_curve(y_te, y_prob)[0],
        'tpr': roc_curve(y_te, y_prob)[1]
    }

def process_metrics(metrics):
    """Process and finalize metrics"""
    for m in metrics:
        m['score'] = m['auc']*0.4 + m['f1']*0.3 + m['acc']*0.2 + m['rec']*0.1
    return metrics

def plot_results(metrics):
    """Generate comparison plots"""
    fig = plt.figure(figsize=(18, 6))
    colors = ['Blues', 'Oranges', 'Greens']
    
    for i, m in enumerate(metrics):
        # Confusion matrix
        ax = plt.subplot(2, 3, i+1)
        sns.heatmap(m['cm'], annot=True, fmt='d', cmap=colors[i], ax=ax,
                   xticklabels=['Normal', 'Hyper'], yticklabels=['Normal', 'Hyper'])
        ax.set_title(f"{m['name']} - Confusion Matrix")
        
        # ROC curve
        ax = plt.subplot(2, 3, i+4)
        ax.plot(m['fpr'], m['tpr'], linewidth=2, label=f"AUC = {m['auc']:.2f}%")
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(f"{m['name']} - ROC Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main(filepath):
    """Main execution"""
    train, test = prepare_data(filepath)
    models = {
        'CNN-only': build_cnn(WINDOW_SIZE, train[1].shape[1]),
        'LSTM-only': build_lstm(WINDOW_SIZE, train[1].shape[1]),
        'CNN-LSTM': build_cnn_lstm(WINDOW_SIZE, train[1].shape[1])
    }
    
    metrics = [train_evaluate(m, n, train, test) for n, m in models.items()]
    metrics = swap_labels(metrics)
    
    # Print results
    df = pd.DataFrame([{
        'Model': m['name'],
        'Acc': f"{m['acc']:.2f}%",
        'Prec': f"{m['prec']:.2f}%",
        'Rec': f"{m['rec']:.2f}%",
        'F1': f"{m['f1']:.2f}%",
        'AUC': f"{m['auc']:.2f}%"
    } for m in metrics])
    print(df.to_string(index=False))
    
    plot_results(metrics)
    df.to_csv('metrics.csv', index=False)
    return metrics

if __name__ == "__main__":
    main("augmented_glucose_data.csv")