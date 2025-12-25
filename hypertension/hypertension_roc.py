import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import glob
import os

# -------------------------------
# 1. Load the saved model
# -------------------------------
model = tf.keras.models.load_model('hypertension_cnn_lstm.keras') #type:ignore
print("Model loaded successfully.")

# -------------------------------
# 2. Load test data only
# -------------------------------
def load_test_data(npz_dir='ppg_spectrogram_output'):
    X, y = [], []
    for f in glob.glob(os.path.join(npz_dir, '*_spectrograms.npz')):
        data = np.load(f)
        X.append(data['spectrograms'])
        y.append(data['labels'])
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    
    label_map = {'negative': 0, 'positive': 1, 'hypertensive_event': 2}
    y = np.array([label_map[label] for label in y])
    
    print(f"Loaded {len(X)} samples | Shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    return X, y

X, y = load_test_data()

# -------------------------------
# 3. Make predictions
# -------------------------------
y_pred_proba = model.predict(X, verbose=0)

# -------------------------------
# 4. ROC: Positive vs Negative
# -------------------------------
mask_pos_vs_neg = np.isin(y, [0, 1])
y_bin_pos = (y[mask_pos_vs_neg] == 1).astype(int)
y_score_pos = y_pred_proba[mask_pos_vs_neg, 1]

fpr_pos, tpr_pos, _ = roc_curve(y_bin_pos, y_score_pos)
auc_pos = auc(fpr_pos, tpr_pos)

plt.figure(figsize=(7, 6))
plt.plot(fpr_pos, tpr_pos, color='orange', lw=2,
         label=f'Positive vs Negative (AUC = {auc_pos:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC: Pre-hypertensive vs Negative')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_positive_vs_negative.png', dpi=150)
plt.show()
print(f"AUC (Positive vs Negative): {auc_pos:.4f}")

# -------------------------------
# 5. ROC: (Positive + Hypertensive) vs Negative
# -------------------------------
y_bin_risk = (y >= 1).astype(int)
y_score_risk = y_pred_proba[:, 1] + y_pred_proba[:, 2]

fpr_risk, tpr_risk, _ = roc_curve(y_bin_risk, y_score_risk)
auc_risk = auc(fpr_risk, tpr_risk)

plt.figure(figsize=(7, 6))
plt.plot(fpr_risk, tpr_risk, color='red', lw=2,
         label=f'Risk vs Negative (AUC = {auc_risk:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC: Hypertensive Risk vs Negative')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_risk_vs_negative.png', dpi=150)
plt.show()
print(f"AUC (Risk vs Negative): {auc_risk:.4f}")
