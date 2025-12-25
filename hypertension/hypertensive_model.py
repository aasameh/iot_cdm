import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models #type:ignore
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('results', exist_ok=True)

def load_data(npz_path='balanced_dataset.npz'):
    data = np.load(npz_path)
    X, y = data['spectrograms'], data['labels']
    label_map = {'negative': 0, 'positive': 1, 'hypertensive_event': 2}
    y = np.array([label_map[label] for label in y])
    print(f"Loaded {len(X)} samples | Shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    return X, y

def build_cnn_model(input_shape, n_classes=3):
    inp = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) # type: ignore
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) # type: ignore
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_lstm_model(input_shape, n_classes=3):
    inp = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Reshape((x.shape[1], -1))(x)
    
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) # type: ignore
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=64):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True) # type: ignore
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6) # type: ignore
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[early_stop, reduce_lr], verbose=1)
    return history

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    y_pred_proba = model.predict(X_test, verbose=0)
    
    report = classification_report(y_test, y_pred, target_names=['negative', 'positive', 'hypertensive'])
    auroc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    cm = confusion_matrix(y_test, y_pred)
    
    # Binary ROC data
    y_binary_hyper = (y_test == 2).astype(int)
    fpr_h, tpr_h, _ = roc_curve(y_binary_hyper, y_pred_proba[:, 2])
    auc_h = roc_auc_score(y_binary_hyper, y_pred_proba[:, 2])
    
    y_binary_pos = (y_test == 1).astype(int)
    fpr_p, tpr_p, _ = roc_curve(y_binary_pos, y_pred_proba[:, 1])
    auc_p = roc_auc_score(y_binary_pos, y_pred_proba[:, 1])
    
    return {
        'report': report, 'auroc': auroc, 'cm': cm,
        'fpr_h': fpr_h, 'tpr_h': tpr_h, 'auc_h': auc_h,
        'fpr_p': fpr_p, 'tpr_p': tpr_p, 'auc_p': auc_p
    }

def compare_models(X, y, epochs=100, batch_size=64):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=42)
    
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    results = {}
    
    # Train CNN
    print("\n" + "="*70)
    print("TRAINING CNN MODEL")
    print("="*70)
    cnn_model = build_cnn_model(X_train.shape[1:])
    print(f"CNN params: {cnn_model.count_params():,}")
    cnn_history = train_model(cnn_model, X_train, y_train, X_val, y_val, epochs, batch_size)
    results['cnn'] = evaluate_model(cnn_model, X_test, y_test, 'CNN')
    results['cnn']['history'] = cnn_history
    cnn_model.save('results/cnn_model.keras')
    
    # Train CNN-LSTM
    print("\n" + "="*70)
    print("TRAINING CNN-LSTM MODEL")
    print("="*70)
    lstm_model = build_cnn_lstm_model(X_train.shape[1:])
    print(f"CNN-LSTM params: {lstm_model.count_params():,}")
    lstm_history = train_model(lstm_model, X_train, y_train, X_val, y_val, epochs, batch_size)
    results['lstm'] = evaluate_model(lstm_model, X_test, y_test, 'CNN-LSTM')
    results['lstm']['history'] = lstm_history
    lstm_model.save('results/cnn_lstm_model.keras')
    
    # Save comparison results
    with open('results/comparison_results.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("="*70 + "\n\n")
        
        for name, res in [('CNN', results['cnn']), ('CNN-LSTM', results['lstm'])]:
            f.write(f"\n{name} MODEL:\n")
            f.write("-"*70 + "\n")
            f.write(res['report'])
            f.write(f"\nAUROC (macro): {res['auroc']:.4f}\n")
            f.write(f"Hypertensive vs Negative AUC: {res['auc_h']:.4f}\n")
            f.write(f"Positive vs Negative AUC: {res['auc_p']:.4f}\n\n")
    
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS")
    print("="*70)
    
    # Confusion matrices comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6)) # type: ignore
    for ax, name, res in [(ax1, 'CNN', results['cnn']), (ax2, 'CNN-LSTM', results['lstm'])]:
        sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['neg', 'pos', 'hyper'], yticklabels=['neg', 'pos', 'hyper'])
        ax.set_title(f'{name} - Confusion Matrix')
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Training history comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12)) # type: ignore
    
    # CNN history
    ax1.plot(results['cnn']['history'].history['loss'], label='Train')
    ax1.plot(results['cnn']['history'].history['val_loss'], label='Val')
    ax1.set_title('CNN - Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(results['cnn']['history'].history['accuracy'], label='Train')
    ax2.plot(results['cnn']['history'].history['val_accuracy'], label='Val')
    ax2.set_title('CNN - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # CNN-LSTM history
    ax3.plot(results['lstm']['history'].history['loss'], label='Train')
    ax3.plot(results['lstm']['history'].history['val_loss'], label='Val')
    ax3.set_title('CNN-LSTM - Loss')
    ax3.set_xlabel('Epoch')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.plot(results['lstm']['history'].history['accuracy'], label='Train')
    ax4.plot(results['lstm']['history'].history['val_accuracy'], label='Val')
    ax4.set_title('CNN-LSTM - Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_history_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ROC comparison - Hypertensive vs Negative
    plt.figure(figsize=(10, 8))
    plt.plot(results['cnn']['fpr_h'], results['cnn']['tpr_h'], 
             label=f"CNN (AUC={results['cnn']['auc_h']:.4f})", linewidth=2)
    plt.plot(results['lstm']['fpr_h'], results['lstm']['tpr_h'], 
             label=f"CNN-LSTM (AUC={results['lstm']['auc_h']:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Comparison: Hypertensive vs Negative', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig('results/roc_comparison_hypertensive.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ROC comparison - Positive vs Negative
    plt.figure(figsize=(10, 8))
    plt.plot(results['cnn']['fpr_p'], results['cnn']['tpr_p'], 
             label=f"CNN (AUC={results['cnn']['auc_p']:.4f})", linewidth=2)
    plt.plot(results['lstm']['fpr_p'], results['lstm']['tpr_p'], 
             label=f"CNN-LSTM (AUC={results['lstm']['auc_p']:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Comparison: Positive vs Negative', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig('results/roc_comparison_positive.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nSaved all comparison results to 'results/' folder")
    print(f"  - comparison_results.txt")
    print(f"  - confusion_matrix_comparison.png")
    print(f"  - training_history_comparison.png")
    print(f"  - roc_comparison_hypertensive.png")
    print(f"  - roc_comparison_positive.png")
    print(f"  - cnn_model.keras")
    print(f"  - cnn_lstm_model.keras")
    
    return results

if __name__ == "__main__":
    X, y = load_data('balanced_dataset.npz')
    results = compare_models(X, y, epochs=500, batch_size=64)
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)