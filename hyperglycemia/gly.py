"""
Hyperglycemia Classification: Model Comparison
Compares CNN-only, LSTM-only, and CNN-LSTM hybrid models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve,
                             accuracy_score, precision_score, recall_score, f1_score)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# ==================== CONFIGURATION ====================
WINDOW_SIZE = 6
PREDICTION_HORIZON = 3
HYPER_THRESHOLD = 180
TRAIN_SPLIT = 0.8

# ==================== DATA PREPROCESSING ====================

def load_and_preprocess_data(filepath, is_augmented=False):
    """Load CSV data (original or augmented)"""
    print("Loading data...")
    df = pd.read_csv(filepath)
    df = df[df['BGLevel'] > 0].copy()
    df['DateTime'] = pd.to_datetime(df['BGDate'] + ' ' + df['BGTime'])
    df = df.sort_values(['PtID', 'DateTime'])
    
    if is_augmented and 'SequenceID' in df.columns:
        print(f"Loaded augmented data: {len(df)} records, {df['SequenceID'].nunique()} sequences")
    else:
        print(f"Loaded original data: {len(df)} records from {df['PtID'].nunique()} patients")
    
    return df

def apply_savitzky_golay_filter(bg_values, window_length=15, polyorder=1):
    """Apply Savitzky-Golay filter"""
    if len(bg_values) < window_length:
        return bg_values
    return savgol_filter(bg_values, window_length, polyorder)

def extract_time_domain_features(window):
    """Extract statistical features"""
    if len(window) == 0 or np.all(np.isnan(window)):
        return [0.0] * 10
    
    clean_window = window[~np.isnan(window)]
    if len(clean_window) == 0:
        return [0.0] * 10
    
    features = {
        'min': np.min(clean_window),
        'max': np.max(clean_window),
        'mean': np.mean(clean_window),
        'std': np.std(clean_window) if len(clean_window) > 1 else 0.0,
        'median': np.median(clean_window),
        'range': np.ptp(clean_window),
        'q25': np.percentile(clean_window, 25),
        'q75': np.percentile(clean_window, 75)
    }
    
    from scipy.stats import skew, kurtosis
    try:
        features['skewness'] = skew(clean_window) if len(clean_window) > 2 else 0.0
        features['kurtosis'] = kurtosis(clean_window) if len(clean_window) > 3 else 0.0
    except:
        features['skewness'] = 0.0
        features['kurtosis'] = 0.0
    
    feature_list = list(features.values())
    feature_list = [0.0 if (np.isnan(f) or np.isinf(f)) else f for f in feature_list]
    
    return feature_list

def create_sequences_from_continuous(patient_data, window_size, prediction_horizon):
    """Create sequences from continuous patient data"""
    bg_values = patient_data['BGLevel'].values
    bg_smoothed = apply_savitzky_golay_filter(bg_values)
    
    X_windows = []
    X_features = []
    y_labels = []
    
    for i in range(len(bg_smoothed) - window_size - prediction_horizon):
        window = bg_smoothed[i:i + window_size]
        future_value = bg_smoothed[i + window_size + prediction_horizon - 1]
        label = 1 if future_value > HYPER_THRESHOLD else 0
        
        X_windows.append(window)
        X_features.append(extract_time_domain_features(window))
        y_labels.append(label)
    
    return np.array(X_windows), np.array(X_features), np.array(y_labels)

def extract_sequences_from_augmented(df, window_size):
    """Extract sequences from augmented data"""
    X_windows = []
    X_features = []
    y_labels = []
    
    for seq_id in df['SequenceID'].unique():
        seq_data = df[df['SequenceID'] == seq_id].sort_values('DateTime')
        
        if len(seq_data) < window_size:
            continue
        
        window = seq_data['BGLevel'].values[:window_size]
        label = seq_data['Label'].iloc[0] if 'Label' in seq_data.columns else 0
        
        X_windows.append(window)
        X_features.append(extract_time_domain_features(window))
        y_labels.append(label)
    
    return np.array(X_windows), np.array(X_features), np.array(y_labels)

def prepare_dataset(df, is_augmented=False):
    """Prepare dataset"""
    print("\nPreparing dataset...")
    
    if is_augmented and 'SequenceID' in df.columns:
        X_windows, X_features, y = extract_sequences_from_augmented(df, WINDOW_SIZE)
    else:
        all_X_windows = []
        all_X_features = []
        all_y = []
        
        patients = df['PtID'].unique()
        
        for patient_id in patients:
            patient_data = df[df['PtID'] == patient_id].copy()
            
            if len(patient_data) < WINDOW_SIZE + PREDICTION_HORIZON + 10:
                continue
            
            X_win, X_feat, y = create_sequences_from_continuous(
                patient_data, WINDOW_SIZE, PREDICTION_HORIZON
            )
            
            if len(X_win) > 0:
                all_X_windows.append(X_win)
                all_X_features.append(X_feat)
                all_y.append(y)
        
        X_windows = np.concatenate(all_X_windows, axis=0)
        X_features = np.concatenate(all_X_features, axis=0)
        y = np.concatenate(all_y, axis=0)
    
    X_windows = np.nan_to_num(X_windows, nan=0.0, posinf=0.0, neginf=0.0)
    X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Final dataset: {len(y)} samples")
    print(f"Hyperglycemic: {np.sum(y)} ({np.sum(y)/len(y)*100:.2f}%)")
    print(f"Normal: {len(y) - np.sum(y)} ({(len(y)-np.sum(y))/len(y)*100:.2f}%)")
    
    return X_windows, X_features, y

# ==================== MODEL ARCHITECTURES ====================

def build_cnn_only(window_size, n_features):
    """CNN-only model"""
    # Input 1: Time series
    input_window = layers.Input(shape=(window_size, 1), name='bg_window')
    
    # CNN layers
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_window)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    
    # Input 2: Features
    input_features = layers.Input(shape=(n_features,), name='time_features')
    f = layers.Dense(32, activation='relu')(input_features)
    f = layers.Dropout(0.2)(f)
    
    # Combine
    combined = layers.Concatenate()([x, f])
    
    # Classification
    z = layers.Dense(64, activation='relu')(combined)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(32, activation='relu')(z)
    z = layers.Dropout(0.2)(z)
    output = layers.Dense(1, activation='sigmoid', name='hyperglycemia')(z)
    
    model = models.Model(inputs=[input_window, input_features], outputs=output)
    return model

def build_lstm_only(window_size, n_features):
    """LSTM-only model"""
    # Input 1: Time series
    input_window = layers.Input(shape=(window_size, 1), name='bg_window')
    
    # LSTM layers
    x = layers.LSTM(128, return_sequences=True)(input_window)
    x = layers.Dropout(0.3)(x)
    
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    
    # Input 2: Features
    input_features = layers.Input(shape=(n_features,), name='time_features')
    f = layers.Dense(32, activation='relu')(input_features)
    f = layers.Dropout(0.2)(f)
    
    # Combine
    combined = layers.Concatenate()([x, f])
    
    # Classification
    z = layers.Dense(64, activation='relu')(combined)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(32, activation='relu')(z)
    z = layers.Dropout(0.2)(z)
    output = layers.Dense(1, activation='sigmoid', name='hyperglycemia')(z)
    
    model = models.Model(inputs=[input_window, input_features], outputs=output)
    return model

def build_cnn_lstm(window_size, n_features):
    """CNN-LSTM hybrid model"""
``    # Input 1: Time series
    input_window = layers.Input(shape=(window_size, 1), name='bg_window')
    
    # CNN layers
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_window)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # LSTM layers
    x = layers.LSTM(100, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.LSTM(50, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    
    # Input 2: Features
    input_features = layers.Input(shape=(n_features,), name='time_features')
    f = layers.Dense(32, activation='relu')(input_features)
    f = layers.Dropout(0.2)(f)
    f = layers.Dense(16, activation='relu')(f)
    
    # Combine
    combined = layers.Concatenate()([x, f])
    
    # Classification
    z = layers.Dense(64, activation='relu')(combined)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(32, activation='relu')(z)
    z = layers.Dropout(0.2)(z)
    output = layers.Dense(1, activation='sigmoid', name='hyperglycemia')(z)
    
    model = models.Model(inputs=[input_window, input_features], outputs=output)
    return model

# ==================== TRAINING ====================

def prepare_train_test_data(X_windows, X_features, y):
    """Prepare and split data"""
    # Normalize
    scaler_window = MinMaxScaler()
    scaler_features = MinMaxScaler()
    
    X_windows_scaled = scaler_window.fit_transform(X_windows.reshape(-1, 1)).reshape(X_windows.shape)
    X_features_scaled = scaler_features.fit_transform(X_features)
    
    # Reshape for models
    X_windows_scaled = X_windows_scaled.reshape(X_windows_scaled.shape[0], X_windows_scaled.shape[1], 1)
    
    # Split
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=1-TRAIN_SPLIT, 
                                           stratify=y, random_state=42)
    
    X_win_train = X_windows_scaled[train_idx]
    X_feat_train = X_features_scaled[train_idx]
    y_train = y[train_idx]
    
    X_win_test = X_windows_scaled[test_idx]
    X_feat_test = X_features_scaled[test_idx]
    y_test = y[test_idx]
    
    return (X_win_train, X_feat_train, y_train), (X_win_test, X_feat_test, y_test)

def train_model(model, model_name, train_data, test_data, epochs=100, batch_size=32):
    """Train a model"""
    X_win_train, X_feat_train, y_train = train_data
    X_win_test, X_feat_test, y_test = test_data
    
    print(f"\n{'='*60}")
    print(f"TRAINING {model_name.upper()}")
    print(f"{'='*60}")
    
    # Calculate class weights
    n_positive = np.sum(y_train)
    n_negative = len(y_train) - n_positive
    
    if n_positive > 0 and n_negative > 0:
        if abs(n_positive - n_negative) / len(y_train) < 0.1:
            class_weight = {0: 1.0, 1: 1.0}
        else:
            class_weight = {0: 1.0, 1: n_negative / (n_positive + 1e-7)}
    else:
        class_weight = {0: 1.0, 1: 1.0}
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(),
                 keras.metrics.AUC(name='auc')]
    )
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, 
                                         restore_best_weights=True, verbose=0)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                            patience=5, min_lr=1e-6, verbose=0)
    
    # Train
    history = model.fit(
        [X_win_train, X_feat_train], y_train,
        validation_data=([X_win_test, X_feat_test], y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    print(f"Training complete: {len(history.history['loss'])} epochs")
    
    return model, history

# ==================== EVALUATION ====================

def evaluate_model(model, model_name, test_data):
    """Evaluate model and return metrics"""
    X_win_test, X_feat_test, y_test = test_data
    
    # Predictions
    y_pred_proba = model.predict([X_win_test, X_feat_test], verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, zero_division=0) * 100
    recall = recall_score(y_test, y_pred, zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, zero_division=0) * 100
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0
    
    # ROC-AUC
    auroc = roc_auc_score(y_test, y_pred_proba) * 100 if len(np.unique(y_test)) > 1 else 0.0
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1,
        'auroc': auroc,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'fpr': fpr,
        'tpr': tpr
    }
    
    return metrics

def print_metrics_table(all_metrics):
    """Print comparison table"""
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Create DataFrame
    rows = []
    for metrics in all_metrics:
        rows.append({
            'Model': metrics['model_name'],
            'Accuracy': f"{metrics['accuracy']:.2f}%",
            'Precision': f"{metrics['precision']:.2f}%",
            'Recall': f"{metrics['recall']:.2f}%",
            'Sensitivity': f"{metrics['sensitivity']:.2f}%",
            'Specificity': f"{metrics['specificity']:.2f}%",
            'F1-Score': f"{metrics['f1_score']:.2f}%",
            'AUROC': f"{metrics['auroc']:.2f}%"
        })
    
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    
    # Confusion Matrices
    print("\n" + "="*80)
    print("CONFUSION MATRICES")
    print("="*80)
    
    for metrics in all_metrics:
        cm = metrics['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        print(f"\n{metrics['model_name']}:")
        print(f"  TN: {tn:5d}  |  FP: {fp:5d}")
        print(f"  FN: {fn:5d}  |  TP: {tp:5d}")

def plot_comparison(all_metrics):
    """Plot comprehensive comparison"""
    n_models = len(all_metrics)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Metrics Bar Chart
    ax1 = plt.subplot(2, 3, 1)
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUROC']
    x = np.arange(len(metrics_names))
    width = 0.25
    
    for i, metrics in enumerate(all_metrics):
        values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score'],
            metrics['auroc']
        ]
        ax1.bar(x + i*width, values, width, label=metrics['model_name'])
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score (%)')
    ax1.set_title('Performance Metrics Comparison')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    
    # 2. Sensitivity vs Specificity
    ax2 = plt.subplot(2, 3, 2)
    for metrics in all_metrics:
        ax2.scatter(metrics['specificity'], metrics['sensitivity'], 
                   s=200, label=metrics['model_name'], alpha=0.7)
    ax2.set_xlabel('Specificity (%)')
    ax2.set_ylabel('Sensitivity (%)')
    ax2.set_title('Sensitivity vs Specificity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 105])
    ax2.set_ylim([0, 105])
    
    # 3. ROC Curves
    ax3 = plt.subplot(2, 3, 3)
    for metrics in all_metrics:
        ax3.plot(metrics['fpr'], metrics['tpr'], 
                label=f"{metrics['model_name']} (AUC={metrics['auroc']:.2f}%)", linewidth=2)
    ax3.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curves Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4-6. Confusion Matrices
    for idx, metrics in enumerate(all_metrics):
        ax = plt.subplot(2, 3, 4 + idx)
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Hyper'],
                    yticklabels=['Normal', 'Hyper'], ax=ax,
                    cbar_kws={'label': 'Count'})
        ax.set_title(f'{metrics["model_name"]} - Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plots saved to 'model_comparison_results.png'")
    plt.show()

# ==================== MAIN EXECUTION ====================

def main(filepath, is_augmented=False):
    """Main comparison pipeline"""
    
    print("="*80)
    print("HYPERGLYCEMIA PREDICTION: MODEL COMPARISON")
    print("CNN-only vs LSTM-only vs CNN-LSTM")
    print("="*80)
    
    # Load data
    df = load_and_preprocess_data(filepath, is_augmented=is_augmented)
    X_windows, X_features, y = prepare_dataset(df, is_augmented=is_augmented)
    
    # Prepare train/test split
    train_data, test_data = prepare_train_test_data(X_windows, X_features, y)
    
    print(f"\nTrain set: {len(train_data[2])} samples")
    print(f"Test set: {len(test_data[2])} samples")
    
    # Build models
    models_dict = {
        'CNN-only': build_cnn_only(WINDOW_SIZE, X_features.shape[1]),
        'LSTM-only': build_lstm_only(WINDOW_SIZE, X_features.shape[1]),
        'CNN-LSTM': build_cnn_lstm(WINDOW_SIZE, X_features.shape[1])
    }
    
    # Train and evaluate all models
    all_metrics = []
    
    for model_name, model in models_dict.items():
        # Train
        trained_model, history = train_model(
            model, model_name, train_data, test_data, 
            epochs=100, batch_size=32
        )
        
        # Evaluate
        metrics = evaluate_model(trained_model, model_name, test_data)
        all_metrics.append(metrics)
        
        # Save model
        trained_model.save(f'hyperglycemia_{model_name.lower().replace("-", "_")}_model.h5')
        print(f"✓ Model saved to 'hyperglycemia_{model_name.lower().replace('-', '_')}_model.h5'")
    
    # Print comparison table
    print_metrics_table(all_metrics)
    
    # Plot comparison
    plot_comparison(all_metrics)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([{
        'Model': m['model_name'],
        'Accuracy (%)': f"{m['accuracy']:.2f}",
        'Precision (%)': f"{m['precision']:.2f}",
        'Recall (%)': f"{m['recall']:.2f}",
        'Sensitivity (%)': f"{m['sensitivity']:.2f}",
        'Specificity (%)': f"{m['specificity']:.2f}",
        'F1-Score (%)': f"{m['f1_score']:.2f}",
        'AUROC (%)': f"{m['auroc']:.2f}"
    } for m in all_metrics])
    
    metrics_df.to_csv('model_comparison_metrics.csv', index=False)
    print("\n✓ Metrics saved to 'model_comparison_metrics.csv'")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    
    return all_metrics

# ==================== USAGE ====================

if __name__ == "__main__":
    # Use augmented data (recommended)
    all_metrics = main("augmented_glucose_data.csv", is_augmented=True)
    
    # Or use original data
    # all_metrics = main("your_glucose_data.csv", is_augmented=False)