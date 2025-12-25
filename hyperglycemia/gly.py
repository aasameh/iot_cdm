"""
Hyperglycemia Classification using CNN-LSTM
Predicts if blood glucose will exceed 180 mg/dL in the next 15 minutes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from scipy import interpolate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==================== CONFIGURATION ====================
WINDOW_SIZE = 6  # 30 minutes of history (5-min intervals)
PREDICTION_HORIZON = 3  # 15 minutes ahead (3 * 5-min)
HYPER_THRESHOLD = 180  # mg/dL
HYPO_THRESHOLD = 70    # mg/dL
TRAIN_SPLIT = 0.8

# ==================== DATA PREPROCESSING ====================

def load_and_preprocess_data(filepath):
    """Load CSV data and preprocess for each patient"""
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Remove rows with zero or missing BG levels
    df = df[df['BGLevel'] > 0].copy()
    
    # Convert date/time
    df['DateTime'] = pd.to_datetime(df['BGDate'] + ' ' + df['BGTime']) # type:ignore
    df = df.sort_values(['PtID', 'DateTime'])
    
    print(f"Loaded {len(df)} records from {df['PtID'].nunique()} patients")
    return df

def apply_savitzky_golay_filter(bg_values, window_length=15, polyorder=1):
    """Apply Savitzky-Golay filter to smooth noise"""
    if len(bg_values) < window_length:
        return bg_values
    return savgol_filter(bg_values, window_length, polyorder)

def interpolate_missing_values(bg_series):
    """Interpolate missing values using spline interpolation"""
    x = np.arange(len(bg_series))
    valid_idx = ~np.isnan(bg_series)
    
    if valid_idx.sum() < 2:
        return bg_series
    
    f = interpolate.interp1d(x[valid_idx], bg_series[valid_idx], 
                             kind='linear', fill_value='extrapolate')
    return f(x)

def extract_time_domain_features(window):
    """Extract statistical time-domain features from a window"""
    # Handle edge cases
    if len(window) == 0 or np.all(np.isnan(window)):
        return [0.0] * 10
    
    # Remove any NaN values for calculations
    clean_window = window[~np.isnan(window)]
    if len(clean_window) == 0:
        return [0.0] * 10
    
    features = {
        'min': np.min(clean_window),
        'max': np.max(clean_window),
        'mean': np.mean(clean_window),
        'std': np.std(clean_window) if len(clean_window) > 1 else 0.0,
        'median': np.median(clean_window),
        'range': np.ptp(clean_window),  # peak-to-peak
        'q25': np.percentile(clean_window, 25),
        'q75': np.percentile(clean_window, 75)
    }
    
    # Handle skewness and kurtosis with error handling
    from scipy.stats import skew, kurtosis
    try:
        features['skewness'] = skew(clean_window) if len(clean_window) > 2 else 0.0
        features['kurtosis'] = kurtosis(clean_window) if len(clean_window) > 3 else 0.0
    except:
        features['skewness'] = 0.0
        features['kurtosis'] = 0.0
    
    # Replace any inf or nan values
    feature_list = list(features.values())
    feature_list = [0.0 if (np.isnan(f) or np.isinf(f)) else f for f in feature_list]
    
    return feature_list

def create_sequences(patient_data, window_size, prediction_horizon):
    """Create input sequences and labels for classification"""
    bg_values = patient_data['BGLevel'].values
    
    # Apply smoothing filter
    bg_smoothed = apply_savitzky_golay_filter(bg_values)
    
    X_windows = []
    X_features = []
    y_labels = []
    
    for i in range(len(bg_smoothed) - window_size - prediction_horizon):
        # Input window (last 30 minutes)
        window = bg_smoothed[i:i + window_size]
        
        # Future value (15 minutes ahead)
        future_value = bg_smoothed[i + window_size + prediction_horizon - 1]
        
        # Classification label: 1 if hyperglycemic, 0 otherwise
        label = 1 if future_value > HYPER_THRESHOLD else 0
        
        X_windows.append(window)
        X_features.append(extract_time_domain_features(window))
        y_labels.append(label)
    
    return np.array(X_windows), np.array(X_features), np.array(y_labels)

def prepare_dataset(df):
    """Prepare dataset for all patients"""
    all_X_windows = []
    all_X_features = []
    all_y = []
    
    patients = df['PtID'].unique()
    
    for patient_id in patients:
        patient_data = df[df['PtID'] == patient_id].copy()
        
        if len(patient_data) < WINDOW_SIZE + PREDICTION_HORIZON + 10:
            print(f"Skipping patient {patient_id}: insufficient data")
            continue
        
        X_win, X_feat, y = create_sequences(patient_data, WINDOW_SIZE, PREDICTION_HORIZON)
        
        if len(X_win) > 0:
            all_X_windows.append(X_win)
            all_X_features.append(X_feat)
            all_y.append(y)
            
            hyper_count = np.sum(y)
            print(f"Patient {patient_id}: {len(y)} samples, {hyper_count} hyperglycemic ({hyper_count/len(y)*100:.1f}%)")
    
    X_windows = np.concatenate(all_X_windows, axis=0)
    X_features = np.concatenate(all_X_features, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    # FIX #3: Clean any NaN values from preprocessing artifacts
    print("\nCleaning data...")
    print(f"NaN in X_windows: {np.isnan(X_windows).sum()}")
    print(f"NaN in X_features: {np.isnan(X_features).sum()}")
    print(f"Inf in X_windows: {np.isinf(X_windows).sum()}")
    print(f"Inf in X_features: {np.isinf(X_features).sum()}")
    
    X_windows = np.nan_to_num(X_windows, nan=0.0, posinf=0.0, neginf=0.0)
    X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\nTotal dataset: {len(y)} samples")
    print(f"Hyperglycemic events: {np.sum(y)} ({np.sum(y)/len(y)*100:.1f}%)")
    print(f"Normal events: {len(y) - np.sum(y)} ({(len(y)-np.sum(y))/len(y)*100:.1f}%)")
    
    return X_windows, X_features, y

# ==================== CNN-LSTM MODEL ====================

def build_cnn_lstm_model(window_size, n_features):
    """
    Build CNN-LSTM hybrid model for hyperglycemia classification
    
    Architecture:
    - CNN layers: Extract local patterns and features from time series
    - LSTM layers: Capture temporal dependencies
    - Dense layers: Final classification
    """
    
    # Input 1: Time series window (BG values over time)
    input_window = layers.Input(shape=(window_size, 1), name='bg_window')
    
    # CNN branch for feature extraction
    # Conv1D extracts local patterns (trends, spikes)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_window)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # LSTM layers for temporal dependencies
    x = layers.LSTM(100, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.LSTM(50, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    
    # Input 2: Statistical time-domain features
    input_features = layers.Input(shape=(n_features,), name='time_features')
    
    # Dense processing of statistical features
    f = layers.Dense(32, activation='relu')(input_features)
    f = layers.Dropout(0.2)(f)
    f = layers.Dense(16, activation='relu')(f)
    
    # Concatenate both branches
    combined = layers.Concatenate()([x, f])
    
    # Final classification layers
    z = layers.Dense(64, activation='relu')(combined)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(32, activation='relu')(z)
    z = layers.Dropout(0.2)(z)
    
    # Output: Binary classification (hyperglycemic or not)
    output = layers.Dense(1, activation='sigmoid', name='hyperglycemia')(z)
    
    # Create model
    model = models.Model(inputs=[input_window, input_features], outputs=output)
    
    return model

# ==================== TRAINING ====================

def train_model(X_windows, X_features, y, epochs=100, batch_size=32):
    """Train the CNN-LSTM model"""
    
    # Additional data validation
    print("\nValidating input data...")
    print(f"X_windows shape: {X_windows.shape}, range: [{X_windows.min():.2f}, {X_windows.max():.2f}]")
    print(f"X_features shape: {X_features.shape}, range: [{X_features.min():.2f}, {X_features.max():.2f}]")
    print(f"y shape: {y.shape}, unique values: {np.unique(y)}")
    
    # Normalize data
    scaler_window = MinMaxScaler()
    scaler_features = MinMaxScaler()
    
    X_windows_scaled = scaler_window.fit_transform(X_windows.reshape(-1, 1)).reshape(X_windows.shape)
    X_features_scaled = scaler_features.fit_transform(X_features)
    
    # Verify scaling didn't introduce NaNs
    assert not np.any(np.isnan(X_windows_scaled)), "NaN detected in scaled windows"
    assert not np.any(np.isnan(X_features_scaled)), "NaN detected in scaled features"
    assert not np.any(np.isinf(X_windows_scaled)), "Inf detected in scaled windows"
    assert not np.any(np.isinf(X_features_scaled)), "Inf detected in scaled features"
    
    # Reshape for CNN-LSTM (samples, timesteps, features)
    X_windows_scaled = X_windows_scaled.reshape(X_windows_scaled.shape[0], X_windows_scaled.shape[1], 1)
    
    # Split data
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=1-TRAIN_SPLIT, 
                                           stratify=y, random_state=42)
    
    X_win_train = X_windows_scaled[train_idx]
    X_feat_train = X_features_scaled[train_idx]
    y_train = y[train_idx]
    
    X_win_test = X_windows_scaled[test_idx]
    X_feat_test = X_features_scaled[test_idx]
    y_test = y[test_idx]
    
    print(f"\nTrain set: {len(y_train)} samples (Hyperglycemic: {np.sum(y_train)})")
    print(f"Test set: {len(y_test)} samples (Hyperglycemic: {np.sum(y_test)})")
    
    # Build model
    model = build_cnn_lstm_model(window_size=WINDOW_SIZE, n_features=X_features.shape[1])
    
    # FIX #1: Prevent division by zero in class weights
    n_positive = np.sum(y_train)
    n_negative = len(y_train) - n_positive
    
    if n_positive > 0 and n_negative > 0:
        class_weight = {
            0: 1.0,
            1: n_negative / (n_positive + 1e-7)  # Add epsilon to prevent division by zero
        }
    else:
        # Fallback if one class is missing
        class_weight = {0: 1.0, 1: 1.0}
    
    print(f"\nClass weights: {class_weight}")
    print(f"Positive samples: {n_positive}, Negative samples: {n_negative}")
    
    # FIX #2: Add gradient clipping to prevent exploding gradients
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),  # Lower LR + Gradient Clipping
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(),
                 keras.metrics.AUC(name='auc')]
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, 
                                         restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                            patience=5, min_lr=1e-6)
    
    # Train
    print("\n" + "="*50)
    print("TRAINING CNN-LSTM MODEL")
    print("="*50)
    
    history = model.fit(
        [X_win_train, X_feat_train], y_train,
        validation_data=([X_win_test, X_feat_test], y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    return model, history, (X_win_test, X_feat_test, y_test), (scaler_window, scaler_features)

# ==================== EVALUATION ====================

def evaluate_model(model, X_win_test, X_feat_test, y_test):
    """Evaluate model performance"""
    
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Predictions
    y_pred_proba = model.predict([X_win_test, X_feat_test]).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Classification metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Normal', 'Hyperglycemic']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nSensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    
    # ROC-AUC
    if len(np.unique(y_test)) > 1:
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC Score: {auc_score:.4f}")
    
    return y_pred, y_pred_proba, cm

def plot_results(history, cm, y_test, y_pred_proba):
    """Plot training history and evaluation metrics"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Training history
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(history.history['accuracy'], label='Train Acc')
    ax2.plot(history.history['val_accuracy'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Confusion Matrix
    ax3 = plt.subplot(2, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Hyper'],
                yticklabels=['Normal', 'Hyper'], ax=ax3)
    ax3.set_title('Confusion Matrix')
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    
    # ROC Curve
    if len(np.unique(y_test)) > 1:
        ax4 = plt.subplot(2, 3, 4)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        ax4.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})')
        ax4.plot([0, 1], [0, 1], 'k--', label='Random')
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title('ROC Curve')
        ax4.legend()
        ax4.grid(True)
    
    # Precision-Recall Curve
    ax5 = plt.subplot(2, 3, 5)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ax5.plot(recall, precision)
    ax5.set_xlabel('Recall')
    ax5.set_ylabel('Precision')
    ax5.set_title('Precision-Recall Curve')
    ax5.grid(True)
    
    # Prediction distribution
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.5, label='Normal', color='blue')
    ax6.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.5, label='Hyperglycemic', color='red')
    ax6.axvline(0.5, color='black', linestyle='--', label='Threshold')
    ax6.set_xlabel('Predicted Probability')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Prediction Distribution')
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('hyperglycemia_cnn_lstm_results.png', dpi=300, bbox_inches='tight')
    print("\nPlots saved to 'hyperglycemia_cnn_lstm_results.png'")
    plt.show()

# ==================== MAIN EXECUTION ====================

def main(filepath):
    """Main execution pipeline"""
    
    print("="*60)
    print("HYPERGLYCEMIA CLASSIFICATION USING CNN-LSTM")
    print("15-minute Prediction Horizon")
    print("="*60)
    
    # Load and preprocess data
    df = load_and_preprocess_data(filepath)
    
    # Prepare dataset
    X_windows, X_features, y = prepare_dataset(df)
    
    # Train model
    model, history, test_data, scalers = train_model(
        X_windows, X_features, y, 
        epochs=100, 
        batch_size=32
    )
    
    # Evaluate
    X_win_test, X_feat_test, y_test = test_data
    y_pred, y_pred_proba, cm = evaluate_model(model, X_win_test, X_feat_test, y_test)
    
    # Plot results
    plot_results(history, cm, y_test, y_pred_proba)
    
    # Save model
    model.save('hyperglycemia_cnn_lstm_model.h5')
    print("\nModel saved to 'hyperglycemia_cnn_lstm_model.h5'")
    
    return model, history, scalers

# ==================== USAGE ====================

if __name__ == "__main__":
    # Example usage
    filepath = "your_glucose_data.csv"  # Replace with your CSV file path
    
    # Run the pipeline
    model, history, scalers = main(filepath)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)