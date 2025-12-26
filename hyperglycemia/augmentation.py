"""
CGM Data Augmentation Script
Balances class distribution for hyperglycemia prediction
Generates: augmented_glucose_data.csv
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# ==================== CONFIGURATION ====================
WINDOW_SIZE = 6  # 30 minutes of history (5-min intervals)
PREDICTION_HORIZON = 3  # 15 minutes ahead
HYPER_THRESHOLD = 180  # mg/dL
TARGET_BALANCE_RATIO = 1.0  # 1:1 ratio of classes

# ==================== HELPER FUNCTIONS ====================

def apply_savitzky_golay_filter(bg_values, window_length=15, polyorder=1):
    """Apply Savitzky-Golay filter to smooth noise"""
    if len(bg_values) < window_length:
        return bg_values
    return savgol_filter(bg_values, window_length, polyorder)

def extract_time_domain_features(window):
    """Extract statistical time-domain features from a window"""
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

# ==================== SEQUENCE CREATION ====================

def create_sequences_with_overlap(patient_data, window_size, prediction_horizon, stride=1):
    """Create sequences with overlapping windows"""
    bg_values = patient_data['BGLevel'].values
    bg_smoothed = apply_savitzky_golay_filter(bg_values)
    
    sequences = []
    
    for i in range(0, len(bg_smoothed) - window_size - prediction_horizon, stride):
        window = bg_smoothed[i:i + window_size]
        future_value = bg_smoothed[i + window_size + prediction_horizon - 1]
        label = 1 if future_value > HYPER_THRESHOLD else 0
        
        # Get corresponding timestamps
        start_idx = i
        end_idx = i + window_size
        future_idx = i + window_size + prediction_horizon - 1
        
        sequences.append({
            'window': window,
            'features': extract_time_domain_features(window),
            'label': label,
            'patient_id': patient_data.iloc[start_idx]['PtID'],
            'start_time': patient_data.iloc[start_idx]['DateTime'],
            'end_time': patient_data.iloc[end_idx-1]['DateTime'],
            'future_time': patient_data.iloc[future_idx]['DateTime'],
            'future_value': future_value
        })
    
    return sequences

# ==================== TIME SERIES AUGMENTATION ====================

def augment_time_series_sequence(window, num_augmentations=1):
    """
    Generate augmented versions of a time series window
    Returns list of augmented windows
    """
    augmented_windows = []
    
    for _ in range(num_augmentations):
        method = np.random.choice(['jitter', 'scale', 'time_warp', 'combination'])
        
        if method == 'jitter':
            # Add Gaussian noise (±2-3% of std)
            noise = np.random.normal(0, np.std(window) * 0.025, window.shape)
            augmented = window + noise
            
        elif method == 'scale':
            # Scale by 0.96-1.04
            scale_factor = np.random.uniform(0.96, 1.04)
            augmented = window * scale_factor
            
        elif method == 'time_warp':
            # Time warping using interpolation
            old_indices = np.arange(len(window))
            warp_factor = np.random.uniform(0.92, 1.08)
            new_indices = np.linspace(0, len(window)-1, int(len(window) * warp_factor))
            
            f = interp1d(old_indices, window, kind='cubic', fill_value='extrapolate') # type: ignore
            warped = f(np.linspace(new_indices[0], new_indices[-1], len(window)))
            augmented = warped
            
        else:  # combination
            noise = np.random.normal(0, np.std(window) * 0.02, window.shape)
            scale = np.random.uniform(0.97, 1.03)
            augmented = (window * scale) + noise
        
        # Ensure realistic values (40-400 mg/dL)
        augmented = np.clip(augmented, 40, 400)
        augmented_windows.append(augmented)
    
    return augmented_windows

# ==================== MAIN AUGMENTATION PIPELINE ====================

def load_and_preprocess_data(filepath):
    """Load CSV data"""
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Remove rows with zero or missing BG levels
    df = df[df['BGLevel'] > 0].copy()
    
    # Convert date/time
    df['DateTime'] = pd.to_datetime(df['BGDate'] + ' ' + df['BGTime']) # type: ignore
    df = df.sort_values(['PtID', 'DateTime'])
    
    print(f"Loaded {len(df)} records from {df['PtID'].nunique()} patients")
    return df

def extract_sequences_from_data(df):
    """Extract all sequences from original data"""
    print("\nExtracting sequences from original data...")
    
    all_sequences = []
    patients = df['PtID'].unique()
    
    for patient_id in patients:
        patient_data = df[df['PtID'] == patient_id].copy()
        
        if len(patient_data) < WINDOW_SIZE + PREDICTION_HORIZON + 10:
            print(f"Skipping patient {patient_id}: insufficient data")
            continue
        
        # Standard stride for majority class
        sequences = create_sequences_with_overlap(
            patient_data, WINDOW_SIZE, PREDICTION_HORIZON, stride=6
        )
        all_sequences.extend(sequences)
        
        # Smaller stride for minority class regions
        hyper_mask = patient_data['BGLevel'] > (HYPER_THRESHOLD - 20)
        if hyper_mask.sum() > WINDOW_SIZE + PREDICTION_HORIZON:
            hyper_data = patient_data[hyper_mask].copy()
            if len(hyper_data) >= WINDOW_SIZE + PREDICTION_HORIZON:
                hyper_sequences = create_sequences_with_overlap(
                    hyper_data, WINDOW_SIZE, PREDICTION_HORIZON, stride=1
                )
                all_sequences.extend(hyper_sequences)
        
        hyper_count = sum(1 for s in sequences if s['label'] == 1)
        print(f"Patient {patient_id}: {len(sequences)} sequences, "
              f"{hyper_count} hyperglycemic ({hyper_count/len(sequences)*100:.1f}%)")
    
    return all_sequences

def balance_with_time_series_augmentation(sequences):
    """Apply time series augmentation to balance classes"""
    print("\n" + "="*60)
    print("APPLYING TIME SERIES AUGMENTATION")
    print("="*60)
    
    # Separate by class
    normal_sequences = [s for s in sequences if s['label'] == 0]
    hyper_sequences = [s for s in sequences if s['label'] == 1]
    
    n_normal = len(normal_sequences)
    n_hyper = len(hyper_sequences)
    
    print(f"\nOriginal distribution:")
    print(f"  Normal: {n_normal}")
    print(f"  Hyperglycemic: {n_hyper}")
    print(f"  Ratio: {n_hyper/n_normal:.3f}")
    
    # Calculate how many augmented samples needed
    target_hyper = int(n_normal * TARGET_BALANCE_RATIO)
    n_augmentations_needed = max(0, target_hyper - n_hyper)
    
    if n_augmentations_needed == 0:
        print("\nClasses already balanced!")
        return sequences
    
    print(f"\nTarget hyperglycemic samples: {target_hyper}")
    print(f"Augmentations needed: {n_augmentations_needed}")
    
    # Generate augmented sequences
    augmented_sequences = []
    augmentations_per_sample = max(1, n_augmentations_needed // n_hyper)
    
    print(f"Augmentations per sample: {augmentations_per_sample}")
    
    for seq in hyper_sequences:
        aug_windows = augment_time_series_sequence(
            seq['window'], 
            num_augmentations=augmentations_per_sample
        )
        
        for aug_window in aug_windows:
            aug_seq = {
                'window': aug_window,
                'features': extract_time_domain_features(aug_window),
                'label': 1,  # Keep label
                'patient_id': f"{seq['patient_id']}_aug",
                'start_time': seq['start_time'],
                'end_time': seq['end_time'],
                'future_time': seq['future_time'],
                'future_value': seq['future_value']
            }
            augmented_sequences.append(aug_seq)
    
    # Trim to exact target if overshot
    if len(augmented_sequences) > n_augmentations_needed:
        augmented_sequences = augmented_sequences[:n_augmentations_needed]
    
    # Combine all sequences
    balanced_sequences = sequences + augmented_sequences
    
    n_final_hyper = sum(1 for s in balanced_sequences if s['label'] == 1)
    n_final_normal = sum(1 for s in balanced_sequences if s['label'] == 0)
    
    print(f"\nFinal distribution:")
    print(f"  Normal: {n_final_normal}")
    print(f"  Hyperglycemic: {n_final_hyper}")
    print(f"  Ratio: {n_final_hyper/n_final_normal:.3f}")
    print(f"  Total sequences: {len(balanced_sequences)}")
    
    return balanced_sequences

def apply_smote_to_sequences(sequences):
    """Apply SMOTE for final balancing"""
    print("\n" + "="*60)
    print("APPLYING SMOTE")
    print("="*60)
    
    # Convert sequences to arrays
    X_windows = np.array([s['window'] for s in sequences])
    X_features = np.array([s['features'] for s in sequences])
    y = np.array([s['label'] for s in sequences])
    
    # Combine for SMOTE
    X_combined = np.concatenate([
        X_windows.reshape(X_windows.shape[0], -1),
        X_features
    ], axis=1)
    
    print(f"\nBefore SMOTE:")
    print(f"  Shape: {X_combined.shape}")
    print(f"  Normal: {np.sum(y == 0)}")
    print(f"  Hyperglycemic: {np.sum(y == 1)}")
    
    # Apply SMOTE
    smote = SMOTE(random_state=42, k_neighbors=min(5, np.sum(y == 1) - 1))
    X_balanced, y_balanced = smote.fit_resample(X_combined, y) # type: ignore
    
    # Reconstruct sequences
    window_size = WINDOW_SIZE
    X_windows_balanced = X_balanced[:, :window_size]
    X_features_balanced = X_balanced[:, window_size:]
    
    print(f"\nAfter SMOTE:")
    print(f"  Shape: {X_balanced.shape}")
    print(f"  Normal: {np.sum(y_balanced == 0)}")
    print(f"  Hyperglycemic: {np.sum(y_balanced == 1)}")
    
    # Find the latest timestamp in original data for synthetic data
    max_time = max(s['future_time'] for s in sequences if isinstance(s['future_time'], pd.Timestamp))
    
    # Create new sequence objects with sequential synthetic timestamps
    balanced_sequences = []
    synthetic_counter = 0
    
    for i in range(len(y_balanced)):
        # Check if this is an original sequence (first len(sequences) are originals)
        if i < len(sequences):
            # Keep original sequence
            balanced_sequences.append(sequences[i])
        else:
            # This is a SMOTE-generated synthetic sequence
            # Create sequential timestamps after the latest real data
            synthetic_start = max_time + pd.Timedelta(days=synthetic_counter, hours=1)
            synthetic_end = synthetic_start + pd.Timedelta(minutes=5 * (window_size - 1))
            synthetic_future = synthetic_end + pd.Timedelta(minutes=5 * PREDICTION_HORIZON)
            
            seq = {
                'window': X_windows_balanced[i],
                'features': X_features_balanced[i],
                'label': y_balanced[i],
                'patient_id': f'smote_syn_{synthetic_counter}',
                'start_time': synthetic_start,
                'end_time': synthetic_end,
                'future_time': synthetic_future,
                'future_value': HYPER_THRESHOLD + 10 if y_balanced[i] == 1 else HYPER_THRESHOLD - 10 # type: ignore
            }
            balanced_sequences.append(seq)
            synthetic_counter += 1
    
    print(f"\nSMOTE generated {synthetic_counter} synthetic sequences")
    
    return balanced_sequences

def sequences_to_dataframe(sequences):
    """Convert sequences to DataFrame format for saving"""
    print("\nConverting sequences to DataFrame...")
    
    rows = []
    rec_id = 1
    
    for seq_idx, seq in enumerate(sequences):
        patient_id = seq['patient_id']
        window = seq['window']
        label = seq['label']
        
        # Create records for each point in the window
        for i, bg_value in enumerate(window):
            # Create time offset (5 minutes per step)
            time_offset = pd.Timedelta(minutes=5*i)
            
            # Use original timestamps or create synthetic ones
            if isinstance(seq['start_time'], pd.Timestamp):
                timestamp = seq['start_time'] + time_offset
            else:
                timestamp = pd.Timestamp('2000-01-01') + pd.Timedelta(days=seq_idx) + time_offset
            
            row = {
                'RecID': rec_id,
                'PtID': patient_id,
                'BGDate': timestamp.strftime('%Y-%m-%d'),
                'BGTime': timestamp.strftime('%I:%M:%S %p'),
                'BGLevel': float(bg_value),
                'Calibration': False,
                'FileType': 'Augmented' if 'aug' in str(patient_id) or 'smote' in str(patient_id) else 'Guardian #1',
                'SequenceID': seq_idx,
                'Label': label
            }
            rows.append(row)
            rec_id += 1
    
    df = pd.DataFrame(rows)
    print(f"Created DataFrame with {len(df)} records")
    return df

# ==================== MAIN EXECUTION ====================

def main(input_filepath, output_filepath='augmented_glucose_data.csv', 
         use_smote=True, use_time_series_aug=True):
    """
    Main augmentation pipeline
    
    Args:
        input_filepath: Path to original CSV file
        output_filepath: Path to save augmented CSV
        use_smote: Whether to apply SMOTE
        use_time_series_aug: Whether to apply time series augmentation
    """
    
    print("="*60)
    print("CGM DATA AUGMENTATION FOR CLASS BALANCING")
    print("="*60)
    
    # Load original data
    df = load_and_preprocess_data(input_filepath)
    
    # Extract sequences
    sequences = extract_sequences_from_data(df)
    
    if len(sequences) == 0:
        print("\nERROR: No sequences extracted! Check your data.")
        return
    
    # Apply augmentation techniques
    if use_time_series_aug:
        sequences = balance_with_time_series_augmentation(sequences)
    
    if use_smote:
        sequences = apply_smote_to_sequences(sequences)
    
    # Convert to DataFrame
    augmented_df = sequences_to_dataframe(sequences)
    
    # Save to CSV
    augmented_df.to_csv(output_filepath, index=False)
    
    print("\n" + "="*60)
    print("AUGMENTATION COMPLETE!")
    print("="*60)
    print(f"\nAugmented data saved to: {output_filepath}")
    print(f"Total records: {len(augmented_df)}")
    print(f"Total sequences: {augmented_df['SequenceID'].nunique()}")
    
    # Summary statistics
    label_counts = augmented_df.groupby('Label').size()
    print(f"\nFinal class distribution (by sequence):")
    print(f"  Normal (0): {label_counts.get(0, 0)}")
    print(f"  Hyperglycemic (1): {label_counts.get(1, 0)}")
    
    return augmented_df

# ==================== USAGE ====================

if __name__ == "__main__":
    # Configuration
    input_file = "hyperglycemia\\DirecNetCounter-RegulatoryStudy\\DataTables\\tblJDataGuardian.csv"  # Your original data
    output_file = "augmented_glucose_data.csv"  # Output file
    
    # Run augmentation
    # Options:
    # 1. Time series augmentation only
    # augmented_df = main(input_file, output_file, use_smote=False, use_time_series_aug=True)
    
    # 2. SMOTE only
    # augmented_df = main(input_file, output_file, use_smote=True, use_time_series_aug=False)
    
    # 3. Both (recommended)
    augmented_df = main(input_file, output_file, use_smote=True, use_time_series_aug=True)
    
    print("\n✓ Ready to use with your CNN-LSTM training script!")
    print(f"  Simply use: df = pd.read_csv('{output_file}')")