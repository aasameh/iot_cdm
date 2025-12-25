import numpy as np
import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesAugmenter:
    """Augments PPG/ECG/ABP time-series data"""
    
    def __init__(self, sampling_rate=125, window_size=1250, random_state=42):
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.random_state = random_state
        np.random.seed(random_state)
    
    def extract_windows(self, df, label_column='label', overlap=0.5):
        """Extract fixed-size windows from continuous time-series"""
        stride = int(self.window_size * (1 - overlap))
        windows = []
        
        for i in range(0, len(df) - self.window_size, stride):
            window_df = df.iloc[i:i + self.window_size].copy()
            
            # Get majority label in window
            label_counts = window_df[label_column].value_counts()
            majority_label = label_counts.idxmax()
            
            windows.append({
                'data': window_df,
                'label': majority_label,
                'start_idx': i,
                'end_idx': i + self.window_size
            })
        
        return windows
    
    def augment_window(self, window_data, aug_type='jitter'):
        """Apply augmentation to a window (returns augmented DataFrame)"""
        augmented_df = window_data.copy()
        
        if aug_type == 'jitter':
            # Add Gaussian noise
            for col in ['ECG', 'ABP', 'PPG']:
                if col in augmented_df.columns:
                    noise = np.random.normal(0, 0.03 * augmented_df[col].std(), 
                                            len(augmented_df))
                    augmented_df[col] += noise
        
        elif aug_type == 'scale':
            # Scale magnitude
            for col in ['ECG', 'ABP', 'PPG']:
                if col in augmented_df.columns:
                    scale_factor = np.random.uniform(0.9, 1.1)
                    augmented_df[col] *= scale_factor
        
        elif aug_type == 'time_warp':
            # Time warping (stretch/compress)
            for col in ['ECG', 'ABP', 'PPG']:
                if col in augmented_df.columns:
                    orig_indices = np.arange(len(augmented_df))
                    warp_indices = orig_indices * np.random.uniform(0.95, 1.05)
                    warp_indices = np.clip(warp_indices, 0, len(augmented_df)-1)
                    augmented_df[col] = np.interp(orig_indices, warp_indices, 
                                                   augmented_df[col].values)
        
        elif aug_type == 'baseline_wander':
            # Add low-frequency drift
            for col in ['ECG', 'ABP', 'PPG']:
                if col in augmented_df.columns:
                    t = np.linspace(0, 10, len(augmented_df))
                    drift = np.sin(2 * np.pi * 0.1 * t) * 0.05 * augmented_df[col].std()
                    augmented_df[col] += drift
        
        # Recalculate MAP if ABP was augmented
        if 'ABP' in augmented_df.columns:
            augmented_df['MAP'] = augmented_df['ABP']  # Simplified
        
        return augmented_df
    
    def balance_windows(self, windows, target_count=None, strategy='oversample'):
        """Balance classes by oversampling minority classes"""
        
        # Count samples per class
        label_counts = Counter([w['label'] for w in windows])
        print(f"\nOriginal window distribution: {dict(label_counts)}")
        
        if target_count is None:
            target_count = max(label_counts.values())
        
        print(f"Target samples per class: {target_count}")
        
        balanced_windows = []
        
        for label in label_counts.keys():
            label_windows = [w for w in windows if w['label'] == label]
            current_count = len(label_windows)
            
            # Add all original windows
            balanced_windows.extend(label_windows)
            
            # Oversample if needed
            if current_count < target_count:
                needed = target_count - current_count
                print(f"  {label}: {current_count} → {target_count} (augmenting {needed})")
                
                # Augment randomly selected windows
                aug_types = ['jitter', 'scale', 'time_warp', 'baseline_wander']
                
                for _ in range(needed):
                    # Pick random original window
                    orig_window = label_windows[np.random.randint(len(label_windows))]
                    
                    # Apply random augmentation
                    aug_type = np.random.choice(aug_types)
                    augmented_data = self.augment_window(orig_window['data'], aug_type)
                    
                    balanced_windows.append({
                        'data': augmented_data,
                        'label': label,
                        'augmented': True,
                        'aug_type': aug_type
                    })
            else:
                print(f"  {label}: {current_count} (no augmentation needed)")
        
        print(f"\nBalanced dataset: {len(balanced_windows)} total windows")
        return balanced_windows


class CompletePipelineCSV:
    """
    Pipeline that saves augmented data as CSV files (for spectrogram generation)
    """
    
    def __init__(self, sampling_rate=125, window_size=1250, 
                 overlap=0.5, random_state=42):
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        self.random_state = random_state
        self.augmenter = TimeSeriesAugmenter(
            sampling_rate=sampling_rate,
            window_size=window_size,
            random_state=random_state
        )
    
    def load_all_labeled_files(self, directory):
        """Load all labeled CSV files"""
        csv_files = glob.glob(os.path.join(directory, '*_labeled.csv'))
        
        if not csv_files:
            raise ValueError(f"No labeled CSV files found in {directory}")
        
        print(f"Found {len(csv_files)} labeled files")
        
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            print(f"  Loaded: {os.path.basename(csv_file)} ({len(df)} samples)")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\nCombined dataset: {len(combined_df)} total samples")
        
        return combined_df
    
    def create_balanced_csv_dataset(self, input_directory, output_directory,
                                    target_per_class=None, test_size=0.2, val_size=0.1):
        """
        Main pipeline: load, window, balance, split, SAVE AS CSV
        
        Args:
            input_directory: Directory with *_labeled.csv files
            output_directory: Where to save train/val/test CSV files
            target_per_class: Target windows per class (None = auto)
            test_size: Test set proportion
            val_size: Validation set proportion
        """
        os.makedirs(output_directory, exist_ok=True)
        
        print("="*70)
        print("STEP 1: Loading labeled data files")
        print("="*70)
        
        df = self.load_all_labeled_files(input_directory)
        
        print("\nOriginal label distribution:")
        label_dist = df['label'].value_counts()
        for label, count in label_dist.items():
            print(f"  {label}: {count:,} ({count/len(df)*100:.2f}%)")
        
        # ================================================================
        print("\n" + "="*70)
        print("STEP 2: Extracting windows from time-series")
        print("="*70)
        
        windows = self.augmenter.extract_windows(
            df, 
            label_column='label',
            overlap=self.overlap
        )
        
        print(f"\nExtracted {len(windows)} windows")
        window_labels = [w['label'] for w in windows]
        print("\nWindow label distribution:")
        for label, count in Counter(window_labels).items():
            print(f"  {label}: {count:,}")
        
        # ================================================================
        print("\n" + "="*70)
        print("STEP 3: Splitting into train/test BEFORE augmentation")
        print("="*70)
        
        train_windows, test_windows = train_test_split(
            windows,
            test_size=test_size,
            stratify=[w['label'] for w in windows],
            random_state=self.random_state
        )
        
        print(f"\nTrain windows: {len(train_windows)}")
        print(f"Test windows: {len(test_windows)}")
        
        # ================================================================
        print("\n" + "="*70)
        print("STEP 4: Balancing ONLY training set with augmentation")
        print("="*70)
        
        balanced_train_windows = self.augmenter.balance_windows(
            train_windows,
            target_count=target_per_class,
            strategy='oversample'
        )
        
        # ================================================================
        print("\n" + "="*70)
        print("STEP 5: Creating validation split")
        print("="*70)
        
        train_final_windows, val_windows = train_test_split(
            balanced_train_windows,
            test_size=val_size/(1-test_size),
            stratify=[w['label'] for w in balanced_train_windows],
            random_state=self.random_state
        )
        
        print(f"\nFinal train windows: {len(train_final_windows)}")
        print(f"Validation windows: {len(val_windows)}")
        print(f"Test windows: {len(test_windows)}")
        
        # ================================================================
        print("\n" + "="*70)
        print("STEP 6: Saving as CSV files for spectrogram generation")
        print("="*70)
        
        # Save each split as CSV
        def save_windows_to_csv(windows, filename):
            """Concatenate all window dataframes and save"""
            all_dfs = []
            for window in tqdm(windows, desc=f"Preparing {filename}"):
                all_dfs.append(window['data'])
            
            combined = pd.concat(all_dfs, ignore_index=True)
            filepath = os.path.join(output_directory, filename)
            combined.to_csv(filepath, index=False)
            print(f"  Saved: {filepath} ({len(combined)} samples)")
            return combined
        
        train_df = save_windows_to_csv(train_final_windows, 'train_balanced_labeled.csv')
        val_df = save_windows_to_csv(val_windows, 'val_labeled.csv')
        test_df = save_windows_to_csv(test_windows, 'test_labeled.csv')
        
        # ================================================================
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        
        print("\nClass distribution in each CSV:")
        for name, df_split in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            print(f"\n  {name}:")
            counts = df_split['label'].value_counts()
            for label, count in counts.items():
                pct = count / len(df_split) * 100
                print(f"    {label}: {count:,} ({pct:.1f}%)")
        
        print("\n" + "="*70)
        print("NEXT STEP: Run spectrogram generation on these CSV files!")
        print("="*70)
        print(f"\nFiles saved in: {output_directory}/")
        print("  - train_balanced_labeled.csv  (balanced with augmentation)")
        print("  - val_labeled.csv             (original distribution)")
        print("  - test_labeled.csv            (original distribution)")
        print("\nNow run your spectrogram generator on these files:")
        print(f"  python ppg_spectrogram_generator.py --input {output_directory}")
        
        return {
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'output_dir': output_directory
        }


# Standalone usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = CompletePipelineCSV(
        sampling_rate=125,
        window_size=1250,  # 10 seconds
        overlap=0.5,       # 50% overlap
        random_state=42
    )
    
    # Process all labeled files and create balanced CSV dataset
    result = pipeline.create_balanced_csv_dataset(
        input_directory='hypertensive_labeled_output_fixed',
        output_directory='balanced_csv_for_spectrograms',
        target_per_class=None,  # Auto: match largest class
        test_size=0.2,          # 20% test
        val_size=0.1            # 10% validation
    )
    
    print("\n✓ Done! Now run spectrogram generation on the balanced CSV files.")