import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm


class SimpleAugmenter:
    def __init__(self, noise_level_range=(0.03, 0.08)):
        """
        Simple augmenter that adds Gaussian noise to PPG signals.
        
        Args:
            noise_level_range: Tuple (min, max) for noise level as fraction of std
        """
        self.noise_level_range = noise_level_range
        print("Simple PPG Augmenter")
        print(f"  Noise level range: {noise_level_range[0]}-{noise_level_range[1]} * std(PPG)")
    
    def augment_ppg_signal(self, ppg, noise_level=None):
        """
        Add Gaussian noise to PPG signal.
        
        Args:
            ppg: 1D array of PPG values
            noise_level: Noise level as fraction of std (None = random from range)
            
        Returns:
            Augmented PPG signal
        """
        if noise_level is None:
            noise_level = np.random.uniform(*self.noise_level_range)
        
        noise = np.random.normal(0, noise_level * np.std(ppg), len(ppg))
        return ppg + noise
    
    def augment_file(self, input_csv, output_dir, num_versions=5):
        """
        Create augmented versions of a labeled CSV file.
        Only PPG column changes, everything else (ECG, ABP, MAP, label) stays the same.
        
        Args:
            input_csv: Path to input CSV file
            output_dir: Directory to save augmented files
            num_versions: Number of augmented versions to create
            
        Returns:
            List of paths to created files
        """
        # Load original
        df = pd.read_csv(input_csv)
        
        if 'PPG' not in df.columns:
            raise ValueError(f"PPG column not found in {input_csv}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base name without extension
        base_name = os.path.splitext(os.path.basename(input_csv))[0]
        
        created_files = []
        
        # Create augmented versions
        for version in range(num_versions):
            # Copy dataframe
            df_aug = df.copy()
            
            # Augment PPG only
            df_aug['PPG'] = self.augment_ppg_signal(df['PPG'].values)
            
            # Save with version suffix
            output_path = os.path.join(output_dir, f"{base_name}_aug{version+1}.csv")
            df_aug.to_csv(output_path, index=False)
            created_files.append(output_path)
        
        return created_files
    
    def identify_minority_files(self, labeled_csv_files, min_positive=1, min_hypertensive=1):
        """
        Identify files that contain positive or hypertensive labels.
        
        Args:
            labeled_csv_files: List of paths to labeled CSV files
            min_positive: Minimum positive samples to consider file
            min_hypertensive: Minimum hypertensive samples to consider file
            
        Returns:
            Dictionary with file info
        """
        print("\nAnalyzing files for minority classes...")
        
        files_with_positive = []
        files_with_hypertensive = []
        
        for csv_file in tqdm(labeled_csv_files, desc="Scanning files"):
            try:
                df = pd.read_csv(csv_file)
                
                if 'label' not in df.columns:
                    continue
                
                label_counts = df['label'].value_counts()
                
                has_positive = label_counts.get('positive', 0) >= min_positive
                has_hypertensive = label_counts.get('hypertensive_event', 0) >= min_hypertensive
                
                if has_positive:
                    files_with_positive.append(csv_file)
                if has_hypertensive:
                    files_with_hypertensive.append(csv_file)
                    
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue
        
        print(f"\nFound:")
        print(f"  Files with positive labels: {len(files_with_positive)}")
        print(f"  Files with hypertensive labels: {len(files_with_hypertensive)}")
        
        # Get unique files (some may have both)
        minority_files = list(set(files_with_positive + files_with_hypertensive))
        print(f"  Unique files with minority classes: {len(minority_files)}")
        
        return {
            'minority_files': minority_files,
            'positive_files': files_with_positive,
            'hypertensive_files': files_with_hypertensive
        }
    
    def augment_minority_files(self, labeled_csv_files, output_dir, 
                               num_versions=5, min_positive=1, min_hypertensive=1):
        """
        Augment all files containing minority classes.
        
        Args:
            labeled_csv_files: List of paths to labeled CSV files
            output_dir: Directory to save augmented files
            num_versions: Number of augmented versions per file
            min_positive: Minimum positive samples to augment file
            min_hypertensive: Minimum hypertensive samples to augment file
            
        Returns:
            Dictionary with augmentation results
        """
        print("="*70)
        print("SIMPLE AUGMENTATION PIPELINE")
        print("="*70)
        
        # Identify minority files
        file_info = self.identify_minority_files(
            labeled_csv_files, min_positive, min_hypertensive
        )
        
        minority_files = file_info['minority_files']
        
        if len(minority_files) == 0:
            print("\nNo files with minority classes found!")
            return {'success': False, 'augmented_files': []}
        
        print(f"\nWill augment {len(minority_files)} files x {num_versions} versions")
        print(f"Output directory: {output_dir}")
        
        # Augment each file
        all_augmented_files = []
        
        for csv_file in tqdm(minority_files, desc="Augmenting files"):
            try:
                created_files = self.augment_file(csv_file, output_dir, num_versions)
                all_augmented_files.extend(created_files)
            except Exception as e:
                print(f"\nError augmenting {csv_file}: {e}")
                continue
        
        print(f"\n{'='*70}")
        print("AUGMENTATION COMPLETE")
        print(f"{'='*70}")
        print(f"  Original files processed: {len(minority_files)}")
        print(f"  Augmented files created: {len(all_augmented_files)}")
        print(f"  Total files for spectrogram generation: {len(labeled_csv_files) + len(all_augmented_files)}")
        print(f"{'='*70}\n")
        
        return {
            'success': True,
            'augmented_files': all_augmented_files,
            'original_minority_files': minority_files,
            'file_info': file_info
        }


def load_all_spectrograms(spectrogram_dir):
    """
    Load and combine all .npz spectrogram files from a directory.
    
    Args:
        spectrogram_dir: Directory containing .npz files from PPGSpectrogramGenerator
        
    Returns:
        Combined dictionary with all spectrograms and labels
    """
    print("\n" + "="*70)
    print("LOADING SPECTROGRAMS")
    print("="*70)
    
    npz_files = glob.glob(os.path.join(spectrogram_dir, '*_spectrograms.npz'))
    
    if not npz_files:
        raise ValueError(f"No .npz files found in {spectrogram_dir}")
    
    print(f"\nFound {len(npz_files)} .npz files")
    
    all_spectrograms = []
    all_labels = []
    
    for npz_file in tqdm(npz_files, desc="Loading files"):
        try:
            data = np.load(npz_file, allow_pickle=True)
            all_spectrograms.append(data['spectrograms'])
            all_labels.append(data['labels'])
        except Exception as e:
            print(f"\nError loading {npz_file}: {e}")
            continue
    
    # Combine all
    combined_spectrograms = np.concatenate(all_spectrograms, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    
    print(f"\nTotal spectrograms loaded: {len(combined_labels)}")
    print(f"Spectrogram shape: {combined_spectrograms[0].shape}")
    
    # Show distribution
    unique_labels, counts = np.unique(combined_labels, return_counts=True)
    print("\nClass distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count} ({count/len(combined_labels)*100:.2f}%)")
    
    print("="*70 + "\n")
    
    return {
        'spectrograms': combined_spectrograms,
        'labels': combined_labels
    }


def balance_spectrograms(spectrogram_data, target_count=None, random_seed=42):
    """
    Balance spectrogram dataset by sampling to equal class counts.
    
    Args:
        spectrogram_data: Dictionary with 'spectrograms' and 'labels' keys
        target_count: Target count per class (None = use minimum class count)
        random_seed: Random seed for reproducibility
        
    Returns:
        Balanced spectrograms and labels
    """
    np.random.seed(random_seed)
    
    spectrograms = spectrogram_data['spectrograms']
    labels = spectrogram_data['labels']
    
    print("\n" + "="*70)
    print("BALANCING SPECTROGRAMS")
    print("="*70)
    
    # Get class counts
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nOriginal distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count}")
    
    # Determine target count
    if target_count is None:
        target_count = int(np.min(counts))
        print(f"\nUsing minimum class count as target: {target_count}")
    else:
        print(f"\nUsing specified target count: {target_count}")
    
    # Sample from each class
    balanced_specs = []
    balanced_labels = []
    
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        
        if len(label_indices) < target_count:
            print(f"\nWARNING: {label} has only {len(label_indices)} samples")
            print(f"         Need {target_count}. Using all samples + duplicates.")
            # Oversample with replacement
            sampled_indices = np.random.choice(label_indices, target_count, replace=True)
        else:
            # Undersample without replacement
            sampled_indices = np.random.choice(label_indices, target_count, replace=False)
        
        balanced_specs.append(spectrograms[sampled_indices])
        balanced_labels.append(labels[sampled_indices])
    
    # Concatenate
    balanced_specs = np.concatenate(balanced_specs, axis=0)
    balanced_labels = np.concatenate(balanced_labels, axis=0)
    
    # Shuffle
    shuffle_indices = np.random.permutation(len(balanced_labels))
    balanced_specs = balanced_specs[shuffle_indices]
    balanced_labels = balanced_labels[shuffle_indices]
    
    print("\nBalanced distribution:")
    unique_labels, counts = np.unique(balanced_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count}")
    
    print(f"\nTotal samples: {len(balanced_labels)}")
    print("="*70 + "\n")
    
    return {
        'spectrograms': balanced_specs,
        'labels': balanced_labels,
        'target_count': target_count
    }


def save_balanced_dataset(balanced_data, output_path='balanced_dataset.npz'):
    """
    Save balanced dataset to a single .npz file.
    
    Args:
        balanced_data: Dictionary from balance_spectrograms()
        output_path: Path to save file
    """
    np.savez_compressed(
        output_path,
        spectrograms=balanced_data['spectrograms'],
        labels=balanced_data['labels'],
        target_count=balanced_data['target_count']
    )
    
    file_size_mb = os.path.getsize(output_path) / (1024*1024)
    print(f"\nSaved balanced dataset to: {output_path}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Total samples: {len(balanced_data['labels'])}")
    
    return output_path


# Example usage
if __name__ == "__main__":
    # Step 1: Augment minority files
    augmenter = SimpleAugmenter(noise_level_range=(0.03, 0.08))
    
    # Get all labeled CSV files
    labeled_files = glob.glob('hypertensive_labeled_output_fixed/*_labeled.csv')
    
    if not labeled_files:
        print("No labeled CSV files found!")
        print("Please run HypertensiveLabeler first.")
    else:
        # Augment files with minority classes
        result = augmenter.augment_minority_files(
            labeled_csv_files=labeled_files,
            output_dir='augmented_labeled_files_fixed',
            num_versions=5,  # Create 5 noisy copies per file
            min_positive=1,
            min_hypertensive=1
        )
        
        if result['success']:
            print("\nNext steps:")
            print("1. Run PPGSpectrogramGenerator on:")
            print("   - Original files: hypertensive_labeled_output/*_labeled.csv")
            print("   - Augmented files: augmented_labeled_files/*_aug*.csv")
            print("2. Load all spectrograms")
            print("3. Use balance_spectrograms() to create balanced dataset")
            
            print("\n" + "="*70)
            print("COMPLETE PIPELINE EXAMPLE")
            print("="*70)
            print("""
# After augmentation, run this complete pipeline:

from ppg_spectrogram_generator import PPGSpectrogramGenerator
from simple_augmenter import load_all_spectrograms, balance_spectrograms, save_balanced_dataset

# Step 2: Generate spectrograms from all files
spec_gen = PPGSpectrogramGenerator(
    sampling_rate=125,
    window_seconds=10,
    nperseg=256,
    noverlap=128,
    nfft=2048,
    freq_range=(0.5, 4)
)

all_files = glob.glob('hypertensive_labeled_output/*_labeled.csv')
all_files += glob.glob('augmented_labeled_files/*_aug*.csv')

results = spec_gen.process_multiple_files(
    labeled_csv_files=all_files,
    output_dir='all_spectrograms',
    stride_samples=625,
    save_npz=True,
    visualize_samples=0
)

# Step 3: Load all spectrograms
all_data = load_all_spectrograms('all_spectrograms')

# Step 4: Balance dataset
balanced_data = balance_spectrograms(all_data, target_count=None)

# Step 5: Save balanced dataset
save_balanced_dataset(balanced_data, 'balanced_dataset.npz')

# Step 6: Load and use
data = np.load('balanced_dataset.npz')
X = data['spectrograms']
y = data['labels']
print(f"Ready for training: X shape = {X.shape}, y shape = {y.shape}")
            """)