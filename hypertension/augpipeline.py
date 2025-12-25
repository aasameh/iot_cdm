from spectrogram import PPGSpectrogramGenerator
from augmenter import load_all_spectrograms, balance_spectrograms, save_balanced_dataset
import glob

# Step 2: Generate spectrograms from all files
spec_gen = PPGSpectrogramGenerator(
    sampling_rate=125,
    window_seconds=10,
    nperseg=256,
    noverlap=128,
    nfft=2048,
    freq_range=(0.5, 4)
)

all_files = glob.glob('hypertensive_labeled_output_fixed/*_labeled.csv')
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