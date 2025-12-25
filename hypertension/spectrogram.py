import pandas as pd
import numpy as np
import os
import glob
from scipy import signal
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class PPGSpectrogramGenerator:
    def __init__(self, sampling_rate=125, window_seconds=10, 
                 nperseg=256, noverlap=128, nfft=None, 
                 freq_range=(0.04, 10)):
        """
        Initialize PPG-only spectrogram generator focused on low frequencies.
        
        Args:
            sampling_rate: Sampling rate in Hz (default: 125)
            window_seconds: Duration of each window in seconds (default: 10)
            nperseg: Length of each FFT segment (default: 256)
            noverlap: Number of points to overlap between segments (default: 128)
            nfft: Length of FFT (default: None = nperseg)
            freq_range: Tuple (min_freq, max_freq) - LOW FREQUENCY FOCUS (0.04-10 Hz)
        """
        self.sampling_rate = sampling_rate
        self.window_seconds = window_seconds
        self.window_samples = window_seconds * sampling_rate
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft if nfft is not None else nperseg
        self.freq_range = freq_range
        
        print(f"PPG-Only Spectrogram Generator Configuration:")
        print(f"  Signal: PPG ONLY (no ECG contamination)")
        print(f"  Sampling Rate: {sampling_rate} Hz")
        print(f"  Window Duration: {window_seconds} seconds ({self.window_samples} samples)")
        print(f"  FFT Segment Length: {nperseg} samples")
        print(f"  Overlap: {noverlap} samples ({noverlap/nperseg*100:.1f}%)")
        print(f"  Frequency Resolution: {sampling_rate/nperseg:.3f} Hz")
        print(f"  Frequency Range: {freq_range[0]}-{freq_range[1]} Hz (LOW FREQUENCY FOCUS)")
        print(f"  Max Frequency: {sampling_rate/2:.2f} Hz (Nyquist)")
    
    def compute_log_spectrogram(self, signal_data, apply_preprocessing=True):
        """
        Compute log-normalized spectrogram for PPG signal.
        
        Args:
            signal_data: 1D array of PPG values (length = window_samples)
            apply_preprocessing: Whether to apply signal preprocessing (default: True)
            
        Returns:
            Log-normalized spectrogram (2D array: frequency_bins × time_steps)
        """
        if apply_preprocessing:
            from scipy.signal import detrend, butter, filtfilt
            
            # Detrend to remove DC offset and slow drifts
            signal_data = detrend(signal_data)
            
            # Apply bandpass filter - PPG low frequency focus
            nyquist = self.sampling_rate / 2
            low = max(0.04, self.freq_range[0]) / nyquist
            high = min(self.freq_range[1], nyquist - 0.1) / nyquist
            
            b, a = butter(4, [low, high], btype='band') # type:ignore
            signal_data = filtfilt(b, a, signal_data)
            
            # Normalize to zero mean and unit variance
            signal_data = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-10)
        
        # Compute spectrogram using STFT
        f, t, Sxx = spectrogram(signal_data, 
                                fs=self.sampling_rate,
                                nperseg=self.nperseg,
                                noverlap=self.noverlap,
                                nfft=self.nfft,
                                window='hann',
                                scaling='density')
        
        # Filter to LOW frequency range
        freq_mask = (f >= self.freq_range[0]) & (f <= self.freq_range[1])
        f = f[freq_mask]
        Sxx = Sxx[freq_mask, :]
        
        # Convert to log scale (dB)
        epsilon = 1e-10
        Sxx_log = 10 * np.log10(Sxx + epsilon)
        
        # Normalize using robust scaling
        p5 = np.percentile(Sxx_log, 5)
        p95 = np.percentile(Sxx_log, 95)
        Sxx_normalized = np.clip((Sxx_log - p5) / (p95 - p5 + epsilon), 0, 1)
        
        return Sxx_normalized, f, t
    
    def enhance_contrast(self, spectrogram, gamma=0.5):
        """
        Apply gamma correction to enhance contrast.
        
        Args:
            spectrogram: 2D normalized spectrogram [0, 1]
            gamma: Gamma value (< 1 brightens, > 1 darkens). Default 0.5
            
        Returns:
            Contrast-enhanced spectrogram
        """
        return np.power(spectrogram, gamma)
    
    def create_rolling_spectrograms(self, labeled_df, stride_samples=None):
        """
        Create rolling window spectrograms from PPG data only.
        
        Args:
            labeled_df: DataFrame with PPG and label columns
            stride_samples: Number of samples to move forward each window 
                          (default: None = window_samples, no overlap)
        
        Returns:
            Dictionary with spectrograms, labels, and metadata
        """
        if stride_samples is None:
            stride_samples = self.window_samples
        
        if 'PPG' not in labeled_df.columns:
            raise ValueError("PPG column not found in DataFrame")
        
        n_samples = len(labeled_df)
        n_windows = (n_samples - self.window_samples) // stride_samples + 1
        
        print(f"\nCreating PPG-only spectrograms:")
        print(f"  Total samples: {n_samples}")
        print(f"  Window size: {self.window_samples} samples")
        print(f"  Stride: {stride_samples} samples")
        print(f"  Number of windows: {n_windows}")
        
        spectrograms_list = []
        labels_list = []
        window_indices = []
        
        # Get first window to determine shape
        temp_data = labeled_df['PPG'].values[:self.window_samples]
        temp_spec, freq_bins, time_bins = self.compute_log_spectrogram(temp_data)
        spec_shape = temp_spec.shape
        
        print(f"  Spectrogram shape: {spec_shape}")
        print(f"  Frequency bins: {len(freq_bins)} ({freq_bins[0]:.3f} to {freq_bins[-1]:.2f} Hz)")
        print(f"  Time bins per window: {len(time_bins)}")
        
        # Process each window
        for i in tqdm(range(n_windows), desc="Generating PPG spectrograms"):
            start_idx = i * stride_samples
            end_idx = start_idx + self.window_samples
            
            # Extract window data
            window_data = labeled_df.iloc[start_idx:end_idx]
            
            # Get label for this window
            window_label = window_data['label'].iloc[-1]
            
            # Extract PPG signal
            ppg_data = window_data['PPG'].values
            
            # Compute log-normalized spectrogram
            spec, _, _ = self.compute_log_spectrogram(ppg_data)
            
            # Apply contrast enhancement
            spec = self.enhance_contrast(spec)
            
            # Add channel dimension for compatibility with CNN (H, W, C)
            spec = np.expand_dims(spec, axis=-1)
            
            spectrograms_list.append(spec)
            labels_list.append(window_label)
            window_indices.append((start_idx, end_idx))
        
        # Convert to numpy arrays
        spectrograms = np.array(spectrograms_list)
        labels = np.array(labels_list)
        
        # Calculate label distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        print(f"\n  Generated {len(spectrograms)} spectrograms")
        print(f"  Label distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"    {label}: {count} ({count/len(labels)*100:.2f}%)")
        
        return {
            'spectrograms': spectrograms,
            'labels': labels,
            'window_indices': window_indices,
            'frequency_bins': freq_bins,
            'time_bins': time_bins,
            'shape': spectrograms.shape
        }
    
    def process_labeled_file(self, labeled_csv_path, output_dir=None, 
                           stride_samples=None, save_npz=True, visualize_samples=3):
        """
        Process a single labeled CSV file and generate PPG spectrograms.
        
        Args:
            labeled_csv_path: Path to labeled CSV file
            output_dir: Directory to save spectrograms (optional)
            stride_samples: Stride for rolling windows (None = no overlap)
            save_npz: Whether to save as compressed numpy file
            visualize_samples: Number of random samples to visualize (0 = none)
        
        Returns:
            Dictionary with spectrograms and metadata
        """
        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(labeled_csv_path)}")
        print(f"{'='*70}")
        
        # Load labeled data
        df = pd.read_csv(labeled_csv_path)
        
        print(f"Loaded data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Create spectrograms
        result = self.create_rolling_spectrograms(df, stride_samples)
        
        # Save if output directory specified
        if output_dir and save_npz:
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(labeled_csv_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_PPG_spectrograms.npz")
            
            np.savez_compressed(
                output_path,
                spectrograms=result['spectrograms'],
                labels=result['labels'],
                window_indices=result['window_indices'],
                frequency_bins=result['frequency_bins'],
                time_bins=result['time_bins'],
                sampling_rate=self.sampling_rate,
                window_seconds=self.window_seconds
            )
            
            print(f"\n  Saved to: {output_path}")
            print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
        # Visualize samples if requested
        if visualize_samples > 0 and output_dir:
            self.visualize_spectrograms(result, labeled_csv_path, output_dir, 
                                       num_samples=visualize_samples)
        
        return result
    
    def visualize_spectrograms(self, result, source_file, output_dir, num_samples=3):
        """
        Visualize random PPG spectrogram samples for each label class.
        """
        spectrograms = result['spectrograms']
        labels = result['labels']
        freq_bins = result['frequency_bins']
        
        unique_labels = np.unique(labels)
        
        print(f"\n  Creating visualizations...")
        
        for label_name in unique_labels:
            label_indices = np.where(labels == label_name)[0]
            
            if len(label_indices) == 0:
                continue
            
            # Randomly select samples
            sample_indices = np.random.choice(label_indices, 
                                             size=min(num_samples, len(label_indices)),
                                             replace=False)
            
            for idx in sample_indices:
                spec = spectrograms[idx, :, :, 0]  # Remove channel dimension
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                im = ax.imshow(spec, aspect='auto', origin='lower',
                              cmap='viridis', interpolation='nearest')
                ax.set_xlabel('Time Bins', fontsize=12)
                ax.set_ylabel('Frequency (Hz)', fontsize=12)
                ax.set_title(f'PPG Spectrogram - Label: {label_name}', fontsize=14, fontweight='bold')
                
                # Set y-axis to show actual frequencies
                y_ticks = np.linspace(0, len(freq_bins)-1, 8, dtype=int)
                y_labels = [f'{freq_bins[int(t)]:.2f}' for t in y_ticks]
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_labels)
                
                plt.colorbar(im, ax=ax, label='Normalized Power (dB)')
                plt.tight_layout()
                
                base_name = os.path.splitext(os.path.basename(source_file))[0]
                vis_path = os.path.join(output_dir, 
                                       f"{base_name}_PPG_sample_{idx}_{label_name}.png")
                plt.savefig(vis_path, dpi=150, bbox_inches='tight')
                plt.close()
        
        print(f"  Visualizations saved to: {output_dir}")
    
    def process_multiple_files(self, labeled_csv_files, output_dir=None,
                             stride_samples=None, save_npz=True, visualize_samples=3):
        """
        Process multiple labeled CSV files.
        """
        all_results = {}
        
        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING: {len(labeled_csv_files)} files (PPG ONLY)")
        print(f"{'='*70}")
        
        for csv_file in labeled_csv_files:
            try:
                result = self.process_labeled_file(
                    csv_file, output_dir, stride_samples,
                    save_npz, visualize_samples
                )
                all_results[csv_file] = result
            except Exception as e:
                print(f"\nError processing {csv_file}: {str(e)}")
                continue
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"Successfully processed: {len(all_results)}/{len(labeled_csv_files)} files")
        
        total_spectrograms = sum(r['spectrograms'].shape[0] for r in all_results.values())
        print(f"Total spectrograms generated: {total_spectrograms}")
        
        # Aggregate label distribution
        all_labels = np.concatenate([r['labels'] for r in all_results.values()])
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        print(f"\nOverall label distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"  {label}: {count} ({count/len(all_labels)*100:.2f}%)")
        
        return all_results


def analyze_ppg_separability(results, output_dir='ppg_separability_analysis'):
    """
    Analyze PPG-only class separability.
    This will show if removing ECG reduces feature contamination.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import euclidean
    from scipy.stats import ks_2samp
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("PPG-ONLY CLASS SEPARABILITY ANALYSIS")
    print(f"{'='*70}")
    
    # Aggregate all spectrograms and labels
    all_specs = []
    all_labels = []
    
    for result in results.values():
        all_specs.append(result['spectrograms'])
        all_labels.append(result['labels'])
    
    all_specs = np.concatenate(all_specs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"\nTotal samples: {len(all_specs)}")
    print(f"Shape per sample: {all_specs[0].shape}")
    
    # Get indices for each class
    positive_idx = np.where(all_labels == 'positive')[0]
    negative_idx = np.where(all_labels == 'negative')[0]
    hypertensive_idx = np.where(all_labels == 'hypertensive_event')[0]
    
    print(f"\nClass distribution:")
    print(f"  Positive: {len(positive_idx)}")
    print(f"  Negative: {len(negative_idx)}")
    print(f"  Hypertensive: {len(hypertensive_idx)}")
    
    if len(positive_idx) == 0 or len(negative_idx) == 0:
        print("\nWARNING: Not enough samples in both classes for analysis!")
        return
    
    # Flatten spectrograms
    specs_flat = all_specs.reshape(len(all_specs), -1)
    
    # 1. Statistical Distance Test
    print(f"\n{'='*70}")
    print("1. STATISTICAL DISTANCE TEST (KS Test)")
    print(f"{'='*70}")
    
    positive_specs = specs_flat[positive_idx]
    negative_specs = specs_flat[negative_idx]
    
    # Test random pixels
    n_pixels_test = min(500, specs_flat.shape[1])
    pixel_indices = np.random.choice(specs_flat.shape[1], n_pixels_test, replace=False)
    
    ks_statistics = []
    ks_pvalues = []
    
    for pixel_idx in pixel_indices[:100]:
        pos_values = positive_specs[:, pixel_idx]
        neg_values = negative_specs[:, pixel_idx]
        ks_stat, p_value = ks_2samp(pos_values, neg_values)
        ks_statistics.append(ks_stat)
        ks_pvalues.append(p_value)
    
    mean_ks = np.mean(ks_statistics)
    significant_pixels = np.sum(np.array(ks_pvalues) < 0.05)
    
    print(f"  Mean KS statistic: {mean_ks:.4f}")
    print(f"  (0 = identical, 1 = completely different)")
    print(f"  Significantly different pixels (p<0.05): {significant_pixels}/100")
    
    if mean_ks > 0.2 and significant_pixels > 50:
        print("  ✓ GOOD: Classes show statistical differences")
    elif mean_ks > 0.1:
        print("  ⚠ MODERATE: Some differences detected")
    else:
        print("  ✗ BAD: Classes are statistically very similar")
    
    # 2. Mean Spectrogram Comparison
    print(f"\n{'='*70}")
    print("2. MEAN PPG SPECTROGRAM COMPARISON")
    print(f"{'='*70}")
    
    # Sample for computing means
    pos_sample_idx = np.random.choice(positive_idx, min(500, len(positive_idx)), replace=False)
    neg_sample_idx = np.random.choice(negative_idx, min(500, len(negative_idx)), replace=False)
    
    mean_positive = all_specs[pos_sample_idx].mean(axis=0)[:, :, 0]
    mean_negative = all_specs[neg_sample_idx].mean(axis=0)[:, :, 0]
    difference = mean_positive - mean_negative
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes = np.array(axes)
    # Mean positive
    im1 = axes[0].imshow(mean_positive, aspect='auto', origin='lower',
                          cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title('Mean PPG - POSITIVE', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Frequency Bin', fontsize=12)
    axes[0].set_xlabel('Time Bin', fontsize=12)
    plt.colorbar(im1, ax=axes[0])
    
    # Mean negative
    im2 = axes[1].imshow(mean_negative, aspect='auto', origin='lower',
                          cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('Mean PPG - NEGATIVE', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Frequency Bin', fontsize=12)
    axes[1].set_xlabel('Time Bin', fontsize=12)
    plt.colorbar(im2, ax=axes[1])
    
    # Difference
    vmax_diff = np.abs(difference).max()
    im3 = axes[2].imshow(difference, aspect='auto', origin='lower',
                          cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
    axes[2].set_title('DIFFERENCE (Pos - Neg)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Frequency Bin', fontsize=12)
    axes[2].set_xlabel('Time Bin', fontsize=12)
    plt.colorbar(im3, ax=axes[2], label='Difference')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ppg_mean_comparison.png'), dpi=150)
    print(f"  Saved: ppg_mean_comparison.png")
    
    diff_magnitude = np.abs(difference).mean()
    print(f"  Mean absolute difference: {diff_magnitude:.4f}")
    
    if diff_magnitude > 0.05:
        print("  ✓ GOOD: Clear visual differences")
    elif diff_magnitude > 0.02:
        print("  ⚠ MODERATE: Subtle differences")
    else:
        print("  ✗ BAD: Nearly identical")
    
    # 3. PCA Visualization
    print(f"\n{'='*70}")
    print("3. PCA VISUALIZATION")
    print(f"{'='*70}")
    
    max_samples = 2000
    if len(specs_flat) > max_samples:
        sample_idx = np.random.choice(len(specs_flat), max_samples, replace=False)
        specs_pca = specs_flat[sample_idx]
        labels_pca = all_labels[sample_idx]
    else:
        specs_pca = specs_flat
        labels_pca = all_labels
    
    print(f"  Computing PCA on {len(specs_pca)} samples...")
    pca = PCA(n_components=2)
    specs_2d = pca.fit_transform(specs_pca)
    
    print(f"  Variance explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    
    plt.figure(figsize=(10, 8))
    for label, color, marker in [('positive', 'red', 'o'), ('negative', 'blue', 'x')]:
        mask = labels_pca == label
        if np.sum(mask) > 0:
            plt.scatter(specs_2d[mask, 0], specs_2d[mask, 1], 
                       c=color, label=label, alpha=0.5, s=20, marker=marker)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('PPG-Only PCA: Class Separability', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ppg_pca_visualization.png'), dpi=150)
    print(f"  Saved: ppg_pca_visualization.png")
    
    # Calculate separability metrics
    pos_mask_2d = labels_pca == 'positive'
    neg_mask_2d = labels_pca == 'negative'
    
    separability_ratio = 0
    if np.sum(pos_mask_2d) > 1 and np.sum(neg_mask_2d) > 1:
        pos_centroid = specs_2d[pos_mask_2d].mean(axis=0)
        neg_centroid = specs_2d[neg_mask_2d].mean(axis=0)
        
        inter_class_dist = euclidean(pos_centroid, neg_centroid)
        
        pos_points = specs_2d[pos_mask_2d]
        neg_points = specs_2d[neg_mask_2d]
        
        n_samples_dist = min(100, len(pos_points), len(neg_points))
        intra_pos = np.mean([euclidean(pos_points[i], pos_centroid) 
                            for i in range(n_samples_dist)])
        intra_neg = np.mean([euclidean(neg_points[i], neg_centroid) 
                            for i in range(n_samples_dist)])
        
        avg_intra = (intra_pos + intra_neg) / 2
        separability_ratio = inter_class_dist / avg_intra
        
        print(f"  Inter-class distance: {inter_class_dist:.4f}")
        print(f"  Avg intra-class distance: {avg_intra:.4f}")
        print(f"  Separability ratio: {separability_ratio:.4f}")
        
        if separability_ratio > 2.0:
            print("  ✓ EXCELLENT: Well-separated")
        elif separability_ratio > 1.0:
            print("  ✓ GOOD: Separable")
        elif separability_ratio > 0.5:
            print("  ⚠ MODERATE: Significant overlap")
        else:
            print("  ✗ BAD: Highly overlapped")
    
    # 4. Final Verdict
    print(f"\n{'='*70}")
    print("FINAL VERDICT: PPG-ONLY ANALYSIS")
    print(f"{'='*70}")
    
    score = 0
    if mean_ks > 0.15 and significant_pixels > 50:
        score += 1
    if diff_magnitude > 0.03:
        score += 1
    if separability_ratio > 1.0:
        score += 1
    
    print(f"\nSeparability Score: {score}/3")
    
    if score >= 2:
        print("✓ VERDICT: PPG-only shows GOOD separability!")
        print("  → No ECG contamination detected")
        print("  → Low-frequency PPG features are discriminative")
        print("  → Proceed with CNN training on PPG spectrograms")
    elif score == 1:
        print("⚠ VERDICT: PPG-only shows WEAK separability")
        print("  → Low-frequency features may be insufficient")
        print("  → Consider: time-domain features, HRV metrics, or hybrid approach")
    else:
        print("✗ VERDICT: PPG-only shows POOR separability")
        print("  → Spectrograms alone may not capture hypertensive patterns")
        print("  → Recommendations:")
        print("     - Extract pulse wave features (PTT, PWV, pulse morphology)")
        print("     - Try time-domain deep learning (LSTM, 1D-CNN)")
        print("     - Consider engineered features over raw spectrograms")
    
    print(f"{'='*70}\n")
    
    return {
        'ks_statistic': mean_ks,
        'diff_magnitude': diff_magnitude,
        'separability_ratio': separability_ratio,
        'score': score
    }


# Example usage
if __name__ == "__main__":
    # Initialize PPG-only generator with low-frequency focus
    spec_gen = PPGSpectrogramGenerator(
        sampling_rate=125,
        window_seconds=10,        
        nperseg=256,              # 2.048 seconds (fits ~4-5 segments in 10s)
        noverlap=128,             # 50% overlap
        nfft=2048,                # High zero-padding for frequency resolution
        freq_range=(0.5, 4)       # CHANGED: Include pulse frequencies!
    )
    # Get all labeled CSV files
    labeled_files = glob.glob('hypertensive_labeled_output_fixed/*_labeled.csv')
    
    if not labeled_files:
        print("No labeled CSV files found in 'hypertensive_labeled_output_fixed/'")
        print("Please run the HypertensiveLabeler first.")
    else:
        # Process all files - PPG ONLY
        results = spec_gen.process_multiple_files(
            labeled_csv_files=labeled_files,
            output_dir='ppg_spectrogram_output',
            stride_samples=625,  # 50% overlap
            save_npz=True,
            visualize_samples=3
        )
        
        # Analyze PPG-only separability
        if results:
            separability_metrics = analyze_ppg_separability(
                results, 
                output_dir='ppg_separability_analysis'
            )
            
            print("\n" + "="*70)
            print("COMPARISON RECOMMENDATION")
            print("="*70)
            print("Compare these results with your previous ECG+PPG analysis:")
            print("  - Check if KS statistic improved")
            print("  - Compare mean difference magnitude")
            print("  - Compare PCA separability ratio")
            print("\nIf PPG-only performs BETTER, it confirms ECG was causing")
            print("feature contamination. Use PPG spectrograms for your CNN!")
            print("="*70)