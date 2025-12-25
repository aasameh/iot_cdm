import pandas as pd
import numpy as np
import os
from scipy.signal import spectrogram, detrend, butter, filtfilt
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class BalancedPPGSpectrogramGenerator:
    """
    Generate spectrograms from balanced CSV files (train/val/test splits)
    """
    
    def __init__(self, sampling_rate=125, window_seconds=10, 
                 nperseg=256, noverlap=128, nfft=2048, 
                 freq_range=(0.5, 4)):
        self.sampling_rate = sampling_rate
        self.window_seconds = window_seconds
        self.window_samples = window_seconds * sampling_rate
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.freq_range = freq_range
        
        print(f"PPG Spectrogram Generator Configuration:")
        print(f"  Sampling Rate: {sampling_rate} Hz")
        print(f"  Window Duration: {window_seconds} seconds ({self.window_samples} samples)")
        print(f"  FFT Segment: {nperseg} samples")
        print(f"  Overlap: {noverlap} samples")
        print(f"  Frequency Range: {freq_range[0]}-{freq_range[1]} Hz")
    
    def compute_log_spectrogram(self, signal_data):
        """Compute log-normalized spectrogram for PPG signal"""
        
        # Preprocessing
        signal_data = detrend(signal_data)
        
        # Bandpass filter
        nyquist = self.sampling_rate / 2
        low = max(0.04, self.freq_range[0]) / nyquist
        high = min(self.freq_range[1], nyquist - 0.1) / nyquist
        
        b, a = butter(4, [low, high], btype='band')
        signal_data = filtfilt(b, a, signal_data)
        
        # Normalize
        signal_data = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-10)
        
        # Compute spectrogram
        f, t, Sxx = spectrogram(signal_data, 
                                fs=self.sampling_rate,
                                nperseg=self.nperseg,
                                noverlap=self.noverlap,
                                nfft=self.nfft,
                                window='hann',
                                scaling='density')
        
        # Filter frequency range
        freq_mask = (f >= self.freq_range[0]) & (f <= self.freq_range[1])
        f = f[freq_mask]
        Sxx = Sxx[freq_mask, :]
        
        # Log scale
        epsilon = 1e-10
        Sxx_log = 10 * np.log10(Sxx + epsilon)
        
        # Normalize
        p5 = np.percentile(Sxx_log, 5)
        p95 = np.percentile(Sxx_log, 95)
        Sxx_normalized = np.clip((Sxx_log - p5) / (p95 - p5 + epsilon), 0, 1)
        
        # Gamma correction for contrast
        Sxx_enhanced = np.power(Sxx_normalized, 0.5)
        
        return Sxx_enhanced, f, t
    
    def process_csv_to_spectrograms(self, csv_path, output_path, 
                                     stride_samples=None, visualize=True):
        """
        Process a single CSV file (train/val/test) into spectrograms
        
        Args:
            csv_path: Path to balanced CSV file
            output_path: Where to save spectrograms.npz
            stride_samples: Stride for rolling windows (default: window_samples)
            visualize: Whether to create sample visualizations
        
        Returns:
            Dictionary with spectrograms and labels
        """
        if stride_samples is None:
            stride_samples = self.window_samples
        
        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(csv_path)}")
        print(f"{'='*70}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"Loaded: {len(df)} samples")
        print(f"Columns: {list(df.columns)}")
        
        if 'PPG' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must have 'PPG' and 'label' columns")
        
        # Label distribution
        print("\nLabel distribution:")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count:,} ({count/len(df)*100:.2f}%)")
        
        # Extract windows
        n_samples = len(df)
        n_windows = (n_samples - self.window_samples) // stride_samples + 1
        
        print(f"\nCreating spectrograms:")
        print(f"  Window size: {self.window_samples} samples")
        print(f"  Stride: {stride_samples} samples")
        print(f"  Number of windows: {n_windows}")
        
        spectrograms_list = []
        labels_list = []
        
        # Get spectrogram shape from first window
        temp_data = df['PPG'].values[:self.window_samples]
        temp_spec, freq_bins, time_bins = self.compute_log_spectrogram(temp_data)
        spec_shape = temp_spec.shape
        
        print(f"  Spectrogram shape: {spec_shape}")
        print(f"  Frequency bins: {len(freq_bins)}")
        print(f"  Time bins: {len(time_bins)}")
        
        # Process each window
        for i in tqdm(range(n_windows), desc="Generating spectrograms"):
            start_idx = i * stride_samples
            end_idx = start_idx + self.window_samples
            
            window_data = df.iloc[start_idx:end_idx]
            
            # Get label (majority vote in window)
            window_label = window_data['label'].value_counts().idxmax()
            
            # Extract PPG signal
            ppg_data = window_data['PPG'].values
            
            # Compute spectrogram
            spec, _, _ = self.compute_log_spectrogram(ppg_data)
            
            # Add channel dimension for CNN (H, W, C)
            spec = np.expand_dims(spec, axis=-1)
            
            spectrograms_list.append(spec)
            labels_list.append(window_label)
        
        # Convert to numpy arrays
        spectrograms = np.array(spectrograms_list)
        labels = np.array(labels_list)
        
        print(f"\nGenerated {len(spectrograms)} spectrograms")
        print(f"Shape: {spectrograms.shape}")
        
        print("\nSpectrogram label distribution:")
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"  {label}: {count} ({count/len(labels)*100:.2f}%)")
        
        # Save
        np.savez_compressed(
            output_path,
            spectrograms=spectrograms,
            labels=labels,
            frequency_bins=freq_bins,
            time_bins=time_bins,
            sampling_rate=self.sampling_rate,
            window_seconds=self.window_seconds
        )
        
        print(f"\nSaved: {output_path}")
        print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")