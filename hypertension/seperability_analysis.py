import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
from scipy.stats import ks_2samp
from collections import Counter


def analyze_spectrogram_separability(spectrogram_dir='all_spectrograms',
                                     output_dir='separability_analysis'):
    """
    Analyze class separability in PPG spectrograms
    
    Args:
        spectrogram_dir: Directory with train/val/test spectrograms
        output_dir: Where to save analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("PPG SPECTROGRAM SEPARABILITY ANALYSIS")
    print("="*70)
    
    # Load spectrograms
    print(f"\nLoading spectrograms from: {spectrogram_dir}")
    
    all_specs = []
    all_labels = []
    
    spectrogram_dir = os.path.abspath(spectrogram_dir)
    
    for file_name in os.listdir(spectrogram_dir):
        if file_name.endswith(".npz"):
            file_path = os.path.join(spectrogram_dir, file_name)

            data = np.load(file_path, allow_pickle=True)

            if 'spectrograms' not in data or 'labels' not in data:
                print(f"Skipping {file_name}: missing required keys")
                continue

            specs = data['spectrograms']
            labels = data['labels']

            all_specs.append(specs)
            all_labels.append(labels)

        
    if not all_specs:
        raise ValueError(f"No spectrogram files found in {spectrogram_dir}")
    
    # Combine all data
    all_specs = np.concatenate(all_specs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"\nTotal samples: {len(all_specs)}")
    print(f"Spectrogram shape per sample: {all_specs[0].shape}")
    
    # Label distribution
    print("\nClass distribution:")
    label_counts = Counter(all_labels)
    for label, count in label_counts.items():
        print(f"  {label}: {count:,} ({count/len(all_labels)*100:.1f}%)")
    
    # Get frequency bins if available
    freq_bins = None
    if os.path.exists(os.path.join(spectrogram_dir, '_spectrograms.npz')):
        data = np.load(os.path.join(spectrogram_dir, '_spectrograms.npz'), 
                       allow_pickle=True)
        if 'frequency_bins' in data:
            freq_bins = data['frequency_bins']
    
    # ================================================================
    # 1. Statistical Distance Test (KS Test)
    # ================================================================
    print("\n" + "="*70)
    print("1. STATISTICAL DISTANCE TEST (Kolmogorov-Smirnov)")
    print("="*70)
    
    # Get class indices
    positive_idx = np.where(all_labels == 'positive')[0]
    negative_idx = np.where(all_labels == 'negative')[0]
    hypertensive_idx = np.where(all_labels == 'hypertensive_event')[0]
    
    if len(positive_idx) == 0 or len(negative_idx) == 0:
        print("\nWARNING: Missing classes - cannot perform separability analysis!")
        return
    
    # Flatten spectrograms
    specs_flat = all_specs.reshape(len(all_specs), -1)
    
    positive_specs = specs_flat[positive_idx]
    negative_specs = specs_flat[negative_idx]
    
    # Test random pixels
    n_pixels_test = min(500, specs_flat.shape[1])
    pixel_indices = np.random.choice(specs_flat.shape[1], n_pixels_test, replace=False)
    
    ks_statistics = []
    ks_pvalues = []
    
    print(f"Testing {min(100, n_pixels_test)} random spectrogram pixels...")
    
    for pixel_idx in pixel_indices[:100]:
        pos_values = positive_specs[:, pixel_idx]
        neg_values = negative_specs[:, pixel_idx]
        ks_stat, p_value = ks_2samp(pos_values, neg_values)
        ks_statistics.append(ks_stat)
        ks_pvalues.append(p_value)
    
    mean_ks = np.mean(ks_statistics)
    significant_pixels = np.sum(np.array(ks_pvalues) < 0.05)
    
    print(f"\n  Mean KS statistic: {mean_ks:.4f}")
    print(f"  (0 = identical distributions, 1 = completely different)")
    print(f"  Significantly different pixels (p<0.05): {significant_pixels}/100")
    
    if mean_ks > 0.2 and significant_pixels > 50:
        print("  ✓ GOOD: Classes show strong statistical differences")
        ks_score = 1
    elif mean_ks > 0.1:
        print("  ⚠ MODERATE: Some statistical differences detected")
        ks_score = 0.5
    else:
        print("  ✗ BAD: Classes are statistically very similar")
        ks_score = 0
    
    # ================================================================
    # 2. Mean Spectrogram Comparison
    # ================================================================
    print("\n" + "="*70)
    print("2. MEAN SPECTROGRAM COMPARISON")
    print("="*70)
    
    # Sample for computing means
    n_samples_mean = min(500, len(positive_idx), len(negative_idx))
    pos_sample_idx = np.random.choice(positive_idx, n_samples_mean, replace=False)
    neg_sample_idx = np.random.choice(negative_idx, n_samples_mean, replace=False)
    
    print(f"Computing mean spectrograms from {n_samples_mean} samples per class...")
    
    mean_positive = all_specs[pos_sample_idx].mean(axis=0)[:, :, 0]
    mean_negative = all_specs[neg_sample_idx].mean(axis=0)[:, :, 0]
    difference = mean_positive - mean_negative
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes = np.array(axes)
    # Mean positive
    im1 = axes[0].imshow(mean_positive, aspect='auto', origin='lower',
                          cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title('Mean Spectrogram - POSITIVE', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Frequency Bin', fontsize=12)
    axes[0].set_xlabel('Time Bin', fontsize=12)
    plt.colorbar(im1, ax=axes[0], label='Normalized Power')
    
    # Mean negative
    im2 = axes[1].imshow(mean_negative, aspect='auto', origin='lower',
                          cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('Mean Spectrogram - NEGATIVE', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Frequency Bin', fontsize=12)
    axes[1].set_xlabel('Time Bin', fontsize=12)
    plt.colorbar(im2, ax=axes[1], label='Normalized Power')
    
    # Difference
    vmax_diff = np.abs(difference).max()
    im3 = axes[2].imshow(difference, aspect='auto', origin='lower',
                          cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
    axes[2].set_title('DIFFERENCE (Pos - Neg)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Frequency Bin', fontsize=12)
    axes[2].set_xlabel('Time Bin', fontsize=12)
    plt.colorbar(im3, ax=axes[2], label='Power Difference')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_spectrogram_comparison.png'), dpi=150)
    print(f"\n  Saved: mean_spectrogram_comparison.png")
    plt.close()
    
    diff_magnitude = np.abs(difference).mean()
    print(f"  Mean absolute difference: {diff_magnitude:.4f}")
    
    if diff_magnitude > 0.05:
        print("  ✓ GOOD: Clear visual differences between classes")
        diff_score = 1
    elif diff_magnitude > 0.02:
        print("  ⚠ MODERATE: Subtle differences present")
        diff_score = 0.5
    else:
        print("  ✗ BAD: Nearly identical spectrograms")
        diff_score = 0
    
    # ================================================================
    # 3. Frequency Band Analysis
    # ================================================================
    print("\n" + "="*70)
    print("3. FREQUENCY BAND POWER ANALYSIS")
    print("="*70)
    
    # Compute mean power per frequency band
    mean_power_pos = mean_positive.mean(axis=1)  # Average over time
    mean_power_neg = mean_negative.mean(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    freq_axis = np.arange(len(mean_power_pos))
    ax.plot(freq_axis, mean_power_pos, 'r-', linewidth=2, label='Positive', alpha=0.7)
    ax.plot(freq_axis, mean_power_neg, 'b-', linewidth=2, label='Negative', alpha=0.7)
    ax.fill_between(freq_axis, mean_power_pos, mean_power_neg, alpha=0.3, color='gray')
    
    ax.set_xlabel('Frequency Bin', fontsize=12)
    ax.set_ylabel('Mean Power', fontsize=12)
    ax.set_title('Frequency Band Power Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add frequency labels if available
    if freq_bins is not None:
        n_ticks = 6
        tick_positions = np.linspace(0, len(freq_bins)-1, n_ticks, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f'{freq_bins[i]:.2f} Hz' for i in tick_positions])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'frequency_band_comparison.png'), dpi=150)
    print(f"\n  Saved: frequency_band_comparison.png")
    plt.close()
    
    # Find most discriminative frequency bands
    freq_diff = np.abs(mean_power_pos - mean_power_neg)
    top_bands = np.argsort(freq_diff)[-5:][::-1]
    
    print("\n  Top 5 most discriminative frequency bands:")
    for i, band_idx in enumerate(top_bands, 1):
        if freq_bins is not None:
            freq_label = f"{freq_bins[band_idx]:.2f} Hz"
        else:
            freq_label = f"Band {band_idx}"
        print(f"    {i}. {freq_label}: diff = {freq_diff[band_idx]:.4f}")
    
    # ================================================================
    # 4. PCA Visualization
    # ================================================================
    print("\n" + "="*70)
    print("4. PCA VISUALIZATION (2D Projection)")
    print("="*70)
    
    # Limit samples for PCA
    max_samples_pca = 2000
    if len(specs_flat) > max_samples_pca:
        sample_idx = np.random.choice(len(specs_flat), max_samples_pca, replace=False)
        specs_pca = specs_flat[sample_idx]
        labels_pca = all_labels[sample_idx]
    else:
        specs_pca = specs_flat
        labels_pca = all_labels
    
    print(f"  Computing PCA on {len(specs_pca)} samples...")
    pca = PCA(n_components=2)
    specs_2d = pca.fit_transform(specs_pca)
    
    variance_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  Variance explained by 2 components: {variance_explained:.2f}%")
    
    # Plot PCA
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {'positive': 'red', 'negative': 'blue', 'hypertensive_event': 'orange'}
    markers = {'positive': 'o', 'negative': 'x', 'hypertensive_event': '^'}
    
    for label in np.unique(labels_pca):
        mask = labels_pca == label
        if np.sum(mask) > 0:
            ax.scatter(specs_2d[mask, 0], specs_2d[mask, 1], 
                      c=colors.get(label, 'gray'), 
                      label=label, 
                      alpha=0.5, 
                      s=30, 
                      marker=markers.get(label, 'o'))
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title('PCA: Class Separability in 2D Space', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_visualization.png'), dpi=150)
    print(f"\n  Saved: pca_visualization.png")
    plt.close()
    
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
        
        print(f"\n  Inter-class distance: {inter_class_dist:.4f}")
        print(f"  Avg intra-class distance: {avg_intra:.4f}")
        print(f"  Separability ratio: {separability_ratio:.4f}")
        print(f"    (>2.0 = excellent, >1.0 = good, >0.5 = moderate, <0.5 = poor)")
        
        if separability_ratio > 2.0:
            print("  ✓ EXCELLENT: Classes are well-separated")
            pca_score = 1
        elif separability_ratio > 1.0:
            print("  ✓ GOOD: Classes are separable")
            pca_score = 1
        elif separability_ratio > 0.5:
            print("  ⚠ MODERATE: Significant class overlap")
            pca_score = 0.5
        else:
            print("  ✗ BAD: Classes are highly overlapped")
            pca_score = 0
    
    # ================================================================
    # 5. Sample Spectrograms Visualization
    # ================================================================
    print("\n" + "="*70)
    print("5. SAMPLE SPECTROGRAMS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten() # type: ignore
    
    for i, label in enumerate(['positive', 'negative']):
        label_idx = np.where(all_labels == label)[0]
        samples = np.random.choice(label_idx, min(3, len(label_idx)), replace=False)
        
        for j, sample_idx in enumerate(samples):
            ax = axes[i*3 + j]
            spec = all_specs[sample_idx, :, :, 0]
            
            im = ax.imshow(spec, aspect='auto', origin='lower',
                          cmap='viridis', interpolation='nearest')
            ax.set_title(f'{label.upper()} - Sample {j+1}', fontweight='bold')
            ax.set_xlabel('Time Bin')
            ax.set_ylabel('Frequency Bin')
            plt.colorbar(im, ax=ax, label='Power')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_spectrograms.png'), dpi=150)
    print(f"\n  Saved: sample_spectrograms.png")
    plt.close()
    
    # ================================================================
    # 6. Final Verdict
    # ================================================================
    print("\n" + "="*70)
    print("FINAL SEPARABILITY VERDICT")
    print("="*70)
    
    total_score = ks_score + diff_score + pca_score # type: ignore
    max_score = 3
    
    print(f"\nSeparability Score: {total_score}/{max_score}")
    print(f"  Statistical Test (KS): {ks_score}")
    print(f"  Visual Difference: {diff_score}")
    print(f"  PCA Separability: {pca_score}") # type: ignore
    
    if total_score >= 2.5:
        verdict = "EXCELLENT"
        recommendation = "✓ Spectrograms show strong discriminative features!\n" \
                        "  → Proceed with CNN training - high confidence in success\n" \
                        "  → Classes are clearly separable in feature space"
    elif total_score >= 1.5:
        verdict = "GOOD"
        recommendation = "✓ Spectrograms show moderate discriminative features\n" \
                        "  → Proceed with CNN training with caution\n" \
                        "  → Consider adding regularization and data augmentation\n" \
                        "  → May benefit from feature engineering or ensemble methods"
    elif total_score >= 0.5:
        verdict = "MODERATE"
        recommendation = "⚠ Spectrograms show weak discriminative features\n" \
                        "  → CNN may struggle with current features\n" \
                        "  → Consider:\n" \
                        "     * Time-domain features (pulse morphology, HRV)\n" \
                        "     * Different frequency ranges or resolutions\n" \
                        "     * Hybrid approach (spectrograms + engineered features)\n" \
                        "     * Try LSTM or 1D-CNN on raw signals"
    else:
        verdict = "POOR"
        recommendation = "✗ Spectrograms show minimal discriminative features\n" \
                        "  → Spectrogram-based CNN NOT recommended\n" \
                        "  → Strongly consider alternative approaches:\n" \
                        "     * Time-domain deep learning (LSTM, 1D-CNN)\n" \
                        "     * Classical ML with engineered features\n" \
                        "     * Pulse wave analysis (PTT, PWV, morphology)\n" \
                        "     * Multi-modal fusion (ECG + PPG + ABP)"
    
    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")
    print(recommendation)
    print(f"{'='*70}")
    
    # Save summary
    summary_path = os.path.join(output_dir, 'separability_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PPG SPECTROGRAM SEPARABILITY ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total samples analyzed: {len(all_specs)}\n")
        f.write(f"Spectrogram shape: {all_specs[0].shape}\n\n")
        f.write("Class distribution:\n")
        for label, count in label_counts.items():
            f.write(f"  {label}: {count:,} ({count/len(all_labels)*100:.1f}%)\n")
        f.write(f"\n{'='*70}\n")
        f.write("METRICS:\n")
        f.write(f"{'='*70}\n")
        f.write(f"KS Statistic: {mean_ks:.4f}\n")
        f.write(f"Mean Absolute Difference: {diff_magnitude:.4f}\n")
        f.write(f"PCA Separability Ratio: {separability_ratio:.4f}\n")
        f.write(f"PCA Variance Explained: {variance_explained:.2f}%\n")
        f.write(f"\nSeparability Score: {total_score}/{max_score}\n")
        f.write(f"\nVERDICT: {verdict}\n\n")
        f.write(recommendation)
    
    print(f"\nSummary saved: {summary_path}")
    
    return {
        'ks_statistic': mean_ks,
        'diff_magnitude': diff_magnitude,
        'separability_ratio': separability_ratio,
        'variance_explained': variance_explained,
        'score': total_score,
        'verdict': verdict
    }


# Standalone usage
if __name__ == "__main__":
    # Analyze spectrograms from balanced dataset
    results = analyze_spectrogram_separability(
        spectrogram_dir='all_spectrograms',
        output_dir='separability_analysis',
    )
    
    print("\n✓ Analysis complete! Check the 'separability_analysis/' directory for results.")