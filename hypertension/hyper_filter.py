import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool, cpu_count
from functools import partial


class HypertensiveLabeler:
    def __init__(self, sampling_rate=125, threshold=110, event_duration=60, 
                 lookback_minutes=5, map_window_seconds=10):
        """
        Initialize the labeler with parameters.
        
        Args:
            sampling_rate: Sampling rate in Hz (default: 125)
            threshold: MAP threshold for hypertension in mmHg (default: 110)
            event_duration: Minimum duration for event in seconds (default: 30)
            lookback_minutes: Minutes to look back for positive labels (default: 2)
            map_window_seconds: Window size for MAP calculation in seconds (default: 10)
        """
        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self.event_duration = event_duration
        self.lookback_minutes = lookback_minutes
        self.map_window_seconds = map_window_seconds
        
        # Convert to samples
        self.event_samples = event_duration * sampling_rate
        self.lookback_samples = lookback_minutes * 60 * sampling_rate
        self.map_window_samples = map_window_seconds * sampling_rate
    
    def calculate_map(self, abp_data):
        """
        Calculate Mean Arterial Pressure using a rolling window average.
        
        Args:
            abp_data: Array of instantaneous ABP values
            
        Returns:
            Array of MAP values (same length as input)
        """
        # Use pandas rolling mean for efficient calculation
        abp_series = pd.Series(abp_data)
        
        # Calculate rolling mean with center alignment
        # min_periods=1 ensures we get values even at the edges
        map_values = abp_series.rolling(
            window=self.map_window_samples,
            center=True,
            min_periods=1
        ).mean().values
        
        return map_values
    
    def identify_hypertensive_events(self, abp_data):
        """
        Identify hypertensive events based on MAP.
        
        Returns:
            List of tuples (start_idx, end_idx) for each event
        """
        # Calculate MAP first
        map_data = self.calculate_map(abp_data)
        
        events = []
        consecutive_hyper = 0
        event_start = None
        in_event = False
        
        for i in range(len(map_data)):
            if map_data[i] > self.threshold:
                consecutive_hyper += 1
                
                # Event begins when we reach the minimum duration
                if not in_event and consecutive_hyper >= self.event_samples:
                    in_event = True
                    event_start = i - self.event_samples + 1
            else:
                if in_event:
                    # Event ends
                    events.append((event_start, i - 1))
                    in_event = False
                consecutive_hyper = 0
        
        # Handle event extending to end of data
        if in_event:
            events.append((event_start, len(map_data) - 1))
        
        return events, map_data
    
    def create_labels(self, abp_data):
        """
        Create labels for all data points based on MAP.
        
        Returns:
            numpy array of labels: 'hypertensive_event', 'positive', or 'negative'
            numpy array of MAP values
        """
        n_samples = len(abp_data)
        labels = np.full(n_samples, 'unlabeled', dtype=object)
        
        # Identify events (now returns MAP data too)
        events, map_data = self.identify_hypertensive_events(abp_data)
        
        # Label hypertensive events
        for start, end in events:
            labels[start:end+1] = 'hypertensive_event'
            
            # Label positive samples (up to lookback_minutes before event)
            positive_start = max(0, start - self.lookback_samples)
            labels[positive_start:start] = 'positive'
        
        # Label all remaining unlabeled samples as negative
        labels[labels == 'unlabeled'] = 'negative'
        
        return labels, len(events), map_data
    
    def process_single_file(self, csv_file, output_file=None):
        """
        Process a single CSV file with 3 rows (ECG, ABP, PPG).
        
        Args:
            csv_file: Path to CSV file with 3 rows
            output_file: Path to save labeled output (optional)
        
        Returns:
            Dictionary with p
        """
        try:
            print(f"Processing: {csv_file}")
            
            # Load data - each row is a signal
            data = pd.read_csv(csv_file, header=None)
            
            if data.shape[0] != 3:
                raise ValueError(f"Expected 3 rows, got {data.shape[0]} rows")
            
            # Extract signals
            ecg = data.iloc[0].values
            abp = data.iloc[1].values
            ppg = data.iloc[2].values
            
            print(f"  Data length: {len(abp)} samples ({len(abp)/self.sampling_rate/60:.2f} minutes)")
            print(f"  MAP window: {self.map_window_seconds} seconds ({self.map_window_samples} samples)")
            
            # Create labels based on MAP calculated from ABP
            labels, num_events, map_data = self.create_labels(abp)
            
            # Create output DataFrame
            df = pd.DataFrame({
                'ECG': ecg,
                'ABP': abp,
                'MAP': map_data,
                'PPG': ppg,
                'label': labels
            })
            
            # Print label distribution
            label_counts = pd.Series(labels).value_counts()
            print(f"  Found {num_events} hypertensive events")
            print("  Label distribution:")
            for label, count in label_counts.items():
                print(f"    {label}: {count} samples ({count/len(labels)*100:.2f}%)")
            
            # Print MAP statistics
            print(f"  MAP statistics:")
            print(f"    Mean: {np.mean(map_data):.2f} mmHg")
            print(f"    Min: {np.min(map_data):.2f} mmHg")
            print(f"    Max: {np.max(map_data):.2f} mmHg")
            
            # Save if output file specified
            if output_file:
                df.to_csv(output_file, index=False)
                print(f"  Saved to: {output_file}")
            
            return {
                'file': csv_file,
                'dataframe': df,
                'num_events': num_events,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            return {
                'file': csv_file,
                'dataframe': None,
                'num_events': 0,
                'success': False,
                'error': str(e)
            }
    
    def process_multiple_files(self, csv_files, output_dir=None, num_workers=None):
        """
        Process multiple CSV files sequentially or in parallel.
        
        Args:
            csv_files: List of paths to CSV files
            output_dir: Directory to save labeled outputs (optional)
            num_workers: Number of parallel workers (default: None = sequential, 0 = auto CPU count)
        
        Returns:
            Dictionary mapping filename to labeled DataFrame
        """
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        results = {}
        total_events = 0
        successful_files = 0
        failed_files = 0
        files_with_events = []  # Track which files have events
        
        print(f"Processing {len(csv_files)} files...")
        print("="*60)
        
        # Prepare output file paths
        file_output_pairs = []
        for csv_file in csv_files:
            output_file = None
            if output_dir:
                base_name = os.path.basename(csv_file)
                name_without_ext = os.path.splitext(base_name)[0]
                output_file = os.path.join(output_dir, f"{name_without_ext}_labeled.csv")
            file_output_pairs.append((csv_file, output_file))
        
        # Process files
        if num_workers is None or num_workers == 1:
            # Sequential processing
            print("Processing sequentially...")
            for csv_file, output_file in file_output_pairs:
                result = self.process_single_file(csv_file, output_file)
                if result['success']:
                    results[result['file']] = result['dataframe']
                    successful_files += 1
                    if result['num_events'] > 0:
                        total_events += result['num_events']
                        files_with_events.append(result['file'])
                else:
                    failed_files += 1
                print()
        else:
            # Parallel processing with multiprocessing
            if num_workers == 0:
                num_workers = cpu_count()
            
            print(f"Processing with {num_workers} workers...")
            
            # Create a worker function with self bound
            worker_func = partial(_process_file_worker, 
                                 sampling_rate=self.sampling_rate,
                                 threshold=self.threshold,
                                 event_duration=self.event_duration,
                                 lookback_minutes=self.lookback_minutes,
                                 map_window_seconds=self.map_window_seconds)
            
            with Pool(processes=num_workers) as pool:
                pool_results = pool.map(worker_func, file_output_pairs)
            
            # Collect results
            for result in pool_results:
                if result['success']:
                    results[result['file']] = result['dataframe']
                    successful_files += 1
                    if result['num_events'] > 0:
                        total_events += result['num_events']
                        files_with_events.append(result['file'])
                else:
                    failed_files += 1
        
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Total files: {len(csv_files)}")
        print(f"  Successfully processed: {successful_files}")
        print(f"  Failed: {failed_files}")
        print(f"  Files with hypertensive events: {len(files_with_events)}")
        print(f"  Total hypertensive events found: {total_events}")
        
        if files_with_events:
            print(f"\n  Files containing hypertensive events:")
            for file_path in files_with_events:
                print(f"    - {os.path.basename(file_path)}")
        else:
            print(f"\n  No hypertensive events found in any files.")
        
        print(f"{'='*60}")
        
        return results


# Worker function for multiprocessing (must be at module level)
def _process_file_worker(file_output_pair, sampling_rate, threshold, event_duration, 
                        lookback_minutes, map_window_seconds):
    """Worker function for parallel processing"""
    csv_file, output_file = file_output_pair
    
    # Create a new labeler instance for this worker
    labeler = HypertensiveLabeler(
        sampling_rate=sampling_rate,
        threshold=threshold,
        event_duration=event_duration,
        lookback_minutes=lookback_minutes,
        map_window_seconds=map_window_seconds
    )
    
    return labeler.process_single_file(csv_file, output_file)

# Example usage
if __name__ == "__main__":
    # Create labeler instance
    labeler = HypertensiveLabeler(
        sampling_rate=125,
        threshold=110,
        event_duration=60,
        lookback_minutes=5,
        map_window_seconds=10
    )

    # Get all CSV files
    csv_files = glob.glob('Samples/*.csv')
    
    # Option 1: Sequential processing (safest, no freezing)
    # results = labeler.process_multiple_files(
    #     csv_files=csv_files,
    #     output_dir='hypertensive_labeled_output',
    #     num_workers=None  # Sequential
    # )
    
    # Option 2: Parallel processing with multiprocessing
    results = labeler.process_multiple_files(
        csv_files=csv_files,
        output_dir='hypertensive_labeled_output',  # Different output directory
        num_workers=0  # 0 = auto-detect CPU count, or set specific number like 4
    )
    
    # Display sample results
    if results:
        print("\n\nSample results from first file:")
        first_file = list(results.keys())[0]
        print(f"\nFile: {first_file}")
        print(results[first_file].head(10))