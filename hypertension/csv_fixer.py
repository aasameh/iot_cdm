import pandas as pd
import glob
import os
import csv
import shutil

def inspect_csv_file(csv_file, max_lines=50):
    """
    Inspect a CSV file to identify issues.
    """
    print(f"\nInspecting: {os.path.basename(csv_file)}")
    
    issues = []
    
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        
        if not lines:
            issues.append("File is empty")
            return issues
        
        # Check header
        header = lines[0].strip()
        header_fields = header.split(',')
        print(f"  Header: {header_fields}")
        print(f"  Expected 5 fields: ECG, ABP, MAP, PPG, label")
        
        if len(header_fields) != 5:
            issues.append(f"Header has {len(header_fields)} fields instead of 5")
        
        # Check first few data lines
        print(f"  Checking first {min(max_lines, len(lines)-1)} data lines...")
        
        problem_lines = []
        for i, line in enumerate(lines[1:max_lines+1], start=2):
            fields = line.strip().split(',')
            if len(fields) != 5:
                problem_lines.append((i, len(fields), line.strip()[:100]))
        
        if problem_lines:
            issues.append(f"Found {len(problem_lines)} lines with wrong field count")
            print(f"  Problem lines (showing first 3):")
            for line_num, field_count, content in problem_lines[:3]:
                print(f"    Line {line_num}: {field_count} fields - {content}")
    
    if not issues:
        print(f"  ✓ No issues found")
    else:
        print(f"  ✗ Issues found:")
        for issue in issues:
            print(f"    - {issue}")
    
    return issues

def fix_csv_file(input_file, output_file=None):
    """
    Fix a malformed CSV file.
    """
    if output_file is None:
        output_file = input_file.replace('.csv', '_fixed.csv')
    
    print(f"\nFixing: {os.path.basename(input_file)}")
    
    fixed_rows = []
    skipped_lines = 0
    
    with open(input_file, 'r') as f:
        # Read and validate header
        header_line = f.readline().strip()
        header = header_line.split(',')
        
        if len(header) != 5:
            print(f"  ✗ Invalid header: {header}")
            return False
        
        fixed_rows.append(header)
        
        # Process data lines
        reader = csv.reader(f)
        for line_num, row in enumerate(reader, start=2):
            try:
                if len(row) == 5:
                    # Perfect row
                    fixed_rows.append(row)
                elif len(row) > 5:
                    # Too many fields - likely comma in label
                    # Merge everything after first 4 fields into label
                    fixed_row = row[:4] + [' '.join(row[4:])]
                    fixed_rows.append(fixed_row)
                elif len(row) < 5:
                    # Too few fields - skip
                    skipped_lines += 1
                else:
                    skipped_lines += 1
            except Exception as e:
                skipped_lines += 1
                continue
    
    # Write fixed file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(fixed_rows)
    
    print(f"  ✓ Fixed {len(fixed_rows)-1} rows")
    if skipped_lines > 0:
        print(f"  ⚠ Skipped {skipped_lines} problematic lines")
    print(f"  Saved to: {output_file}")
    
    return True

def verify_fixed_file(csv_file):
    """
    Verify that a fixed file can be loaded properly.
    """
    try:
        df = pd.read_csv(csv_file)
        
        required_cols = ['ECG', 'ABP', 'MAP', 'PPG', 'label']
        if not all(col in df.columns for col in required_cols):
            print(f"  ✗ Missing columns: {[c for c in required_cols if c not in df.columns]}")
            return False
        
        # Check for NaN values
        nan_counts = df[required_cols].isna().sum()
        if nan_counts.any():
            print(f"  ⚠ Found NaN values: {nan_counts[nan_counts > 0].to_dict()}")
        
        print(f"  ✓ Successfully loaded: {len(df)} rows")
        print(f"    Label distribution: {df['label'].value_counts().to_dict()}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error loading: {str(e)}")
        return False

def fix_all_files_in_directory(directory, output_directory=None, backup=True):
    """
    Fix all CSV files in a directory.
    """
    if output_directory is None:
        output_directory = directory + '_fixed'
    
    os.makedirs(output_directory, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return
    
    print("="*70)
    print(f"FIXING {len(csv_files)} CSV FILES")
    print("="*70)
    
    results = {
        'total': len(csv_files),
        'fixed': 0,
        'failed': 0,
        'no_issues': 0
    }
    
    for csv_file in csv_files:
        base_name = os.path.basename(csv_file)
        output_file = os.path.join(output_directory, base_name)
        
        # First, inspect the file
        issues = inspect_csv_file(csv_file)
        
        if not issues:
            # No issues, just copy
            if backup:
                shutil.copy2(csv_file, output_file)
                print(f"  ✓ Copied (no fixes needed)")
            results['no_issues'] += 1
        else:
            # Try to fix
            if fix_csv_file(csv_file, output_file):
                # Verify the fix worked
                if verify_fixed_file(output_file):
                    results['fixed'] += 1
                else:
                    results['failed'] += 1
            else:
                results['failed'] += 1
        
        print()
    
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total files: {results['total']}")
    print(f"  No issues: {results['no_issues']}")
    print(f"  Fixed: {results['fixed']}")
    print(f"  Failed: {results['failed']}")
    print(f"\nFixed files saved to: {output_directory}")
    print("="*70)

if __name__ == "__main__":
    # Run the fixer on your directory
    fix_all_files_in_directory(
        directory='hypertensive_labeled_output',
        output_directory='hypertensive_labeled_output_fixed',
        backup=True  # Copy good files too
    )
    
    # Now you can use the fixed directory in your pipeline:
    # input_directory='hypertensive_labeled_output_fixed'