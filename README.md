# iot_cdm

# How to Run the Project

## Prerequisites

- Python 3.8+ recommended
- Install required packages (TensorFlow, scikit-learn, pandas, numpy, matplotlib, seaborn, tqdm, etc.)
- You may use:
	```
	pip install tensorflow scikit-learn pandas numpy matplotlib seaborn tqdm
	```

---

## Hyperglycemia Pipeline

### 1. Data Augmentation

- Script: `hyperglycemia/augmentation.py`
- Usage:
	Edit the `input_file` and `output_file` in the `__main__` block if needed.
- Run:
	```
	python hyperglycemia/augmentation.py
	```
- Output: `augmented_glucose_data.csv` (used for model training)

### 2. Model Training & Evaluation

- Script: `hyperglycemia/gly.py`
- Usage:
	By default, uses the augmented data.
- Run:
	```
	python hyperglycemia/gly.py
	```
- Output: Trains and compares CNN-only, LSTM-only, and CNN-LSTM models.
	Models are saved as `.h5` files (e.g., `hyperglycemia_cnn_lstm_model.h5`).
	Metrics and plots are generated for comparison.

---

## Hypertension Pipeline

### 1. Data Augmentation

- Script: `hypertension/augmenter.py`
- Usage:
	Augments minority class CSV files in `hypertensive_labeled_output_fixed/`.
- Run:
	```
	python hypertension/augmenter.py
	```
- Output: Augmented files in `augmented_labeled_files_fixed/`.

### 2. Spectrogram Generation

- Script: `hypertension/balanced_spectrogram.py`
- Usage:
	Converts balanced CSV files to spectrograms.
- Run:
	```
	python hypertension/balanced_spectrogram.py
	```
- Output: Spectrogram `.npz` files.

### 3. Model Training & Evaluation

- Script: `hypertension/hypertensive_model.py`
- Usage:
	Trains CNN and CNN-LSTM models on spectrogram data.
- Run:
	```
	python hypertension/hypertensive_model.py
	```
- Output: Model files (e.g., `hypertension_cnn_lstm.keras`), metrics, and plots.

### 4. ROC Analysis

- Script: `hypertension/hypertension_roc.py`
- Usage:
	Evaluates trained models and plots ROC curves.
- Run:
	```
	python hypertension/hypertension_roc.py
	```

### 5. Separability Analysis

- Script: `hypertension/seperability_analysis.py`
- Usage:
	Analyzes class separability in spectrograms.
- Run:
	```
	python hypertension/seperability_analysis.py
	```
- Output: Results in `separability_analysis/`.

---

## Notes

- All scripts can be run from the command line.
- Adjust file paths in scripts if your data is in a different location.
- Outputs (models, metrics, plots) are saved in their respective folders.

---

For more details, see comments and docstrings in each script.