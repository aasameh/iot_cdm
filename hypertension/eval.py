import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf

def load_test_data(npz_path='balanced_dataset.npz', test_size=0.2, val_size=0.25, random_state=42):
    """Load and split data to get the same test set as training"""
    from sklearn.model_selection import train_test_split
    
    data = np.load(npz_path)
    X, y = data['spectrograms'], data['labels']
    label_map = {'negative': 0, 'positive': 1, 'hypertensive_event': 2}
    y = np.array([label_map[label] for label in y])
    
    # Split exactly as in training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    
    return X_test, y_test

def evaluate_model_precision(model_path, X_test, y_test, model_name):
    """Evaluate model with high precision calculations"""
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = y_pred_proba.argmax(axis=1)
    
    # Calculate metrics with full precision
    accuracy = accuracy_score(y_test, y_pred) * 100
    
    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None) * 100
    recall_per_class = recall_score(y_test, y_pred, average=None) * 100
    f1_per_class = f1_score(y_test, y_pred, average=None) * 100
    
    # Macro averages
    precision_macro = precision_score(y_test, y_pred, average='macro') * 100
    recall_macro = recall_score(y_test, y_pred, average='macro') * 100
    f1_macro = f1_score(y_test, y_pred, average='macro') * 100
    
    # AUROC
    auroc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') * 100
    
    # Binary AUC for each class vs rest
    auc_per_class = []
    for i in range(3):
        y_binary = (y_test == i).astype(int)
        auc = roc_auc_score(y_binary, y_pred_proba[:, i]) * 100
        auc_per_class.append(auc)
    
    return {
        'accuracy': accuracy,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'auroc': auroc,
        'auc_per_class': auc_per_class
    }

def print_results(results, model_name):
    """Print results in formatted style"""
    class_names = ['Negative', 'Positive', 'Hypertensive']
    
    print("\n" + "="*70)
    print(f"{model_name} MODEL EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nOverall Accuracy: {results['accuracy']:.2f}%")
    print(f"Overall AUROC:    {results['auroc']:.2f}%")
    
    print("\n" + "-"*70)
    print("Per-Class Metrics:")
    print("-"*70)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC':<12}")
    print("-"*70)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} "
              f"{results['precision_per_class'][i]:>10.2f}%  "
              f"{results['recall_per_class'][i]:>9.2f}%  "
              f"{results['f1_per_class'][i]:>9.2f}%  "
              f"{results['auc_per_class'][i]:>9.2f}%")
    
    print("-"*70)
    print(f"{'Macro Average':<15} "
          f"{results['precision_macro']:>10.2f}%  "
          f"{results['recall_macro']:>9.2f}%  "
          f"{results['f1_macro']:>9.2f}%")
    print("="*70)

def compare_models_precision(cnn_path='results/cnn_model.keras', 
                             lstm_path='results/cnn_lstm_model.keras',
                             data_path='balanced_dataset.npz'):
    """Compare both models with precision formatting"""
    
    print("Loading test data...")
    X_test, y_test = load_test_data(data_path)
    print(f"Test samples: {len(X_test)}")
    
    # Evaluate CNN
    print("\nEvaluating CNN model...")
    cnn_results = evaluate_model_precision(cnn_path, X_test, y_test, 'CNN')
    print_results(cnn_results, 'CNN')
    
    # Evaluate CNN-LSTM
    print("\nEvaluating CNN-LSTM model...")
    lstm_results = evaluate_model_precision(lstm_path, X_test, y_test, 'CNN-LSTM')
    print_results(lstm_results, 'CNN-LSTM')
    
    # Comparison summary
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Metric':<20} {'CNN':<15} {'CNN-LSTM':<15} {'Difference':<15}")
    print("-"*70)
    print(f"{'Accuracy':<20} {cnn_results['accuracy']:>13.2f}%  {lstm_results['accuracy']:>13.2f}%  "
          f"{lstm_results['accuracy']-cnn_results['accuracy']:>+13.2f}%")
    print(f"{'AUROC':<20} {cnn_results['auroc']:>13.2f}%  {lstm_results['auroc']:>13.2f}%  "
          f"{lstm_results['auroc']-cnn_results['auroc']:>+13.2f}%")
    print(f"{'Precision (macro)':<20} {cnn_results['precision_macro']:>13.2f}%  {lstm_results['precision_macro']:>13.2f}%  "
          f"{lstm_results['precision_macro']-cnn_results['precision_macro']:>+13.2f}%")
    print(f"{'Recall (macro)':<20} {cnn_results['recall_macro']:>13.2f}%  {lstm_results['recall_macro']:>13.2f}%  "
          f"{lstm_results['recall_macro']-cnn_results['recall_macro']:>+13.2f}%")
    print(f"{'F1-Score (macro)':<20} {cnn_results['f1_macro']:>13.2f}%  {lstm_results['f1_macro']:>13.2f}%  "
          f"{lstm_results['f1_macro']-cnn_results['f1_macro']:>+13.2f}%")
    print("="*70)

if __name__ == "__main__":
    compare_models_precision()