import joblib
import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    confusion_matrix,
    hamming_loss
)

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import Config

def evaluate_on_test_set():
    """
    Evaluate trained models on test set and save detailed metrics.
    """
    
    print("Loading models and test data...")
    models = joblib.load(Config.MODEL_DIR / 'document_tagger.pkl')
    encoders = joblib.load(Config.PROCESSED_DIR / 'label_encoders.pkl')
    test_data = joblib.load(Config.PROCESSED_DIR / 'documents_test.pkl')
    
    X_test = test_data['X']
    y_test_doctype = test_data['y_document_type']
    y_test_party = test_data['y_party_types']
    doc_ids = test_data['doc_ids']
    
    print(f"Test set size: {len(doc_ids)} documents")
    
    # ========== Document Type Evaluation ==========
    print("\n" + "="*60)
    print("EVALUATING DOCUMENT_TYPE CLASSIFIER")
    print("="*60)
    
    y_pred_doctype = models['document_type'].predict(X_test)
    y_pred_doctype_proba = models['document_type'].predict_proba(X_test)
    
    # Overall metrics
    doctype_accuracy = accuracy_score(y_test_doctype, y_pred_doctype)
    doctype_precision = precision_score(y_test_doctype, y_pred_doctype, average='weighted', zero_division=0)
    doctype_recall = recall_score(y_test_doctype, y_pred_doctype, average='weighted', zero_division=0)
    doctype_f1_weighted = f1_score(y_test_doctype, y_pred_doctype, average='weighted', zero_division=0)
    doctype_f1_macro = f1_score(y_test_doctype, y_pred_doctype, average='macro', zero_division=0)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:           {doctype_accuracy:.4f}")
    print(f"  Precision (weighted): {doctype_precision:.4f}")
    print(f"  Recall (weighted):    {doctype_recall:.4f}")
    print(f"  F1 Score (weighted):  {doctype_f1_weighted:.4f}")
    print(f"  F1 Score (macro):     {doctype_f1_macro:.4f}")
    
    # Per-class report
    print("\nPer-Class Performance:")
    class_names = encoders['document_type'].classes_
    print(classification_report(
        y_test_doctype, 
        y_pred_doctype, 
        target_names=class_names,
        zero_division=0
    ))
    
    # ========== Party Types Evaluation ==========
    print("\n" + "="*60)
    print("EVALUATING PARTY_TYPES CLASSIFIER")
    print("="*60)
    
    y_pred_party = models['party_types'].predict(X_test)
    
    # Multi-label metrics
    party_hamming = hamming_loss(y_test_party, y_pred_party)
    party_precision_micro = precision_score(y_test_party, y_pred_party, average='micro', zero_division=0)
    party_recall_micro = recall_score(y_test_party, y_pred_party, average='micro', zero_division=0)
    party_f1_micro = f1_score(y_test_party, y_pred_party, average='micro', zero_division=0)
    party_f1_macro = f1_score(y_test_party, y_pred_party, average='macro', zero_division=0)
    party_f1_samples = f1_score(y_test_party, y_pred_party, average='samples', zero_division=0)
    
    # Subset accuracy (exact match)
    subset_accuracy = np.mean([
        np.array_equal(y_test_party[i], y_pred_party[i]) 
        for i in range(len(y_test_party))
    ])
    
    print(f"\nOverall Metrics:")
    print(f"  Hamming Loss:         {party_hamming:.4f}")
    print(f"  Subset Accuracy:      {subset_accuracy:.4f}")
    print(f"  Precision (micro):    {party_precision_micro:.4f}")
    print(f"  Recall (micro):       {party_recall_micro:.4f}")
    print(f"  F1 Score (micro):     {party_f1_micro:.4f}")
    print(f"  F1 Score (macro):     {party_f1_macro:.4f}")
    print(f"  F1 Score (samples):   {party_f1_samples:.4f}")
    
    # Per-class report for party types
    print("\nPer-Class Performance:")
    party_names = encoders['party_types'].classes_
    print(classification_report(
        y_test_party, 
        y_pred_party, 
        target_names=party_names,
        zero_division=0
    ))
    
    # ========== Save Predictions ==========
    print("\n" + "="*60)
    print("SAVING TEST SET PREDICTIONS")
    print("="*60)
    
    # Decode predictions
    pred_doctype_labels = encoders['document_type'].inverse_transform(y_pred_doctype)
    pred_party_labels = [
        list(encoders['party_types'].inverse_transform(row.reshape(1, -1))[0])
        for row in y_pred_party
    ]
    
    # Decode ground truth
    true_doctype_labels = encoders['document_type'].inverse_transform(y_test_doctype)
    true_party_labels = [
        list(encoders['party_types'].inverse_transform(row.reshape(1, -1))[0])
        for row in y_test_party
    ]
    
    # Get prediction confidence (max probability for document_type)
    pred_confidence = np.max(y_pred_doctype_proba, axis=1)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'doc_id': doc_ids,
        'true_document_type': true_doctype_labels,
        'predicted_document_type': pred_doctype_labels,
        'doctype_confidence': pred_confidence,
        'doctype_correct': y_test_doctype == y_pred_doctype,
        'true_party_types': [str(p) for p in true_party_labels],
        'predicted_party_types': [str(p) for p in pred_party_labels],
        'party_exact_match': [
            np.array_equal(y_test_party[i], y_pred_party[i])
            for i in range(len(y_test_party))
        ]
    })
    
    # Save predictions CSV
    output_csv = Config.RESULTS_DIR / 'test_predictions.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"Saved predictions to: {output_csv}")
    
    # ========== Save Metrics JSON ==========
    metrics = {
        'test_set_size': len(doc_ids),
        'document_type': {
            'accuracy': float(doctype_accuracy),
            'precision_weighted': float(doctype_precision),
            'recall_weighted': float(doctype_recall),
            'f1_weighted': float(doctype_f1_weighted),
            'f1_macro': float(doctype_f1_macro),
            'num_classes': len(class_names),
            'classes': list(class_names)
        },
        'party_types': {
            'hamming_loss': float(party_hamming),
            'subset_accuracy': float(subset_accuracy),
            'precision_micro': float(party_precision_micro),
            'recall_micro': float(party_recall_micro),
            'f1_micro': float(party_f1_micro),
            'f1_macro': float(party_f1_macro),
            'f1_samples': float(party_f1_samples),
            'num_classes': len(party_names),
            'classes': list(party_names)
        }
    }
    
    output_json = Config.RESULTS_DIR / 'test_metrics.json'
    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {output_json}")
    
    # ========== Summary ==========
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"✅ Test set evaluation complete!")
    print(f"   - Document Type Accuracy: {doctype_accuracy:.1%}")
    print(f"   - Party Types Subset Accuracy: {subset_accuracy:.1%}")
    print(f"   - Results saved to: {Config.RESULTS_DIR}")
    
    return metrics

if __name__ == '__main__':
    Config.create_dirs()
    metrics = evaluate_on_test_set()
