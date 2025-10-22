"""
Make predictions on new documents.
Can work with either production data or test set.
Memory-safe batch processing.
"""
import json
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


def clean_text(text):
    """Basic text cleaning (same as training)"""
    if pd.isna(text) or not text:
        return ""
    return str(text).lower().strip()


def predict_on_production_data(cases_dir: Path, output_csv: Path):
    """
    Predict tags for all documents in production cases.
    Memory-safe batch processing.
    """
    
    print("Loading models and preprocessors...")
    models = joblib.load(Config.MODEL_DIR / 'document_tagger.pkl')
    encoders = joblib.load(Config.PROCESSED_DIR / 'label_encoders.pkl')
    vectorizer = joblib.load(Config.PROCESSED_DIR / 'vectorizer.pkl')
    
    print("Models loaded successfully!")
    print(f"  Document types: {len(encoders['document_type'].classes_)}")
    print(f"  Party types: {len(encoders['party_types'].classes_)}")
    
    # Extract documents from production cases (same as training)
    print(f"\nExtracting documents from {cases_dir}...")
    from extract_documents import process_json_files_in_batches
    
    temp_csv = Config.PROCESSED_DIR / 'documents_production_temp.csv'
    process_json_files_in_batches(cases_dir, temp_csv)
    
    # Read and process production documents
    print("\nProcessing features...")
    df = pd.read_csv(temp_csv)
    print(f"Total documents: {len(df)}")
    
    # Clean text (same as training)
    df['title_clean'] = df['title'].apply(clean_text)
    df['source_clean'] = df['source'].apply(clean_text)
    df['text_clean'] = df['text'].apply(clean_text)
    
    df['combined_text'] = (
        df['title_clean'] + ' ' +
        df['source_clean'] + ' ' +
        df['text_clean']
    )
    
    # Vectorize
    print("Creating TF-IDF features...")
    X = vectorizer.transform(df['combined_text'])
    print(f"Feature matrix shape: {X.shape}")
    
    # Make predictions
    print("\nMaking predictions...")
    
    # Predict document_type
    print("  Predicting document_type...")
    y_pred_doctype = models['document_type'].predict(X)
    df['predicted_document_type'] = encoders['document_type'].inverse_transform(y_pred_doctype)
    
    # Get prediction probabilities for confidence scores
    y_pred_proba = models['document_type'].predict_proba(X)
    df['document_type_confidence'] = y_pred_proba.max(axis=1)
    
    # Predict party_types
    print("  Predicting party_types...")
    y_pred_party = models['party_types'].predict(X)
    
    # Convert binary matrix to list of labels
    predicted_party_lists = []
    for row in y_pred_party:
        indices = [i for i, val in enumerate(row) if val == 1]
        labels = encoders['party_types'].classes_[indices] if len(indices) > 0 else []
        predicted_party_lists.append(list(labels))
    df['predicted_party_types'] = predicted_party_lists
    
    # Calculate confidence for multi-label (average probability of predicted labels)
    y_pred_party_proba = models['party_types'].predict_proba(X)
    party_confidence = []
    for i, row in enumerate(y_pred_party):
        if row.sum() == 0:  # No labels predicted
            party_confidence.append(0.0)
        else:
            # Average probability of predicted labels
            predicted_indices = row.nonzero()[0]
            probs = [y_pred_party_proba[i][:, 1][idx] for idx in predicted_indices]
            party_confidence.append(sum(probs) / len(probs) if probs else 0.0)
    df['party_types_confidence'] = party_confidence
    
    # Save predictions
    print(f"\nSaving predictions to {output_csv}...")
    output_cols = [
        'case_id', 'doc_id', 'title', 
        'predicted_document_type', 'document_type_confidence',
        'predicted_party_types', 'party_types_confidence'
    ]
    df[output_cols].to_csv(output_csv, index=False)
    
    print(f"Predictions saved!")
    print(f"Total documents predicted: {len(df)}")
    
    # Show prediction distribution
    print("\n" + "=" * 60)
    print("Prediction Distribution")
    print("=" * 60)
    print("\nDocument Type Predictions:")
    print(df['predicted_document_type'].value_counts().head(10))
    
    print(f"\nAverage Confidence Scores:")
    print(f"  Document Type: {df['document_type_confidence'].mean():.4f}")
    print(f"  Party Types: {df['party_types_confidence'].mean():.4f}")
    
    # Clean up temp file
    if temp_csv.exists():
        temp_csv.unlink()
    
    return df


def predict_on_test_set(output_csv: Path):
    """
    Make predictions on the test set for demonstration.
    This shows how well the model performs on unseen data.
    """
    
    print("Loading models and test data...")
    models = joblib.load(Config.MODEL_DIR / 'document_tagger.pkl')
    encoders = joblib.load(Config.PROCESSED_DIR / 'label_encoders.pkl')
    test_data = joblib.load(Config.PROCESSED_DIR / 'documents_test.pkl')
    
    X_test = test_data['X']
    y_true_doctype = test_data['y_document_type']
    y_true_party = test_data['y_party_types']
    doc_ids = test_data['doc_ids']
    
    print(f"Test set: {X_test.shape[0]} documents")
    
    # Make predictions
    print("\nMaking predictions on test set...")
    
    # Predict document_type
    y_pred_doctype = models['document_type'].predict(X_test)
    pred_doc_types = encoders['document_type'].inverse_transform(y_pred_doctype)
    true_doc_types = encoders['document_type'].inverse_transform(y_true_doctype)
    
    # Get confidence
    y_pred_proba = models['document_type'].predict_proba(X_test)
    doc_confidence = y_pred_proba.max(axis=1)
    
    # Predict party_types
    y_pred_party = models['party_types'].predict(X_test)
    
    # Convert binary matrices to lists of labels
    pred_party_types = []
    for row in y_pred_party:
        indices = [i for i, val in enumerate(row) if val == 1]
        labels = encoders['party_types'].classes_[indices] if len(indices) > 0 else []
        pred_party_types.append(list(labels))
    
    true_party_types = []
    for row in y_true_party:
        indices = [i for i, val in enumerate(row) if val == 1]
        labels = encoders['party_types'].classes_[indices] if len(indices) > 0 else []
        true_party_types.append(list(labels))
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, hamming_loss
    
    print("\n" + "=" * 60)
    print("Test Set Performance")
    print("=" * 60)
    
    doc_acc = accuracy_score(y_true_doctype, y_pred_doctype)
    doc_f1 = f1_score(y_true_doctype, y_pred_doctype, average='weighted')
    print(f"\nDocument Type:")
    print(f"  Accuracy: {doc_acc:.4f}")
    print(f"  F1 (weighted): {doc_f1:.4f}")
    
    party_subset_acc = accuracy_score(y_true_party, y_pred_party)
    party_f1_micro = f1_score(y_true_party, y_pred_party, average='micro')
    party_hamming = hamming_loss(y_true_party, y_pred_party)
    print(f"\nParty Types:")
    print(f"  Subset Accuracy: {party_subset_acc:.4f}")
    print(f"  F1 (micro): {party_f1_micro:.4f}")
    print(f"  Hamming Loss: {party_hamming:.4f}")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'doc_id': doc_ids,
        'true_document_type': true_doc_types,
        'predicted_document_type': pred_doc_types,
        'document_type_confidence': doc_confidence,
        'document_type_correct': (y_true_doctype == y_pred_doctype),
        'true_party_types': [str(x) for x in true_party_types],
        'predicted_party_types': [str(x) for x in pred_party_types],
        'party_types_correct': [
            set(true) == set(pred) 
            for true, pred in zip(true_party_types, pred_party_types)
        ]
    })
    
    # Save results
    print(f"\nSaving test predictions to {output_csv}...")
    results_df.to_csv(output_csv, index=False)
    
    print("\nSample predictions (first 10):")
    print(results_df[['doc_id', 'predicted_document_type', 'document_type_confidence', 
                      'document_type_correct']].head(10).to_string())
    
    return results_df


if __name__ == '__main__':
    print("=" * 60)
    print("PHASE 5: Making Predictions")
    print("=" * 60)
    print()
    
    # Check if models exist
    if not (Config.MODEL_DIR / 'document_tagger.pkl').exists():
        print("ERROR: Model not found. Please run train_model.py first (Phase 4)")
        sys.exit(1)
    
    Config.create_dirs()
    
    # Check if production data exists
    if Config.CASES_PRODUCTION_DIR.exists() and list(Config.CASES_PRODUCTION_DIR.glob('case_*.json')):
        print("Production data found! Running predictions on production cases...")
        output_file = Config.RESULTS_DIR / 'predictions_production.csv'
        predict_on_production_data(Config.CASES_PRODUCTION_DIR, output_file)
    else:
        print("No production data found in cases/production/")
        print("Running predictions on test set instead for demonstration...\n")
        output_file = Config.RESULTS_DIR / 'predictions_test.csv'
        predict_on_test_set(output_file)
    
    print("\n" + "=" * 60)
    print("Phase 5 Complete!")
    print("=" * 60)
    print(f"\nPredictions saved to: {output_file}")
    print("\nYou can now:")
    print("  1. Review the predictions")
    print("  2. Add production cases to cases/production/ and re-run")
    print("  3. Use the model for inference in your application")
