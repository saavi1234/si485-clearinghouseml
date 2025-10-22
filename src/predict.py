import joblib
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import Config

def clean_text(text):
    """Clean and normalize text."""
    if pd.isna(text):
        return ""
    return str(text).lower().strip()

def predict_on_production_data():
    """
    Extract documents from production cases, make predictions, and save results.
    """
    
    print("Loading models and encoders...")
    models = joblib.load(Config.MODEL_DIR / 'document_tagger.pkl')
    vectorizer = joblib.load(Config.PROCESSED_DIR / 'vectorizer.pkl')
    encoders = joblib.load(Config.PROCESSED_DIR / 'label_encoders.pkl')
    
    # Extract production documents
    print("\nExtracting production documents...")
    from extract_documents import process_json_files_in_batches
    
    prod_csv = Config.PROCESSED_DIR / 'documents_production.csv'
    process_json_files_in_batches(Config.CASES_PRODUCTION_DIR, prod_csv)
    
    # Read and process production documents
    print("\nProcessing features...")
    df = pd.read_csv(prod_csv)
    print(f"Total production documents: {len(df)}")
    
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
    
    # Make predictions
    print("\nMaking predictions...")
    
    # Predict document_type
    y_pred_doctype = models['document_type'].predict(X)
    y_pred_doctype_proba = models['document_type'].predict_proba(X)
    df['predicted_document_type'] = encoders['document_type'].inverse_transform(y_pred_doctype)
    df['doctype_confidence'] = y_pred_doctype_proba.max(axis=1)
    
    # Predict party_types
    y_pred_party = models['party_types'].predict(X)
    df['predicted_party_types'] = [
        list(encoders['party_types'].inverse_transform(row.reshape(1, -1))[0])
        for row in y_pred_party
    ]
    
    # Save predictions
    print("\nSaving predictions...")
    output_file = Config.RESULTS_DIR / 'predictions.csv'
    df[[
        'case_id', 
        'doc_id', 
        'title', 
        'source',
        'predicted_document_type',
        'doctype_confidence',
        'predicted_party_types'
    ]].to_csv(output_file, index=False)
    
    print(f"\n✅ Predictions complete!")
    print(f"   - Total documents predicted: {len(df)}")
    print(f"   - Predictions saved to: {output_file}")
    
    # Summary statistics
    print(f"\n📊 Prediction Summary:")
    print(f"\nDocument Type Distribution:")
    doctype_dist = df['predicted_document_type'].value_counts()
    for doctype, count in doctype_dist.head(10).items():
        print(f"   {doctype}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\nAverage confidence: {df['doctype_confidence'].mean():.3f}")
    print(f"Low confidence predictions (<0.5): {(df['doctype_confidence'] < 0.5).sum()}")

if __name__ == '__main__':
    Config.create_dirs()
    predict_on_production_data()
