import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import Config

def clean_text(text):
    """Clean and normalize text."""
    if pd.isna(text):
        return ""
    return str(text).lower().strip()

def prepare_features_and_labels(input_csv: Path):
    """
    Read documents CSV, clean text, create features, split data.
    Memory-safe using chunked reading.
    """
    print(f"Reading {input_csv}...")
    
    # Read CSV (filter out documents without tags)
    df = pd.read_csv(input_csv)
    print(f"Total documents: {len(df)}")
    
    # Remove documents with missing tags
    df = df.dropna(subset=['document_type'])
    print(f"Documents with tags: {len(df)}")
    
    # Clean text fields
    print("Cleaning text...")
    df['title_clean'] = df['title'].apply(clean_text)
    df['source_clean'] = df['source'].apply(clean_text)
    df['text_clean'] = df['text'].apply(clean_text)
    
    # Combine text features
    df['combined_text'] = (
        df['title_clean'] + ' ' + 
        df['source_clean'] + ' ' + 
        df['text_clean']
    )
    
    # Filter rare tags (document_type appearing < MIN_TAG_FREQUENCY times)
    doc_type_counts = df['document_type'].value_counts()
    valid_types = doc_type_counts[doc_type_counts >= Config.MIN_TAG_FREQUENCY].index
    df = df[df['document_type'].isin(valid_types)]
    print(f"After filtering rare tags (< {Config.MIN_TAG_FREQUENCY} occurrences): {len(df)} documents")
    
    # Split by case_id (train/val/test)
    print("\nSplitting data by cases...")
    unique_cases = df['case_id'].unique()
    
    cases_train, cases_temp = train_test_split(
        unique_cases, 
        test_size=(Config.VAL_RATIO + Config.TEST_RATIO),
        random_state=Config.RANDOM_SEED
    )
    
    cases_val, cases_test = train_test_split(
        cases_temp,
        test_size=Config.TEST_RATIO / (Config.VAL_RATIO + Config.TEST_RATIO),
        random_state=Config.RANDOM_SEED
    )
    
    df_train = df[df['case_id'].isin(cases_train)]
    df_val = df[df['case_id'].isin(cases_val)]
    df_test = df[df['case_id'].isin(cases_test)]
    
    print(f"Train: {len(df_train)} docs from {len(cases_train)} cases")
    print(f"Val:   {len(df_val)} docs from {len(cases_val)} cases")
    print(f"Test:  {len(df_test)} docs from {len(cases_test)} cases")
    
    # Create TF-IDF features
    print("\nCreating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=Config.MAX_FEATURES,
        min_df=Config.MIN_DF,
        max_df=Config.MAX_DF,
        ngram_range=(1, 2)
    )
    
    X_train = vectorizer.fit_transform(df_train['combined_text'])
    X_val = vectorizer.transform(df_val['combined_text'])
    X_test = vectorizer.transform(df_test['combined_text'])
    
    print(f"Feature matrix shape: {X_train.shape}")
    
    # Encode document_type labels (single-label)
    print("\nEncoding document_type labels...")
    from sklearn.preprocessing import LabelEncoder
    encoder_doctype = LabelEncoder()
    y_train_doctype = encoder_doctype.fit_transform(df_train['document_type'])
    y_val_doctype = encoder_doctype.transform(df_val['document_type'])
    y_test_doctype = encoder_doctype.transform(df_test['document_type'])
    
    print(f"Document types: {list(encoder_doctype.classes_)}")
    
    # Encode party_types labels (multi-label)
    print("\nEncoding party_types labels...")
    
    # Parse party_types from string lists
    def parse_party_types(x):
        if pd.isna(x):
            return []
        # Handle string representation of lists
        if isinstance(x, str):
            # Remove brackets and quotes, split by comma
            x = x.strip('[]').replace("'", "").replace('"', '')
            if not x:
                return []
            return [pt.strip() for pt in x.split(',') if pt.strip()]
        return list(x) if hasattr(x, '__iter__') else []
    
    df_train['party_types_list'] = df_train['party_types'].apply(parse_party_types)
    df_val['party_types_list'] = df_val['party_types'].apply(parse_party_types)
    df_test['party_types_list'] = df_test['party_types'].apply(parse_party_types)
    
    encoder_party = MultiLabelBinarizer()
    y_train_party = encoder_party.fit_transform(df_train['party_types_list'])
    y_val_party = encoder_party.transform(df_val['party_types_list'])
    y_test_party = encoder_party.transform(df_test['party_types_list'])
    
    print(f"Party types: {list(encoder_party.classes_)}")
    
    # Save processed data
    print("\nSaving processed data...")
    
    train_data = {
        'X': X_train,
        'y_document_type': y_train_doctype,
        'y_party_types': y_train_party,
        'doc_ids': df_train['doc_id'].values,
        'case_ids': df_train['case_id'].values
    }
    
    val_data = {
        'X': X_val,
        'y_document_type': y_val_doctype,
        'y_party_types': y_val_party,
        'doc_ids': df_val['doc_id'].values,
        'case_ids': df_val['case_id'].values
    }
    
    test_data = {
        'X': X_test,
        'y_document_type': y_test_doctype,
        'y_party_types': y_test_party,
        'doc_ids': df_test['doc_id'].values,
        'case_ids': df_test['case_id'].values
    }
    
    joblib.dump(train_data, Config.PROCESSED_DIR / 'documents_train.pkl')
    joblib.dump(val_data, Config.PROCESSED_DIR / 'documents_val.pkl')
    joblib.dump(test_data, Config.PROCESSED_DIR / 'documents_test.pkl')
    
    # Save vectorizer and encoders
    joblib.dump(vectorizer, Config.PROCESSED_DIR / 'vectorizer.pkl')
    
    encoders = {
        'document_type': encoder_doctype,
        'party_types': encoder_party
    }
    joblib.dump(encoders, Config.PROCESSED_DIR / 'label_encoders.pkl')
    
    print(f"\n✅ Feature processing complete!")
    print(f"   - Saved to: {Config.PROCESSED_DIR}")
    print(f"   - Train set: {len(df_train)} documents")
    print(f"   - Val set: {len(df_val)} documents")
    print(f"   - Test set: {len(df_test)} documents")
    print(f"   - TF-IDF features: {X_train.shape[1]}")
    print(f"   - Document types: {len(encoder_doctype.classes_)}")
    print(f"   - Party types: {len(encoder_party.classes_)}")

if __name__ == '__main__':
    Config.create_dirs()
    input_csv = Config.PROCESSED_DIR / 'documents_all.csv'
    prepare_features_and_labels(input_csv)
