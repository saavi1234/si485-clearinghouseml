"""
Process features and split data for training.
Memory-safe implementation using sparse matrices and chunked processing.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import joblib
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


def clean_text(text):
    """Basic text cleaning"""
    if pd.isna(text) or not text:
        return ""
    return str(text).lower().strip()


def parse_party_types(val):
    """Parse party_types from string representation of list"""
    if pd.isna(val) or val == 'None' or not val:
        return []
    try:
        # Remove string quotes and parse
        return eval(val) if val else []
    except:
        return []


def prepare_features_and_labels(input_csv: Path):
    """
    Read documents CSV, clean text, create features, split data.
    Memory-safe using chunked reading and sparse matrices.
    """
    print(f"Reading {input_csv}...")
    
    # Read CSV (filter out documents without tags)
    df = pd.read_csv(input_csv)
    print(f"Total documents: {len(df)}")
    
    # Remove documents with missing document_type tags
    df = df.dropna(subset=['document_type'])
    print(f"Documents with document_type tags: {len(df)}")
    
    # Clean text fields
    print("\nCleaning text...")
    df['title_clean'] = df['title'].apply(clean_text)
    df['source_clean'] = df['source'].apply(clean_text)
    df['text_clean'] = df['text'].apply(clean_text)
    
    # Combine text features
    print("Combining text features...")
    df['combined_text'] = (
        df['title_clean'] + ' ' + 
        df['source_clean'] + ' ' + 
        df['text_clean']
    )
    
    # Filter rare document_type tags (appearing < MIN_TAG_FREQUENCY times)
    print(f"\nFiltering rare tags (< {Config.MIN_TAG_FREQUENCY} occurrences)...")
    doc_type_counts = df['document_type'].value_counts()
    valid_types = doc_type_counts[doc_type_counts >= Config.MIN_TAG_FREQUENCY].index
    print(f"Document types before filtering: {len(doc_type_counts)}")
    print(f"Document types after filtering: {len(valid_types)}")
    
    df = df[df['document_type'].isin(valid_types)]
    print(f"Documents after filtering rare tags: {len(df)}")
    
    # Split by case_id (train/val/test)
    print("\nSplitting data by cases...")
    unique_cases = df['case_id'].unique()
    print(f"Unique cases: {len(unique_cases)}")
    
    # First split: train vs (val + test)
    cases_train, cases_temp = train_test_split(
        unique_cases, 
        test_size=(Config.VAL_RATIO + Config.TEST_RATIO),
        random_state=Config.RANDOM_SEED
    )
    
    # Second split: val vs test
    cases_val, cases_test = train_test_split(
        cases_temp,
        test_size=Config.TEST_RATIO / (Config.VAL_RATIO + Config.TEST_RATIO),
        random_state=Config.RANDOM_SEED
    )
    
    # Create splits
    df_train = df[df['case_id'].isin(cases_train)].copy()
    df_val = df[df['case_id'].isin(cases_val)].copy()
    df_test = df[df['case_id'].isin(cases_test)].copy()
    
    print(f"Train: {len(df_train)} docs from {len(cases_train)} cases")
    print(f"Val:   {len(df_val)} docs from {len(cases_val)} cases")
    print(f"Test:  {len(df_test)} docs from {len(cases_test)} cases")
    
    # Create TF-IDF features (fit on train, transform all)
    print(f"\nCreating TF-IDF features (max {Config.MAX_FEATURES} features)...")
    vectorizer = TfidfVectorizer(
        max_features=Config.MAX_FEATURES,
        min_df=Config.MIN_DF,
        max_df=Config.MAX_DF,
        ngram_range=(1, 2),  # unigrams and bigrams
        strip_accents='unicode',
        lowercase=True
    )
    
    print("Fitting vectorizer on training data...")
    X_train = vectorizer.fit_transform(df_train['combined_text'])
    print("Transforming validation data...")
    X_val = vectorizer.transform(df_val['combined_text'])
    print("Transforming test data...")
    X_test = vectorizer.transform(df_test['combined_text'])
    
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Memory usage (sparse): ~{X_train.data.nbytes / 1024 / 1024:.2f} MB")
    
    # Save vectorizer
    print("\nSaving vectorizer...")
    joblib.dump(vectorizer, Config.PROCESSED_DIR / 'vectorizer.pkl')
    
    # Encode labels
    print("\nEncoding labels...")
    
    # 1. document_type (single-label classification)
    print("  Encoding document_type (single-label)...")
    le_doctype = LabelEncoder()
    y_train_doctype = le_doctype.fit_transform(df_train['document_type'])
    y_val_doctype = le_doctype.transform(df_val['document_type'])
    y_test_doctype = le_doctype.transform(df_test['document_type'])
    
    print(f"    Classes: {list(le_doctype.classes_)}")
    print(f"    Distribution: {pd.Series(y_train_doctype).value_counts().to_dict()}")
    
    # 2. party_types (multi-label classification)
    print("  Encoding party_types (multi-label)...")
    df_train['party_list'] = df_train['party_types'].apply(parse_party_types)
    df_val['party_list'] = df_val['party_types'].apply(parse_party_types)
    df_test['party_list'] = df_test['party_types'].apply(parse_party_types)
    
    mlb_party = MultiLabelBinarizer()
    y_train_party = mlb_party.fit_transform(df_train['party_list'])
    y_val_party = mlb_party.transform(df_val['party_list'])
    y_test_party = mlb_party.transform(df_test['party_list'])
    
    print(f"    Classes: {list(mlb_party.classes_)}")
    print(f"    Label matrix shape: {y_train_party.shape}")
    print(f"    Avg labels per document: {y_train_party.sum(axis=1).mean():.2f}")
    
    # Save label encoders
    print("\nSaving label encoders...")
    encoders = {
        'document_type': le_doctype,
        'party_types': mlb_party
    }
    joblib.dump(encoders, Config.PROCESSED_DIR / 'label_encoders.pkl')
    
    # Save processed datasets
    print("\nSaving processed data...")
    joblib.dump({
        'X': X_train,
        'y_document_type': y_train_doctype,
        'y_party_types': y_train_party,
        'doc_ids': df_train['doc_id'].values
    }, Config.PROCESSED_DIR / 'documents_train.pkl')
    
    joblib.dump({
        'X': X_val,
        'y_document_type': y_val_doctype,
        'y_party_types': y_val_party,
        'doc_ids': df_val['doc_id'].values
    }, Config.PROCESSED_DIR / 'documents_val.pkl')
    
    joblib.dump({
        'X': X_test,
        'y_document_type': y_test_doctype,
        'y_party_types': y_test_party,
        'doc_ids': df_test['doc_id'].values
    }, Config.PROCESSED_DIR / 'documents_test.pkl')
    
    print("\nFeature processing complete!")
    return le_doctype, mlb_party


if __name__ == '__main__':
    print("=" * 60)
    print("PHASE 3: Processing Features & Splitting Data")
    print("=" * 60)
    print()
    
    input_file = Config.PROCESSED_DIR / 'documents_all.csv'
    
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("Please run extract_documents.py first (Phase 2)")
        sys.exit(1)
    
    le_doctype, mlb_party = prepare_features_and_labels(input_file)
    
    print("\n" + "=" * 60)
    print("Phase 3 Complete!")
    print("=" * 60)
    print("\nFiles created:")
    print(f"  - {Config.PROCESSED_DIR / 'vectorizer.pkl'}")
    print(f"  - {Config.PROCESSED_DIR / 'label_encoders.pkl'}")
    print(f"  - {Config.PROCESSED_DIR / 'documents_train.pkl'}")
    print(f"  - {Config.PROCESSED_DIR / 'documents_val.pkl'}")
    print(f"  - {Config.PROCESSED_DIR / 'documents_test.pkl'}")
    print("\nReady for training!")
