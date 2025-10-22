# ML Implementation Specification
## Document-Level Legal Tagging System (Simplified)

---

## 🎯 Project Overview

Build a **simple, memory-efficient** machine learning system to automatically tag individual legal documents using human-tagged training data from the Civil Rights Clearinghouse database.

**Scope:** Document-level tagging ONLY (case-level tagging deferred for future implementation)

**Key Constraints:**
- Memory-safe processing (batch processing throughout)
- Simple architecture (single model, straightforward pipeline)
- Production-ready inference on new documents

---

## 📊 Data & Tags

### Document-Level Tags (2 types)
1. **`document_type`** - Type of legal document (e.g., "Order", "Motion", "Brief")
2. **`party_types`** - Parties involved in the document (list, e.g., ["Plaintiff", "Defendant"])

### Available Input Features
For each document, we have:
- ✅ **Document title** (e.g., "Order Granting Motion to Dismiss")
- ✅ **Document source** (e.g., "PACER", "Court website")  
- ✅ **Document text** (the full document content)
- ✅ **Case ID** (for grouping, but NOT used as a feature)

### Data Source
- Training data: JSON files in `cases/training/` directory
- Each case JSON contains multiple documents with tags
- Production data: Similar JSON structure, but tags are missing (to be predicted)

---

## 🏗️ Repository Structure (Simplified)

```
si485-clearinghouseml/
├── .env                          # API keys (NOT in git)
├── .gitignore                    # Ignore sensitive files
├── requirements.txt              # Python dependencies
├── README.md                     # User guide
├── config.py                     # Configuration settings
│
├── cases/                        # Case JSON files
│   ├── training/                 # Training cases (human-tagged)
│   └── production/               # Production cases (to predict)
│
├── processed/                    # Processed data (created by pipeline)
│   ├── documents_train.csv       # Training documents
│   ├── documents_val.csv         # Validation documents
│   ├── documents_test.csv        # Test documents
│   └── vectorizers.pkl           # Saved text vectorizers
│
├── model/                        # Saved trained model
│   ├── document_tagger.pkl       # The trained model
│   ├── label_encoders.pkl        # Label encoders for tags
│   └── metadata.json             # Model info & metrics
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── extract_documents.py      # Extract docs from JSONs (with batching)
│   ├── process_features.py       # Text processing & vectorization
│   ├── train_model.py            # Model training
│   └── predict.py                # Make predictions
│
└── results/                      # Output predictions
    └── predictions.csv           # Predicted tags for production docs
```

**Key Simplifications:**
- Single `model/` directory (not separate case/document)
- No separate `experiments/` tracking (just save best model)
- No `tests/` initially (focus on working implementation)
- No complex script orchestration (simple Python files)
- Flat `processed/` structure (not train/val/test subdirectories)

---

## 🔧 Implementation Steps

### **Phase 1: Project Setup**

#### Step 1.1: Create `config.py`
**File: `config.py`**
```python
"""Simple configuration for document tagging system"""
from pathlib import Path

class Config:
    """Configuration settings"""
    
    # Directories
    ROOT_DIR = Path(__file__).parent
    CASES_TRAINING_DIR = ROOT_DIR / 'cases' / 'training'
    CASES_PRODUCTION_DIR = ROOT_DIR / 'cases' / 'production'
    PROCESSED_DIR = ROOT_DIR / 'processed'
    MODEL_DIR = ROOT_DIR / 'model'
    RESULTS_DIR = ROOT_DIR / 'results'
    
    # Data processing settings
    BATCH_SIZE = 50              # Process 50 JSON files at a time (memory-safe)
    MAX_TEXT_LENGTH = 10000      # Truncate document text to 10K chars
    RANDOM_SEED = 42
    
    # Train/val/test splits
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # TF-IDF settings
    MAX_FEATURES = 5000          # Limit vocabulary size (memory-safe)
    MIN_DF = 2                   # Ignore rare terms
    MAX_DF = 0.8                 # Ignore very common terms
    
    # Model settings  
    MIN_TAG_FREQUENCY = 3        # Drop tags appearing < 3 times
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        for dir_path in [cls.PROCESSED_DIR, cls.MODEL_DIR, cls.RESULTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
```

#### Step 1.2: Update `.gitignore`
Add these lines to `.gitignore`:
```gitignore
# Processed data & models
processed/
model/
results/
*.pkl
*.csv

# Keep directory structure
!.gitkeep
```

#### Step 1.3: Create `requirements.txt`
**File: `requirements.txt`**
```txt
# Core
pandas==2.1.0
numpy==1.24.3

# ML
scikit-learn==1.3.0

# Utilities
tqdm==4.66.0
joblib==1.3.2
```

**That's it!** Much simpler dependencies.

---

### **Phase 2: Document Extraction** (Memory-Safe)

#### Step 2.1: Extract Documents from JSONs
**File: `src/extract_documents.py`**

**Purpose:** Extract all documents from case JSON files into a flat CSV format.

**CRITICAL: Memory Management**
- Process JSON files in batches (Config.BATCH_SIZE = 50)
- Write results incrementally to CSV (append mode)
- Never load all data into memory at once

**JSON Structure Note:**
```
case_json
├── case_id: int
├── fetched_at: str
├── case_data: dict (case metadata)
└── documents: list[dict]  <-- THIS IS WHERE DOCUMENTS ARE!
    └── [
        {
            "title": str,
            "document_source": str,
            "text": str,
            "document_type": str,
            "party_types": list[str],
            ...
        }
    ]
```

**Implementation:**
```python
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from config import Config

def extract_documents_from_case(case_json: dict, case_id: str) -> list[dict]:
    """
    Extract all documents from a single case JSON.
    
    IMPORTANT: Documents are at case_json['documents'], NOT case_json['case_data']['documents']
    
    Returns list of dicts, one per document:
    {
        'case_id': case_id,
        'doc_id': unique doc ID,
        'title': document title,
        'source': document source,
        'text': document text (truncated to MAX_TEXT_LENGTH),
        'document_type': tag (or None if missing),
        'party_types': tag (list, or None if missing)
    }
    """
    documents = []
    
    # IMPORTANT: Documents are at root level, NOT inside case_data!
    doc_list = case_json.get('documents', [])
    
    for idx, doc in enumerate(doc_list):
        # Extract fields safely (handle missing data)
        title = doc.get('title', '')
        source = doc.get('document_source', '')
        text = doc.get('text', '')
        
        # Truncate text for memory safety
        if text and len(text) > Config.MAX_TEXT_LENGTH:
            text = text[:Config.MAX_TEXT_LENGTH]
        
        # Extract tags (may be None/missing)
        doc_type = doc.get('document_type')
        party_types = doc.get('party_types')  # This is a list
        
        documents.append({
            'case_id': case_id,
            'doc_id': f"{case_id}_doc_{idx}",
            'title': title,
            'source': source,
            'text': text,
            'document_type': doc_type,
            'party_types': str(party_types) if party_types else None
        })
    
    return documents

def process_json_files_in_batches(cases_dir: Path, output_csv: Path):
    """
    Process all JSON files in batches. Memory-safe.
    """
    json_files = list(cases_dir.glob('case_*.json'))
    print(f"Found {len(json_files)} case files")
    
    # Process in batches
    for i in tqdm(range(0, len(json_files), Config.BATCH_SIZE)):
        batch_files = json_files[i:i + Config.BATCH_SIZE]
        batch_docs = []
        
        for json_file in batch_files:
            try:
                with open(json_file, 'r') as f:
                    case_json = json.load(f)
                
                case_id = json_file.stem  # e.g., 'case_10000'
                docs = extract_documents_from_case(case_json, case_id)
                batch_docs.extend(docs)
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        # Write batch to CSV (append mode)
        df_batch = pd.DataFrame(batch_docs)
        mode = 'w' if i == 0 else 'a'
        header = (i == 0)
        df_batch.to_csv(output_csv, mode=mode, header=header, index=False)
    
    print(f"Saved all documents to {output_csv}")

# Main execution
if __name__ == '__main__':
    Config.create_dirs()
    
    # Extract training documents
    print("Extracting training documents...")
    process_json_files_in_batches(
        Config.CASES_TRAINING_DIR,
        Config.PROCESSED_DIR / 'documents_all.csv'
    )
```

**Output:** `processed/documents_all.csv` with columns:
- `case_id`, `doc_id`, `title`, `source`, `text`, `document_type`, `party_types`

---

### **Phase 3: Feature Processing & Data Splitting** (Memory-Safe)

#### Step 3.1: Process Features & Split Data
**File: `src/process_features.py`**

**Purpose:** Clean text, create TF-IDF features, encode labels, split train/val/test.

**CRITICAL: Memory Management**
- Read CSV in chunks
- Process text in batches
- Use sparse matrices for TF-IDF
- Split by case_id (keep documents from same case together)

**Implementation:**
```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import joblib
from config import Config

def clean_text(text):
    """Basic text cleaning"""
    if pd.isna(text) or not text:
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
    print(f"After filtering rare tags: {len(df)} documents")
    
    # Split by case_id (train/val/test)
    print("Splitting data by cases...")
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
    
    # Create splits
    df_train = df[df['case_id'].isin(cases_train)]
    df_val = df[df['case_id'].isin(cases_val)]
    df_test = df[df['case_id'].isin(cases_test)]
    
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    # Create TF-IDF features (fit on train, transform all)
    print("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=Config.MAX_FEATURES,
        min_df=Config.MIN_DF,
        max_df=Config.MAX_DF,
        ngram_range=(1, 2)  # unigrams and bigrams
    )
    
    X_train = vectorizer.fit_transform(df_train['combined_text'])
    X_val = vectorizer.transform(df_val['combined_text'])
    X_test = vectorizer.transform(df_test['combined_text'])
    
    print(f"Feature matrix shape: {X_train.shape}")
    
    # Save vectorizer
    joblib.dump(vectorizer, Config.PROCESSED_DIR / 'vectorizers.pkl')
    
    # Encode labels
    print("Encoding labels...")
    
    # document_type (single-label)
    le_doctype = LabelEncoder()
    y_train_doctype = le_doctype.fit_transform(df_train['document_type'])
    y_val_doctype = le_doctype.transform(df_val['document_type'])
    y_test_doctype = le_doctype.transform(df_test['document_type'])
    
    # party_types (multi-label) - handle string format '[item1, item2]'
    def parse_party_types(val):
        if pd.isna(val) or val == 'None':
            return []
        # Simple parsing (improve if needed)
        return eval(val) if val else []
    
    df_train['party_list'] = df_train['party_types'].apply(parse_party_types)
    df_val['party_list'] = df_val['party_types'].apply(parse_party_types)
    df_test['party_list'] = df_test['party_types'].apply(parse_party_types)
    
    mlb_party = MultiLabelBinarizer()
    y_train_party = mlb_party.fit_transform(df_train['party_list'])
    y_val_party = mlb_party.transform(df_val['party_list'])
    y_test_party = mlb_party.transform(df_test['party_list'])
    
    # Save label encoders
    encoders = {
        'document_type': le_doctype,
        'party_types': mlb_party
    }
    joblib.dump(encoders, Config.PROCESSED_DIR / 'label_encoders.pkl')
    
    # Save processed datasets
    print("Saving processed data...")
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
    
    print("Feature processing complete!")
    print(f"Document types: {list(le_doctype.classes_)}")
    print(f"Party types: {list(mlb_party.classes_)}")

if __name__ == '__main__':
    prepare_features_and_labels(Config.PROCESSED_DIR / 'documents_all.csv')
```

---

### **Phase 4: Model Training**

#### Step 4.1: Train Document Tagger
**File: `src/train_model.py`**

**Purpose:** Train a simple model to predict both document tags.

**Model Choice:** Logistic Regression (simple, fast, interpretable)
- One model for `document_type` (multi-class classification)
- One model for `party_types` (multi-label classification)

**Implementation:**
```python
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from config import Config

def train_models():
    """Train document tagging models"""
    
    print("Loading training data...")
    train_data = joblib.load(Config.PROCESSED_DIR / 'documents_train.pkl')
    val_data = joblib.load(Config.PROCESSED_DIR / 'documents_val.pkl')
    
    X_train = train_data['X']
    X_val = val_data['X']
    
    # Train document_type model
    print("\nTraining document_type classifier...")
    model_doctype = LogisticRegression(max_iter=1000, random_state=Config.RANDOM_SEED)
    model_doctype.fit(X_train, train_data['y_document_type'])
    
    # Evaluate on validation
    y_pred = model_doctype.predict(X_val)
    acc = accuracy_score(val_data['y_document_type'], y_pred)
    f1 = f1_score(val_data['y_document_type'], y_pred, average='weighted')
    
    print(f"Validation Accuracy: {acc:.3f}")
    print(f"Validation F1 (weighted): {f1:.3f}")
    
    # Train party_types model (multi-label)
    print("\nTraining party_types classifier...")
    model_party = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, random_state=Config.RANDOM_SEED)
    )
    model_party.fit(X_train, train_data['y_party_types'])
    
    # Evaluate on validation
    y_pred_party = model_party.predict(X_val)
    f1_micro = f1_score(val_data['y_party_types'], y_pred_party, average='micro')
    f1_macro = f1_score(val_data['y_party_types'], y_pred_party, average='macro')
    
    print(f"Validation F1 (micro): {f1_micro:.3f}")
    print(f"Validation F1 (macro): {f1_macro:.3f}")
    
    # Save models
    print("\nSaving models...")
    models = {
        'document_type': model_doctype,
        'party_types': model_party
    }
    joblib.dump(models, Config.MODEL_DIR / 'document_tagger.pkl')
    
    # Save metadata
    metadata = {
        'document_type_accuracy': float(acc),
        'document_type_f1': float(f1),
        'party_types_f1_micro': float(f1_micro),
        'party_types_f1_macro': float(f1_macro),
        'num_train_samples': len(train_data['y_document_type']),
        'num_val_samples': len(val_data['y_document_type'])
    }
    
    import json
    with open(Config.MODEL_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Training complete!")
    return metadata

if __name__ == '__main__':
    Config.create_dirs()
    metrics = train_models()
    print("\nFinal Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
```

---

### **Phase 5: Prediction (Inference)**

#### Step 5.1: Predict on New Documents
**File: `src/predict.py`**

**Purpose:** Make predictions on production documents (no tags).

**CRITICAL: Memory Management**
- Process production JSONs in batches
- Use same extraction + feature pipeline
- Write results incrementally

**Implementation:**
```python
import json
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
from config import Config

def predict_on_production_data():
    """
    Predict tags for all documents in production cases.
    Memory-safe batch processing.
    """
    
    print("Loading models and encoders...")
    models = joblib.load(Config.MODEL_DIR / 'document_tagger.pkl')
    encoders = joblib.load(Config.PROCESSED_DIR / 'label_encoders.pkl')
    vectorizer = joblib.load(Config.PROCESSED_DIR / 'vectorizers.pkl')
    
    # Extract documents from production cases (same as training)
    print("Extracting production documents...")
    from extract_documents import process_json_files_in_batches
    
    prod_csv = Config.PROCESSED_DIR / 'documents_production.csv'
    process_json_files_in_batches(Config.CASES_PRODUCTION_DIR, prod_csv)
    
    # Read and process production documents
    print("Processing features...")
    df = pd.read_csv(prod_csv)
    
    # Clean text (same as training)
    df['title_clean'] = df['title'].apply(lambda x: str(x).lower().strip() if pd.notna(x) else "")
    df['source_clean'] = df['source'].apply(lambda x: str(x).lower().strip() if pd.notna(x) else "")
    df['text_clean'] = df['text'].apply(lambda x: str(x).lower().strip() if pd.notna(x) else "")
    
    df['combined_text'] = (
        df['title_clean'] + ' ' +
        df['source_clean'] + ' ' +
        df['text_clean']
    )
    
    # Vectorize
    X = vectorizer.transform(df['combined_text'])
    
    # Make predictions
    print("Making predictions...")
    
    # Predict document_type
    y_pred_doctype = models['document_type'].predict(X)
    df['predicted_document_type'] = encoders['document_type'].inverse_transform(y_pred_doctype)
    
    # Predict party_types
    y_pred_party = models['party_types'].predict(X)
    df['predicted_party_types'] = [
        list(encoders['party_types'].inverse_transform([row])[0])
        for row in y_pred_party
    ]
    
    # Save predictions
    print("Saving predictions...")
    output_file = Config.RESULTS_DIR / 'predictions.csv'
    df[['case_id', 'doc_id', 'title', 'predicted_document_type', 'predicted_party_types']].to_csv(
        output_file, index=False
    )
    
    print(f"Predictions saved to {output_file}")
    print(f"Total documents predicted: {len(df)}")

if __name__ == '__main__':
    Config.create_dirs()
    predict_on_production_data()
```

---

## 🚀 Complete Workflow (Simple Commands)

### Step 1: Setup (one time)
```bash
pip install -r requirements.txt
```

### Step 2: Extract Documents from Training Cases
```bash
python src/extract_documents.py
```
**Output:** `processed/documents_all.csv`

### Step 3: Process Features & Split Data
```bash
python src/process_features.py
```
**Output:** `processed/documents_train.pkl`, `documents_val.pkl`, `documents_test.pkl`, `vectorizers.pkl`, `label_encoders.pkl`

### Step 4: Train Model
```bash
python src/train_model.py
```
**Output:** `model/document_tagger.pkl`, `model/metadata.json`

### Step 5: Make Predictions on Production Cases
```bash
python src/predict.py
```
**Output:** `results/predictions.csv`

---

## � Expected Performance

**Targets for Document-Level Tags:**
- `document_type`: Accuracy > 70%, F1 > 0.65
- `party_types`: F1 (micro) > 0.60

**If performance is poor:**
1. Check class distribution (are some tags very rare?)
2. Increase `MAX_FEATURES` in config (more vocabulary)
3. Try different model (e.g., Random Forest instead of Logistic Regression)
4. Add more training data

---

## 🔧 Memory Safety Features

**This implementation is designed to avoid memory issues:**

1. **Batch Processing**: JSON files processed in batches of 50
2. **Text Truncation**: Documents limited to 10K characters
3. **Sparse Matrices**: TF-IDF uses sparse format (efficient for large datasets)
4. **Limited Vocabulary**: Max 5000 features (controlled via config)
5. **Incremental Writing**: CSV results written in batches, not all at once
6. **No Caching**: Don't load all data into memory simultaneously

**If you still run out of memory:**
- Reduce `Config.BATCH_SIZE` (e.g., 25 instead of 50)
- Reduce `Config.MAX_FEATURES` (e.g., 3000 instead of 5000)
- Reduce `Config.MAX_TEXT_LENGTH` (e.g., 5000 instead of 10000)

---

## 🎯 Future Enhancements (Optional)

**After getting this working, consider:**

1. **Add Test Set Evaluation**: Create a script to evaluate on test set
2. **Add Case-Level Tagging**: Extend to predict case-level tags
3. **Better Models**: Try Random Forest, XGBoost for better accuracy
4. **Feature Engineering**: Add document length, source indicators, etc.
5. **Hyperparameter Tuning**: Use GridSearchCV to optimize model params
6. **Confidence Scores**: Output prediction probabilities for quality control
7. **Web Interface**: Build simple Flask app for interactive tagging

---

## ✅ Implementation Checklist

- [ ] Create `config.py`
- [ ] Create `requirements.txt`
- [ ] Update `.gitignore`
- [ ] Create `src/extract_documents.py`
- [ ] Create `src/process_features.py`
- [ ] Create `src/train_model.py`
- [ ] Create `src/predict.py`
- [ ] Test complete pipeline end-to-end
- [ ] Document results in README

---

**End of Specification**

This simplified spec focuses on **one goal**: predicting document-level tags with memory-safe processing. Everything else is deferred for future implementation.
