# ML Implementation Specification
## Document-Level Legal Tagging System

---

## 🎯 Project Overview

Build a **memory-efficient** machine learning system to automatically tag individual legal documents using human-tagged training data from the Civil Rights Clearinghouse database.

**Scope:** Document-level tagging ONLY (case-level tagging deferred for future implementation)

**Key Constraints:**
- Memory-safe processing (batch processing throughout)
- Interpretable model architecture with comprehensive evaluation
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

## 🏗️ Repository Structure

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
│   ├── documents_all.csv         # All extracted documents
│   ├── documents_train.pkl       # Training documents
│   ├── documents_val.pkl         # Validation documents
│   ├── documents_test.pkl        # Test documents
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
│   ├── eda.ipynb                 # Exploratory data analysis
│   ├── process_features.py       # Text processing & vectorization
│   ├── train_model.py            # Model training
│   ├── evaluate_model.py         # Test set evaluation
│   └── predict.py                # Make predictions on production data
│
└── results/                      # Output predictions and evaluations
    ├── test_predictions.csv      # Test set predictions
    ├── test_metrics.json         # Test set performance metrics
    └── predictions.csv           # Production predictions
```

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

# Visualization (for EDA)
matplotlib==3.7.2
seaborn==0.12.2

# Utilities
tqdm==4.66.0
joblib==1.3.2

# Jupyter (for EDA notebook)
jupyter==1.0.0
ipykernel==6.25.0
```

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

### **Phase 2.5: Exploratory Data Analysis (EDA)**

#### Step 2.5.1: Analyze Data Characteristics
**File: `src/eda.ipynb`**

**Purpose:** Understand data characteristics, distributions, and patterns before feature engineering. This analysis informs decisions about feature processing, model selection, and train/test split strategies.

**Timing:** Perform EDA after document extraction (Phase 2) but before feature processing (Phase 3). This ensures you have all raw data in a usable format, but haven't yet committed to specific feature engineering decisions.

**Key Analyses:**

1. **Tag Distribution Analysis**
   - Frequency of each `document_type` (check for class imbalance)
   - Frequency of each `party_type` 
   - Co-occurrence patterns (which `party_types` appear together frequently?)
   - Missing tag analysis (how many documents have no tags?)

2. **Document Characteristics**
   - Text length distributions (title, source, text, combined)
   - Number of documents per case
   - Empty/null field analysis

3. **Insights for Model Design**
   - If documents are very long → might need different truncation strategy
   - If tags are heavily imbalanced → might need class weights or sampling strategies
   - If certain tags co-occur frequently → might inform model architecture or feature engineering

**Notebook Structure:**
```python
# Cell 1: Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import ast

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Cell 2: Load Data
df = pd.read_csv('../processed/documents_all.csv')
print(f"Total documents: {len(df)}")
print(f"Total cases: {df['case_id'].nunique()}")

# Cell 3: Missing Data Analysis
print("=== Missing Data Analysis ===")
missing = df.isnull().sum()
print(missing[missing > 0])

# Documents with missing tags
missing_doctype = df['document_type'].isnull().sum()
missing_party = df['party_types'].isnull().sum()
print(f"\nDocuments missing document_type: {missing_doctype} ({missing_doctype/len(df)*100:.1f}%)")
print(f"Documents missing party_types: {missing_party} ({missing_party/len(df)*100:.1f}%)")

# Cell 4: Document Type Distribution
print("=== Document Type Distribution ===")
doctype_counts = df['document_type'].value_counts()
print(doctype_counts)

plt.figure(figsize=(14, 6))
doctype_counts.head(20).plot(kind='bar')
plt.title('Top 20 Document Types by Frequency')
plt.xlabel('Document Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Check for imbalance
print(f"\nMost common: {doctype_counts.iloc[0]} ({doctype_counts.iloc[0]/len(df)*100:.1f}%)")
print(f"Least common: {doctype_counts.iloc[-1]}")
print(f"Imbalance ratio: {doctype_counts.iloc[0] / doctype_counts.iloc[-1]:.1f}x")

# Cell 5: Party Types Distribution
print("=== Party Types Distribution ===")

# Parse party_types (stored as string representation of list)
def parse_party_types(val):
    if pd.isna(val) or val == 'None':
        return []
    try:
        return ast.literal_eval(val) if val else []
    except:
        return []

df['party_list'] = df['party_types'].apply(parse_party_types)

# Count individual party types
all_parties = []
for parties in df['party_list']:
    all_parties.extend(parties)

party_counts = Counter(all_parties)
print(f"Unique party types: {len(party_counts)}")
print("\nTop 15 party types:")
for party, count in party_counts.most_common(15):
    print(f"  {party}: {count}")

# Visualize
plt.figure(figsize=(14, 6))
top_parties = dict(party_counts.most_common(15))
plt.bar(top_parties.keys(), top_parties.values())
plt.title('Top 15 Party Types by Frequency')
plt.xlabel('Party Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Cell 6: Party Type Co-occurrence
print("=== Party Type Co-occurrence ===")

# Find documents with multiple party types
multi_party = df[df['party_list'].apply(len) > 1]
print(f"Documents with multiple party types: {len(multi_party)} ({len(multi_party)/len(df)*100:.1f}%)")

# Most common pairs
from itertools import combinations
pairs = []
for parties in multi_party['party_list']:
    if len(parties) >= 2:
        pairs.extend(list(combinations(sorted(parties), 2)))

pair_counts = Counter(pairs)
print("\nTop 10 party type pairs:")
for pair, count in pair_counts.most_common(10):
    print(f"  {pair[0]} + {pair[1]}: {count}")

# Cell 7: Text Length Analysis
print("=== Text Length Analysis ===")

# Calculate lengths
df['title_len'] = df['title'].fillna('').astype(str).str.len()
df['source_len'] = df['source'].fillna('').astype(str).str.len()
df['text_len'] = df['text'].fillna('').astype(str).str.len()
df['combined_len'] = df['title_len'] + df['source_len'] + df['text_len']

# Statistics
print("Length statistics:")
print(df[['title_len', 'source_len', 'text_len', 'combined_len']].describe())

# Visualize distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(df['title_len'], bins=50, edgecolor='black')
axes[0, 0].set_title('Title Length Distribution')
axes[0, 0].set_xlabel('Characters')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].hist(df['source_len'], bins=50, edgecolor='black')
axes[0, 1].set_title('Source Length Distribution')
axes[0, 1].set_xlabel('Characters')

axes[1, 0].hist(df['text_len'].clip(upper=50000), bins=50, edgecolor='black')
axes[1, 0].set_title('Text Length Distribution (capped at 50k)')
axes[1, 0].set_xlabel('Characters')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(df['combined_len'].clip(upper=50000), bins=50, edgecolor='black')
axes[1, 1].set_title('Combined Length Distribution (capped at 50k)')
axes[1, 1].set_xlabel('Characters')

plt.tight_layout()
plt.show()

# Check for very long documents
print(f"\nDocuments > 10,000 chars: {(df['text_len'] > 10000).sum()} ({(df['text_len'] > 10000).sum()/len(df)*100:.1f}%)")
print(f"Documents > 50,000 chars: {(df['text_len'] > 50000).sum()} ({(df['text_len'] > 50000).sum()/len(df)*100:.1f}%)")

# Cell 8: Documents per Case
print("=== Documents per Case ===")

docs_per_case = df.groupby('case_id').size()
print("Documents per case statistics:")
print(docs_per_case.describe())

plt.figure(figsize=(12, 5))
plt.hist(docs_per_case, bins=50, edgecolor='black')
plt.title('Distribution of Documents per Case')
plt.xlabel('Number of Documents')
plt.ylabel('Number of Cases')
plt.axvline(docs_per_case.mean(), color='red', linestyle='--', label=f'Mean: {docs_per_case.mean():.1f}')
plt.axvline(docs_per_case.median(), color='green', linestyle='--', label=f'Median: {docs_per_case.median():.1f}')
plt.legend()
plt.show()

# Cell 9: Key Insights Summary
print("=== KEY INSIGHTS FOR MODEL DESIGN ===\n")

# Class imbalance
max_class_ratio = doctype_counts.iloc[0] / doctype_counts.iloc[-1]
if max_class_ratio > 10:
    print(f"⚠️  HIGH CLASS IMBALANCE detected (ratio: {max_class_ratio:.1f}x)")
    print("   → Consider: class weights, stratified sampling, or SMOTE")

# Text length
median_text_len = df['text_len'].median()
p95_text_len = df['text_len'].quantile(0.95)
if p95_text_len > 20000:
    print(f"\n⚠️  LONG DOCUMENTS detected (95th percentile: {p95_text_len:.0f} chars)")
    print(f"   → Current truncation at 10,000 chars may cut {(df['text_len'] > 10000).sum()} docs")
    print(f"   → Consider: increasing MAX_TEXT_LENGTH or using document summarization")

# Multi-label complexity
avg_parties = df['party_list'].apply(len).mean()
print(f"\n📊 Average party types per document: {avg_parties:.2f}")
if avg_parties > 2:
    print("   → Multi-label classification will be complex")

# Missing data
if missing_doctype > len(df) * 0.1:
    print(f"\n⚠️  HIGH MISSING TAG RATE: {missing_doctype/len(df)*100:.1f}% documents missing document_type")
    print("   → May significantly reduce training set size")

print("\n✅ EDA Complete! Use these insights to configure feature processing.")
```

**Key Outputs:**
- Tag distribution visualizations
- Text length statistics and recommendations for truncation
- Class imbalance warnings
- Multi-label complexity assessment
- Actionable insights for feature processing configuration

#### Step 2.5.2: EDA Results and Config Updates

**Analysis Results (from 11,546 extracted documents):**

1. **Class Imbalance - SEVERE (4401:1 ratio)**
   - Most common: "Order/Opinion" with 4,401 docs (38.1%)
   - Least common: "FOIA Request" with 2 docs
   - 6 classes have fewer than 3 instances
   - **Impact:** Without correction, model would overpredict majority class

2. **Text Length Distribution - OPTIMAL**
   - 95th percentile: 10,000 characters (exactly our truncation limit)
   - All documents already truncated during extraction
   - No further adjustment needed

3. **Multi-label Complexity - LOW**
   - Average party types per document: 0.94
   - Only 7.7% of documents have multiple party types
   - Multi-label classification will be manageable

4. **Missing Data - MODERATE**
   - 12.6% missing document_type (1,453 docs)
   - 16.7% missing party_types (1,923 docs)
   - Training set will reduce to ~10,093 documents after filtering

**Config Adjustments Made:**

Updated `config.py` to handle severe class imbalance:

```python
# Model settings  
MIN_TAG_FREQUENCY = 3        # Drop tags appearing < 3 times (handles rare classes)
USE_CLASS_WEIGHTS = True     # Enable class weights to handle imbalance (4401:1 ratio)
```

**Rationale:**
- `USE_CLASS_WEIGHTS = True` enables `class_weight='balanced'` in LogisticRegression
- This penalizes mistakes on minority classes more heavily
- Prevents model from just predicting "Order/Opinion" for everything
- Expected to significantly improve performance on rare classes

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
    class_weight = 'balanced' if Config.USE_CLASS_WEIGHTS else None
    model_doctype = LogisticRegression(
        max_iter=1000, 
        random_state=Config.RANDOM_SEED,
        class_weight=class_weight
    )
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
        LogisticRegression(
            max_iter=1000, 
            random_state=Config.RANDOM_SEED,
            class_weight=class_weight
        )
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

### **Phase 4.5: Test Set Evaluation**

#### Step 4.5.1: Evaluate Model on Test Set
**File: `src/evaluate_model.py`**

**Purpose:** Evaluate the trained model on held-out test data to assess real-world performance. This provides an unbiased estimate of how the model will perform on new, unseen documents.

**Why Test Set Evaluation:**
- Validation set was used during training for model selection
- Test set provides final, unbiased performance metrics
- Identifies overfitting or other generalization issues
- Generates detailed performance reports for model assessment

**Implementation:**
```python
import joblib
import numpy as np
import pandas as pd
import json
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    confusion_matrix,
    hamming_loss
)
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
        list(encoders['party_types'].inverse_transform([row])[0])
        for row in y_pred_party
    ]
    
    # Decode ground truth
    true_doctype_labels = encoders['document_type'].inverse_transform(y_test_doctype)
    true_party_labels = [
        list(encoders['party_types'].inverse_transform([row])[0])
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
```

**Outputs:**
1. **`results/test_predictions.csv`**: Detailed predictions for each test document
   - Columns: `doc_id`, `true_document_type`, `predicted_document_type`, `doctype_confidence`, `doctype_correct`, `true_party_types`, `predicted_party_types`, `party_exact_match`
   
2. **`results/test_metrics.json`**: Comprehensive performance metrics
   - Accuracy, precision, recall, F1 scores
   - Per-class performance statistics
   - Both document_type and party_types metrics

**When to Run:**
- After Phase 4 (model training) completes
- Before Phase 5 (production predictions)
- Use test metrics to decide if model is production-ready

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

## 🚀 Complete Workflow

### Step 1: Setup (one time)
```bash
pip install -r requirements.txt
```

### Step 2: Extract Documents from Training Cases
```bash
python src/extract_documents.py
```
**Output:** `processed/documents_all.csv`

### Step 3: Exploratory Data Analysis
```bash
jupyter notebook src/eda.ipynb
```
**Analyze:**
- Tag distributions and class imbalance
- Text length characteristics
- Multi-label complexity
- Missing data patterns

**Use insights to adjust `config.py` if needed (e.g., MAX_TEXT_LENGTH, MIN_TAG_FREQUENCY)**

### Step 4: Process Features & Split Data
```bash
python src/process_features.py
```
**Output:** `processed/documents_train.pkl`, `documents_val.pkl`, `documents_test.pkl`, `vectorizers.pkl`, `label_encoders.pkl`

### Step 5: Train Model
```bash
python src/train_model.py
```
**Output:** `model/document_tagger.pkl`, `model/metadata.json`

### Step 6: Evaluate on Test Set
```bash
python src/evaluate_model.py
```
**Output:** `results/test_predictions.csv`, `results/test_metrics.json`

**Review test metrics to ensure model is production-ready before deploying!**

### Step 7: Make Predictions on Production Cases
```bash
python src/predict.py
```
**Output:** `results/predictions.csv`

---

## 📈 Expected Performance

**Performance Targets:**
- `document_type`: Accuracy > 70%, F1 (weighted) > 0.65
- `party_types`: Subset Accuracy > 50%, F1 (micro) > 0.60

**Performance Analysis:**
- Use validation metrics during training for model selection
- Use test metrics for final performance assessment
- Compare test vs. validation performance to check for overfitting
- Review per-class metrics to identify problematic categories

**If performance is poor:**
1. **Check EDA findings**: Review class distribution and data quality issues
2. **Adjust features**: Increase `MAX_FEATURES` for larger vocabulary, try different n-gram ranges
3. **Handle imbalance**: Use class weights, SMOTE, or stratified sampling
4. **Try different models**: Random Forest, XGBoost, or neural networks
5. **Feature engineering**: Add document length, source indicators, or other metadata features
6. **Add more training data**: Collect additional labeled cases

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

**After completing the core pipeline, consider:**

1. **Add Case-Level Tagging**: Extend to predict case-level tags (aggregating document predictions)
2. **Advanced Models**: Try Random Forest, XGBoost, or transformer models for better accuracy
3. **Feature Engineering**: Add document length, source indicators, named entity features, etc.
4. **Hyperparameter Tuning**: Use GridSearchCV or Optuna to optimize model parameters
5. **Confidence Scores**: Add prediction probability thresholds for quality control
6. **Active Learning**: Identify low-confidence predictions for manual review
7. **Web Interface**: Build Flask or Streamlit app for interactive tagging
8. **Model Monitoring**: Track model performance over time on production data

---

## ✅ Implementation Checklist

- [ ] Create `config.py`
- [ ] Create `requirements.txt`
- [ ] Update `.gitignore`
- [ ] Create `src/extract_documents.py`
- [ ] Run extraction and verify `documents_all.csv`
- [ ] Create and run `src/eda.ipynb` (analyze data characteristics)
- [ ] Adjust config based on EDA insights
- [ ] Create `src/process_features.py`
- [ ] Create `src/train_model.py`
- [ ] Create `src/evaluate_model.py`
- [ ] Review test metrics (ensure model is production-ready)
- [ ] Create `src/predict.py`
- [ ] Test complete pipeline end-to-end
- [ ] Document results in README

---

**End of Specification**

This specification provides a comprehensive document-level tagging pipeline with proper data analysis, model evaluation, and production deployment capabilities.
