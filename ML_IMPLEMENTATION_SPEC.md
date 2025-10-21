# ML Implementation Specification
## Legal Case & Document Tagging System

---

## 🎯 Project Overview

Build a machine learning system to automatically tag legal cases and their documents using human-tagged training data from the Civil Rights Clearinghouse database. The system will predict tags at two levels:
1. **Case-level tags** (12 different tag types)
2. **Document-level tags** (2 different tag types)

---

## 📊 Data & Tags

**Summary:** 12 case-level tags, 2 document-level tags

### ⚠️ Available Input Data (CRITICAL)

**For Case-Level Tag Prediction:**
- ✅ **Case name** (e.g., "Price v. Jefferson County")
- ✅ **All document texts** (combined from all documents in the case)
- ✅ **All document titles** (combined from all documents)
- ✅ **All document sources** (e.g., "PACER", "Court website")
- ✅ **Docket entries** (court filing information)
- ❌ **Case summary** - NOT AVAILABLE (human-created after the fact)

**For Document-Level Tag Prediction:**
- ✅ **Document title** (e.g., "Order", "Docket [PACER]")
- ✅ **Document source** (e.g., "PACER [Public Access to Court Electronic Records]")
- ✅ **Document text** (the actual document content)
- ❌ Cannot use case-level information or other documents

### Case-Level Tags (12 types)
1. `state` - Geographic location (JSON field: "state")
2. `case_types` - Type(s) of case (list)
3. `case_ongoing` - Boolean: is case still active
4. `plaintiff_description` - Text description of plaintiff
5. `plaintiff_type` - Type(s) of plaintiff (list) (JSON field: "plaintiff_type")
6. `case_defendants` - List of defendant info (JSON field: "case_defendants")
7. `defendant_type` - Type(s) of defendants (list) (JSON field: "defendant_type")
8. `causes` - Legal causes (list) (JSON field: "causes")
9. `constitutional_clause` - Constitutional provisions (list) (JSON field: "constitutional_clause")
10. `prevailing_party` - Who won the case
11. `relief_natures` - Type of remedy granted (JSON field: "relief_natures")
12. `relief_sources` - Origin of remedy (JSON field: "relief_sources")

### Document-Level Tags (2 types)
1. `document_type` - Type of document
2. `party_types` - Parties involved (list)

---

## 🏗️ Repository Structure (Clean & Organized)

```
si485-clearinghouseml/
├── .env                          # API keys & secrets (NOT in git)
├── .gitignore                    # Already configured
├── requirements.txt              # Python dependencies
├── README.md                     # Main project documentation
├── config.py                     # Configuration management
│
├── data/                         # All data-related files
│   ├── raw/                      # Original CSV files
│   │   └── training_cases.csv
│   ├── cases/                    # Fetched case JSON files
│   │   ├── training/             # Human-tagged cases
│   │   └── production/           # Cases to tag
│   └── processed/                # Processed data for ML
│       ├── train/
│       ├── val/
│       └── test/
│
├── src/                          # All source code
│   ├── __init__.py
│   ├── ingestion/                # Data collection
│   │   ├── __init__.py
│   │   ├── data_ingestion.py    # API fetcher (moved from root)
│   │   └── data_utils.py        # Data exploration (moved from root)
│   │
│   ├── preprocessing/            # Data preparation for ML
│   │   ├── __init__.py
│   │   ├── tag_extractor.py     # Extract tags from JSON
│   │   ├── text_processor.py    # Text cleaning & vectorization
│   │   ├── dataset_builder.py   # Build train/val/test sets
│   │   └── label_encoder.py     # Encode tags for ML
│   │
│   ├── models/                   # ML model definitions
│   │   ├── __init__.py
│   │   ├── case_tagger.py       # Case-level model
│   │   ├── document_tagger.py   # Document-level model
│   │   └── model_utils.py       # Shared utilities
│   │
│   ├── training/                 # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py           # Training orchestrator
│   │   └── evaluator.py         # Model evaluation
│   │
│   └── inference/                # Prediction on new data
│       ├── __init__.py
│       └── predictor.py         # Make predictions
│
├── models/                       # Saved trained models
│   ├── case_tagger/
│   │   ├── model.pkl
│   │   └── metadata.json
│   └── document_tagger/
│       ├── model.pkl
│       └── metadata.json
│
├── experiments/                  # Training experiments & logs
│   └── experiment_YYYYMMDD_HHMMSS/
│       ├── config.json
│       ├── metrics.json
│       ├── training.log
│       └── plots/
│
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_experiments.ipynb
│   └── 03_results_analysis.ipynb
│
├── scripts/                      # Executable scripts
│   ├── fetch_data.py            # Run data ingestion
│   ├── prepare_data.py          # Run preprocessing
│   ├── train_models.py          # Train both models
│   ├── evaluate_models.py       # Evaluate performance
│   └── predict.py               # Make predictions on new data
│
└── tests/                        # Unit tests
    ├── __init__.py
    ├── test_preprocessing.py
    ├── test_models.py
    └── test_inference.py
```

---

## 🔧 Implementation Steps

### **Phase 1: Project Setup & Security** (Priority: CRITICAL)

#### Step 1.1: Create `.env` file for secrets
**File: `.env`**
```env
# API Configuration
CLEARINGHOUSE_API_TOKEN=11a7d2a0a5c673e0d4391cb578563eb43d629a49
CLEARINGHOUSE_API_BASE_URL=https://clearinghouse.net/api/v2

# Paths
DATA_DIR=data
MODELS_DIR=models
EXPERIMENTS_DIR=experiments
```

#### Step 1.2: Update `.gitignore` to ensure security
Add to `.gitignore`:
```gitignore
# Project-specific
.env
*.env
data/cases/
data/processed/
models/
experiments/
*.pkl
*.joblib
*.h5
*.pth

# Keep structure but ignore contents
!data/cases/.gitkeep
!data/processed/.gitkeep
!models/.gitkeep
!experiments/.gitkeep
```

#### Step 1.3: Create `config.py` for configuration management
**File: `config.py`**
```python
"""
Configuration management using environment variables.
Loads secrets from .env file safely.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Centralized configuration"""
    
    # API Configuration
    API_TOKEN = os.getenv('CLEARINGHOUSE_API_TOKEN')
    API_BASE_URL = os.getenv('CLEARINGHOUSE_API_BASE_URL', 'https://clearinghouse.net/api/v2')
    
    # Directory paths
    ROOT_DIR = Path(__file__).parent
    DATA_DIR = ROOT_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    CASES_DIR = DATA_DIR / 'cases'
    PROCESSED_DIR = DATA_DIR / 'processed'
    MODELS_DIR = ROOT_DIR / 'models'
    EXPERIMENTS_DIR = ROOT_DIR / 'experiments'
    
    # Data splits
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_SEED = 42
    
    # Model configuration
    MAX_TEXT_LENGTH = 5000  # Maximum characters for text features
    MIN_TAG_FREQUENCY = 2   # Minimum occurrences to keep a tag
    
    @classmethod
    def validate(cls):
        """Validate that all required config is present"""
        if not cls.API_TOKEN:
            raise ValueError("CLEARINGHOUSE_API_TOKEN not found in .env file")
        
        # Create directories if they don't exist
        for dir_path in [cls.DATA_DIR, cls.RAW_DATA_DIR, cls.CASES_DIR, 
                         cls.PROCESSED_DIR, cls.MODELS_DIR, cls.EXPERIMENTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        return True
```

#### Step 1.4: Create `requirements.txt`
**File: `requirements.txt`**
```txt
# Core dependencies
python-dotenv==1.0.0
requests==2.31.0
pandas==2.1.0
numpy==1.24.3

# ML frameworks
scikit-learn==1.3.0
xgboost==2.0.0
lightgbm==4.1.0

# Text processing
nltk==3.8.1
spacy==3.7.0

# Data validation
pydantic==2.5.0

# Visualization
matplotlib==3.8.0
seaborn==0.13.0

# Utilities
tqdm==4.66.0
joblib==1.3.2

# Development
pytest==7.4.0
black==23.10.0
flake8==6.1.0
jupyter==1.0.0
```

---

### **Phase 2: Data Preprocessing Pipeline**

#### Step 2.1: Tag Extraction
**File: `src/preprocessing/tag_extractor.py`**

**Purpose:** Extract relevant tags from the JSON case files into a structured format.

**Implementation Requirements:**
1. Load JSON files from `data/cases/training/`
2. For each case file:
   - Extract all 12 case-level tags from `case_data` field
   - Extract case name from `case_data['name']`
   - Extract all document texts and combine them (for case-level features)
   - Extract all document titles and combine them (for case-level features)
   - Extract all document sources and combine them (for case-level features)
   - Extract docket entries from `case_data['main_docket']['docket_entries']`
   - Extract individual documents and their 2 document-level tags (`document_type`, `party_types`)
   - **EXCLUDE** case summary - it will not be available in production
   - Handle missing/null values appropriately
3. Create two DataFrames:
   - `cases_df`: One row per case with:
     * All 12 case-level tags (labels)
     * Case name
     * Combined document text (all docs concatenated)
     * Combined document titles (all titles concatenated)
     * Combined document sources (all sources concatenated)
     * Docket text (all docket entries concatenated)
     * Case ID
   - `documents_df`: One row per document with:
     * 2 document-level tags (labels)
     * Document title (from `title` field)
     * Document source (from `document_source` field)
     * Document text (from `text` field)
     * Case ID
     * Document ID
4. Save to `data/processed/`:
   - `cases_raw.csv`
   - `documents_raw.csv`

**Key Functions:**
```python
def extract_case_tags(case_json: dict) -> dict:
    """Extract case-level tags from JSON"""

def extract_case_name(case_json: dict) -> str:
    """Extract case name"""
    
def combine_document_texts(case_json: dict) -> str:
    """Combine all document texts for a case (for case-level features)"""

def combine_document_titles(case_json: dict) -> str:
    """Combine all document titles for a case (for case-level features)"""
    
def combine_document_sources(case_json: dict) -> str:
    """Combine all document sources for a case (for case-level features)"""
    
def extract_docket_text(case_json: dict) -> str:
    """Extract and combine docket entry descriptions"""
    
def extract_document_tags(case_json: dict) -> list[dict]:
    """Extract document-level tags from JSON"""
    
def process_all_cases(cases_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process all cases and return dataframes"""
```

**Handle these special cases:**
- Tags that are lists (e.g., `case_types`, `causes`)
- Tags that are nested dicts (e.g., `constitutional_clause`)
- Missing/None values

#### Step 2.2: Text Processing
**File: `src/preprocessing/text_processor.py`**

**Purpose:** Clean and vectorize text data for ML models.

**IMPORTANT - Available Data:**
- **For Case-Level Predictions:** Use case name, all document texts/titles/sources (combined), and docket information. Do NOT use case summary (human-created).
- **For Document-Level Predictions:** Use document title, document source, and document text.

**Implementation Requirements:**
1. Text cleaning:
   - Remove special characters but keep legal terminology
   - Convert to lowercase
   - Remove extra whitespace
   - Handle None/empty text
2. Text truncation:
   - Limit to `MAX_TEXT_LENGTH` characters (config)
   - Use smart truncation (don't cut mid-word)
3. Feature extraction:
   - **For case-level model:**
     * TF-IDF vectorization for case names
     * TF-IDF vectorization for ALL document texts combined (concatenate all docs for a case)
     * TF-IDF vectorization for ALL document titles combined
     * TF-IDF vectorization for ALL document sources combined
     * TF-IDF vectorization for docket entries (if available)
     * N-grams (1-3) to capture legal phrases
   - **For document-level model:**
     * TF-IDF vectorization for document title
     * TF-IDF vectorization for document source
     * TF-IDF vectorization for document text
     * N-grams (1-3) to capture legal phrases
4. Save vectorizers for later use on production data

**Key Functions:**
```python
def clean_text(text: str) -> str:
    """Clean and normalize text"""

def combine_case_documents(documents: list[dict]) -> str:
    """Combine all document texts for a case into single text"""
    
def extract_docket_text(case_data: dict) -> str:
    """Extract text from docket entries"""
    
def create_text_features(texts: list[str], fit: bool = True) -> tuple[np.ndarray, object]:
    """Create TF-IDF features from text"""
    
def save_vectorizer(vectorizer: object, filepath: Path):
    """Save fitted vectorizer"""
    
def load_vectorizer(filepath: Path) -> object:
    """Load saved vectorizer"""
```

#### Step 2.3: Label Encoding
**File: `src/preprocessing/label_encoder.py`**

**Purpose:** Convert tags into ML-compatible format.

**Implementation Requirements:**
1. **For single-value tags** (e.g., `state`, `case_ongoing`):
   - Use LabelEncoder
   - Handle unseen values in production
2. **For multi-value tags** (e.g., `case_types`, `causes`):
   - Use MultiLabelBinarizer
   - Filter rare tags (< MIN_TAG_FREQUENCY occurrences)
3. **Handle missing values:**
   - Use special "UNKNOWN" category
   - Or use -1 for numeric encodings
4. Save all encoders for production use

**Key Functions:**
```python
class TagEncoder:
    """Handles encoding/decoding of all tag types"""
    
    def fit(self, data: pd.DataFrame):
        """Fit encoders on training data"""
    
    def transform(self, data: pd.DataFrame) -> dict:
        """Transform tags to ML format"""
    
    def inverse_transform(self, encoded: dict) -> dict:
        """Convert predictions back to original format"""
    
    def save(self, filepath: Path):
        """Save all encoders"""
    
    @classmethod
    def load(cls, filepath: Path):
        """Load saved encoders"""
```

#### Step 2.4: Dataset Builder
**File: `src/preprocessing/dataset_builder.py`**

**Purpose:** Create train/validation/test splits and final ML-ready datasets.

**Implementation Requirements:**
1. Split data by **case ID** (not by documents):
   - 70% train, 15% validation, 15% test
   - Use stratified split on key tags (e.g., `state`)
   - Ensure no case appears in multiple splits
2. Create feature matrices:
   - Combine text features with encoded metadata
   - Create separate datasets for case-level and document-level
3. Handle class imbalance:
   - Calculate class weights for rare tags
   - Document imbalance statistics
4. Save processed datasets:
   - `data/processed/train/`
   - `data/processed/val/`
   - `data/processed/test/`

**Key Functions:**
```python
def split_cases(case_ids: list, stratify_on: str = None) -> tuple:
    """Split case IDs into train/val/test"""
    
def build_case_dataset(cases_df: pd.DataFrame, case_ids: list) -> dict:
    """Build ML-ready case dataset"""
    
def build_document_dataset(docs_df: pd.DataFrame, case_ids: list) -> dict:
    """Build ML-ready document dataset"""
    
def calculate_class_weights(labels: np.ndarray) -> dict:
    """Calculate weights for imbalanced classes"""
```

---

### **Phase 3: Model Development**

#### Step 3.1: Model Architecture Selection

**For Case-Level Tagging:**
- **Model Type:** Multi-output classifier (one model predicting all 12 tags)
- **Base Model:** XGBoost or LightGBM (handles mixed data types well)
- **Alternative:** Random Forest (simpler, good baseline)

**For Document-Level Tagging:**
- **Model Type:** Multi-output classifier (predicting 2 tags)
- **Base Model:** XGBoost or LightGBM
- **Alternative:** Logistic Regression (simpler for fewer tags)

**Rationale:**
- Tree-based models handle mixed features (text + categorical + numeric) well
- Multi-output approach is simpler than separate models per tag
- XGBoost/LightGBM handle missing values natively
- Fast training and inference

#### Step 3.2: Case-Level Model
**File: `src/models/case_tagger.py`**

**Implementation Requirements:**
1. Create a `CaseTagger` class that wraps the ML model
2. Handle multi-output prediction (12 different tags)
3. Support both single-label and multi-label tags
4. Implement proper handling of each tag type:
   - Binary classification (e.g., `case_ongoing`)
   - Multi-class (e.g., `state`)
   - Multi-label (e.g., `case_types`)
5. Use separate models for different tag types if needed

**Key Class Structure:**
```python
class CaseTagger:
    """Case-level tagging model"""
    
    def __init__(self, config: dict):
        self.models = {}  # One model per tag or group of tags
        self.encoders = None
        self.vectorizers = None
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        # For each tag type:
        # - Train appropriate model (binary/multiclass/multilabel)
        # - Use validation set for early stopping
        # - Track metrics
        
    def predict(self, X) -> dict:
        """Predict all tags for new cases"""
        
    def predict_proba(self, X) -> dict:
        """Get probability scores for predictions"""
        
    def save(self, filepath: Path):
        """Save model to disk"""
        
    @classmethod
    def load(cls, filepath: Path):
        """Load saved model"""
```

#### Step 3.3: Document-Level Model
**File: `src/models/document_tagger.py`**

**Implementation Requirements:**
1. Create a `DocumentTagger` class (similar to CaseTagger)
2. Handle 2 document-level tags (`document_type` and `party_types`)
3. Include case context features (case ID, case tags) if helpful
4. Simpler than case model (fewer tags)

**Key Class Structure:**
```python
class DocumentTagger:
    """Document-level tagging model"""
    
    def __init__(self, config: dict):
        self.models = {}
        self.encoders = None
        self.vectorizers = None
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        
    def predict(self, X) -> dict:
        """Predict tags for new documents"""
        
    def save(self, filepath: Path):
        """Save model"""
        
    @classmethod
    def load(cls, filepath: Path):
        """Load model"""
```

---

### **Phase 4: Training Pipeline**

#### Step 4.1: Training Orchestrator
**File: `src/training/trainer.py`**

**Purpose:** Coordinate the entire training process with proper logging and experiment tracking.

**Implementation Requirements:**
1. Create experiment directory with timestamp
2. Save configuration used for training
3. Train both models (case & document)
4. Log all metrics during training
5. Save best models based on validation performance
6. Create visualization of training progress

**Key Functions:**
```python
class Trainer:
    """Orchestrates model training"""
    
    def __init__(self, config: Config):
        self.config = config
        self.experiment_dir = None
        
    def setup_experiment(self) -> Path:
        """Create experiment directory and logging"""
        
    def train_case_model(self, train_data, val_data):
        """Train case-level model"""
        
    def train_document_model(self, train_data, val_data):
        """Train document-level model"""
        
    def save_results(self, metrics: dict):
        """Save training results and plots"""
```

#### Step 4.2: Model Evaluation
**File: `src/training/evaluator.py`**

**Purpose:** Comprehensive evaluation of model performance.

**Implementation Requirements:**
1. **For each tag**, calculate:
   - Accuracy (for single-label tags)
   - F1-score (micro, macro, weighted)
   - Precision and Recall
   - Confusion matrix (for categorical tags)
   - ROC-AUC (for binary/multi-class)
2. **For multi-label tags**, calculate:
   - Hamming loss
   - Subset accuracy
   - Per-label metrics
3. Generate visualizations:
   - Confusion matrices
   - ROC curves
   - Feature importance plots
4. Create human-readable report

**Key Functions:**
```python
class Evaluator:
    """Evaluate model performance"""
    
    def evaluate_case_model(self, model, test_data) -> dict:
        """Evaluate case-level model"""
        
    def evaluate_document_model(self, model, test_data) -> dict:
        """Evaluate document-level model"""
        
    def generate_report(self, metrics: dict, output_dir: Path):
        """Create evaluation report with plots"""
        
    def compare_models(self, results_list: list[dict]):
        """Compare multiple model versions"""
```

---

### **Phase 5: Inference Pipeline**

#### Step 5.1: Production Predictor
**File: `src/inference/predictor.py`**

**Purpose:** Make predictions on new, untagged cases.

**Implementation Requirements:**
1. Load trained models and preprocessors
2. Accept new case JSON files
3. Extract features using same pipeline as training
4. Make predictions for all tags
5. Return predictions in original tag format (decoded)
6. Save predictions to file

**Key Functions:**
```python
class Predictor:
    """Make predictions on new cases"""
    
    def __init__(self, models_dir: Path):
        self.case_model = CaseTagger.load(models_dir / 'case_tagger')
        self.doc_model = DocumentTagger.load(models_dir / 'document_tagger')
        self.encoders = TagEncoder.load(models_dir / 'encoders')
        self.vectorizers = load_vectorizers(models_dir)
        
    def predict_case(self, case_json: dict) -> dict:
        """Predict tags for a single case"""
        
    def predict_batch(self, case_jsons: list[dict]) -> list[dict]:
        """Predict tags for multiple cases"""
        
    def predict_from_files(self, cases_dir: Path, output_file: Path):
        """Process all cases in directory and save results"""
```

---

### **Phase 6: Executable Scripts**

#### Script 6.1: Data Fetching
**File: `scripts/fetch_data.py`**
```python
"""
Fetch cases from Clearinghouse API

Usage:
    python scripts/fetch_data.py --csv data/raw/training_cases.csv --output-dir training
    python scripts/fetch_data.py --csv data/raw/production_cases.csv --output-dir production
"""
# This wraps the existing data_ingestion.py with proper imports from src/
```

#### Script 6.2: Data Preparation
**File: `scripts/prepare_data.py`**
```python
"""
Prepare data for ML training

Steps:
1. Extract tags from JSON files
2. Process text features
3. Encode labels
4. Create train/val/test splits
5. Save processed datasets

Usage:
    python scripts/prepare_data.py --input data/cases/training
"""
```

#### Script 6.3: Model Training
**File: `scripts/train_models.py`**
```python
"""
Train both case and document tagging models

Usage:
    python scripts/train_models.py --data data/processed --experiment-name baseline_v1
    
Options:
    --model-type: xgboost, lightgbm, random_forest
    --tune-hyperparams: Run hyperparameter tuning
"""
```

#### Script 6.4: Model Evaluation
**File: `scripts/evaluate_models.py`**
```python
"""
Evaluate trained models on test set

Usage:
    python scripts/evaluate_models.py --models models/ --data data/processed/test
"""
```

#### Script 6.5: Prediction
**File: `scripts/predict.py`**
```python
"""
Make predictions on new cases

Usage:
    python scripts/predict.py --input data/cases/production --output predictions.json
"""
```

---

### **Phase 7: Testing**

#### Step 7.1: Unit Tests
**Files: `tests/test_*.py`**

**Requirements:**
1. Test all preprocessing functions
2. Test model loading/saving
3. Test prediction pipeline
4. Test configuration validation
5. Aim for >80% code coverage

**Example Structure:**
```python
# tests/test_preprocessing.py
def test_clean_text():
    """Test text cleaning"""
    
def test_tag_extraction():
    """Test extracting tags from JSON"""
    
def test_label_encoding():
    """Test encoding/decoding labels"""

# tests/test_models.py
def test_case_model_fit():
    """Test case model training"""
    
def test_document_model_predict():
    """Test document prediction"""

# tests/test_inference.py
def test_predictor_single_case():
    """Test single case prediction"""
    
def test_predictor_batch():
    """Test batch prediction"""
```

---

## 🚀 Execution Guide for User (No Programming Knowledge Required)

### Initial Setup (One Time)

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create `.env` file** (copy API token from existing code):
   ```bash
   echo "CLEARINGHOUSE_API_TOKEN=11a7d2a0a5c673e0d4391cb578563eb43d629a49" > .env
   ```

3. **Validate setup:**
   ```bash
   python -c "from config import Config; Config.validate(); print('✅ Setup complete!')"
   ```

### Complete ML Workflow

**Step 1: Fetch Training Data**
```bash
python scripts/fetch_data.py --csv data/raw/training_cases.csv --output-dir training
```

**Step 2: Prepare Data for ML**
```bash
python scripts/prepare_data.py --input data/cases/training
```

**Step 3: Train Models**
```bash
python scripts/train_models.py --data data/processed --experiment-name baseline_v1
```

**Step 4: Evaluate Models**
```bash
python scripts/evaluate_models.py --models models/ --data data/processed/test
```

**Step 5: Fetch Production Data**
```bash
python scripts/fetch_data.py --csv data/raw/production_cases.csv --output-dir production
```

**Step 6: Make Predictions**
```bash
python scripts/predict.py --input data/cases/production --output predictions.json
```

---

## 📋 File Migration Plan

Move existing files to new structure:

```bash
# Move data ingestion scripts
mv data_ingestion.py src/ingestion/
mv data_utils.py src/ingestion/

# Move CSV to raw data
mv "clearinghouse_case_links_by_collections(without 5679 and 5666).csv" data/raw/

# Move cases directory
mv cases/ data/

# Clean up old files
mv extract_json.ipynb notebooks/00_old_exploration.ipynb
rm -f data.py  # Old test file, no longer needed
```

**Update Documentation After Migration**

After moving files, update the following:

1. **Update `data_ingestion.py`:**
   - Update all imports to use `from config import Config` instead of hardcoded values
   - Update documentation/docstrings to reflect new location (`src/ingestion/`)
   - Update path references to use `Config.CASES_DIR` instead of relative paths
   - Ensure API token is loaded from `.env` via Config class

2. **Update `data_utils.py`:**
   - Update imports to use config module
   - Update documentation/docstrings to reflect new location
   - Update default paths to use config constants
   - Update examples in docstrings to reflect new directory structure

3. **Update `DATA_INGESTION_README.md`:**
   - Update all file paths to reflect new structure (`src/ingestion/`, `data/cases/`, etc.)
   - Update example commands to reference new locations
   - Add note about using `.env` file for configuration

4. **Create new `README.md`:**
   - Document complete ML workflow
   - Include setup instructions with `.env` configuration
   - Reference all executable scripts in `scripts/` directory
   - Include architecture overview of new structure

---

## 🎯 Success Criteria

### Model Performance Targets
- **Case-level tags:** 
  - Accuracy > 70% for single-label tags
  - F1-score > 0.6 for multi-label tags
- **Document-level tags:**
  - Accuracy > 75% for all tags
  - F1-score > 0.65

### System Requirements
- ✅ API token never exposed in code
- ✅ All models can be saved and loaded
- ✅ Predictions can be made on new data without retraining
- ✅ Complete experiment tracking
- ✅ Reproducible results (fixed random seeds)
- ✅ Clean, documented code
- ✅ User can run entire pipeline with simple commands

---

## 📚 Additional Considerations

### Handling Data Challenges
1. **Imbalanced tags:** Use class weights or oversampling
2. **Missing tags:** Train with incomplete data, use special "unknown" category
3. **New tags in production:** Have fallback strategy (predict "OTHER" or skip)
4. **Text length variation:** Use truncation/padding or aggregation

### Model Improvement Ideas (Future)
1. Use BERT/transformer models for better text understanding
2. Ensemble multiple models for better accuracy
3. Active learning: identify cases where model is uncertain
4. Transfer learning from legal domain pre-trained models

### Monitoring in Production
1. Log prediction confidence scores
2. Track distribution of predicted tags
3. Identify cases with low confidence for human review
4. Regular model retraining with new tagged data

---

## ✅ Implementation Checklist for Copilot

- [ ] Phase 1: Setup (`.env`, `config.py`, `requirements.txt`, directory structure)
- [ ] Phase 2: Preprocessing (tag extraction, text processing, encoding, dataset building)
- [ ] Phase 3: Models (CaseTagger, DocumentTagger)
- [ ] Phase 4: Training (Trainer, Evaluator)
- [ ] Phase 5: Inference (Predictor)
- [ ] Phase 6: Scripts (all 5 executable scripts)
- [ ] Phase 7: Tests (unit tests for all components)
- [ ] File Migration (move existing files to new structure)
- [ ] Documentation Updates:
  - [ ] Update imports in `data_ingestion.py` and `data_utils.py` to use config
  - [ ] Update docstrings in moved files to reflect new paths
  - [ ] Update `DATA_INGESTION_README.md` with new structure
  - [ ] Create comprehensive main `README.md` with ML workflow
- [ ] Final Validation (run complete pipeline end-to-end)

---

## 🆘 Support & Documentation

All scripts include `--help` flag:
```bash
python scripts/prepare_data.py --help
python scripts/train_models.py --help
python scripts/predict.py --help
```

For detailed information, see:
- `README.md` - Main project documentation
- `data/README.md` - Data structure explanation
- `models/README.md` - Model architecture details
- `experiments/README.md` - How to interpret results

---

**End of Specification**

This spec provides complete guidance for implementing a production-ready ML system for legal case tagging. Every component is designed to be secure, maintainable, and accessible to non-programmers through simple command-line scripts.
