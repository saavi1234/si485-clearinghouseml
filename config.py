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
