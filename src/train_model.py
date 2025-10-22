import joblib
import json
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import Config

def train_models():
    """
    Train document_type and party_types classifiers.
    Evaluate on validation set and save models.
    """
    
    print("Loading processed data...")
    train_data = joblib.load(Config.PROCESSED_DIR / 'documents_train.pkl')
    val_data = joblib.load(Config.PROCESSED_DIR / 'documents_val.pkl')
    encoders = joblib.load(Config.PROCESSED_DIR / 'label_encoders.pkl')
    
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
        'train_samples': len(train_data['doc_ids']),
        'val_samples': len(val_data['doc_ids']),
        'num_features': X_train.shape[1],
        'document_type': {
            'num_classes': len(encoders['document_type'].classes_),
            'classes': list(encoders['document_type'].classes_),
            'val_accuracy': float(acc),
            'val_f1_weighted': float(f1),
            'class_weights_enabled': Config.USE_CLASS_WEIGHTS
        },
        'party_types': {
            'num_classes': len(encoders['party_types'].classes_),
            'classes': list(encoders['party_types'].classes_),
            'val_f1_micro': float(f1_micro),
            'val_f1_macro': float(f1_macro),
            'class_weights_enabled': Config.USE_CLASS_WEIGHTS
        },
        'config': {
            'max_features': Config.MAX_FEATURES,
            'min_df': Config.MIN_DF,
            'max_df': Config.MAX_DF,
            'min_tag_frequency': Config.MIN_TAG_FREQUENCY,
            'random_seed': Config.RANDOM_SEED
        }
    }
    
    with open(Config.MODEL_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"   - Models saved to: {Config.MODEL_DIR / 'document_tagger.pkl'}")
    print(f"   - Metadata saved to: {Config.MODEL_DIR / 'metadata.json'}")
    print(f"\n📊 Summary:")
    print(f"   Document Type Classifier:")
    print(f"     - Classes: {len(encoders['document_type'].classes_)}")
    print(f"     - Val Accuracy: {acc:.1%}")
    print(f"     - Val F1 (weighted): {f1:.3f}")
    print(f"     - Class weights: {'enabled' if Config.USE_CLASS_WEIGHTS else 'disabled'}")
    print(f"\n   Party Types Classifier:")
    print(f"     - Classes: {len(encoders['party_types'].classes_)}")
    print(f"     - Val F1 (micro): {f1_micro:.3f}")
    print(f"     - Val F1 (macro): {f1_macro:.3f}")
    print(f"     - Class weights: {'enabled' if Config.USE_CLASS_WEIGHTS else 'disabled'}")

if __name__ == '__main__':
    Config.create_dirs()
    train_models()
