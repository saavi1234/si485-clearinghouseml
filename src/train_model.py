"""
Train document tagging models.
Simple Logistic Regression for both document_type and party_types.
"""
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, hamming_loss
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


def train_models():
    """Train document tagging models"""
    
    print("Loading training data...")
    train_data = joblib.load(Config.PROCESSED_DIR / 'documents_train.pkl')
    val_data = joblib.load(Config.PROCESSED_DIR / 'documents_val.pkl')
    
    X_train = train_data['X']
    X_val = val_data['X']
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # ================================================================
    # Train document_type model (multi-class classification)
    # ================================================================
    print("\n" + "=" * 60)
    print("Training document_type classifier (multi-class)")
    print("=" * 60)
    
    model_doctype = LogisticRegression(
        max_iter=1000, 
        random_state=Config.RANDOM_SEED,
        solver='lbfgs',
        multi_class='multinomial',
        n_jobs=-1  # Use all CPU cores
    )
    
    print("Fitting model...")
    model_doctype.fit(X_train, train_data['y_document_type'])
    
    # Evaluate on validation
    print("\nEvaluating on validation set...")
    y_pred = model_doctype.predict(X_val)
    y_true = val_data['y_document_type']
    
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\nValidation Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    
    # Detailed classification report
    encoders = joblib.load(Config.PROCESSED_DIR / 'label_encoders.pkl')
    target_names = encoders['document_type'].classes_
    
    print("\nDetailed Classification Report:")
    # Get unique labels that appear in validation set
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    print(classification_report(
        y_true, 
        y_pred, 
        labels=unique_labels,
        target_names=[target_names[i] for i in unique_labels],
        zero_division=0
    ))
    
    doctype_metrics = {
        'accuracy': float(acc),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted)
    }
    
    # ================================================================
    # Train party_types model (multi-label classification)
    # ================================================================
    print("\n" + "=" * 60)
    print("Training party_types classifier (multi-label)")
    print("=" * 60)
    
    model_party = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000, 
            random_state=Config.RANDOM_SEED,
            solver='lbfgs',
            n_jobs=-1
        )
    )
    
    print("Fitting model...")
    model_party.fit(X_train, train_data['y_party_types'])
    
    # Evaluate on validation
    print("\nEvaluating on validation set...")
    y_pred_party = model_party.predict(X_val)
    y_true_party = val_data['y_party_types']
    
    # Multi-label metrics
    hamming = hamming_loss(y_true_party, y_pred_party)
    f1_micro = f1_score(y_true_party, y_pred_party, average='micro')
    f1_macro = f1_score(y_true_party, y_pred_party, average='macro')
    f1_weighted = f1_score(y_true_party, y_pred_party, average='weighted')
    f1_samples = f1_score(y_true_party, y_pred_party, average='samples')
    
    # Subset accuracy (exact match)
    subset_acc = accuracy_score(y_true_party, y_pred_party)
    
    print(f"\nValidation Results:")
    print(f"  Hamming Loss: {hamming:.4f} (lower is better)")
    print(f"  Subset Accuracy: {subset_acc:.4f} (exact match)")
    print(f"  F1 (micro): {f1_micro:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(f"  F1 (samples): {f1_samples:.4f}")
    
    # Per-label report
    party_names = encoders['party_types'].classes_
    print("\nPer-Label Classification Report:")
    print(classification_report(
        y_true_party, 
        y_pred_party, 
        target_names=party_names,
        zero_division=0
    ))
    
    party_metrics = {
        'hamming_loss': float(hamming),
        'subset_accuracy': float(subset_acc),
        'f1_micro': float(f1_micro),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'f1_samples': float(f1_samples)
    }
    
    # ================================================================
    # Save models
    # ================================================================
    print("\n" + "=" * 60)
    print("Saving models...")
    print("=" * 60)
    
    models = {
        'document_type': model_doctype,
        'party_types': model_party
    }
    joblib.dump(models, Config.MODEL_DIR / 'document_tagger.pkl')
    print(f"Saved model to {Config.MODEL_DIR / 'document_tagger.pkl'}")
    
    # Save metadata
    metadata = {
        'config': {
            'max_features': Config.MAX_FEATURES,
            'min_df': Config.MIN_DF,
            'max_df': Config.MAX_DF,
            'random_seed': Config.RANDOM_SEED
        },
        'data': {
            'num_train_samples': len(train_data['y_document_type']),
            'num_val_samples': len(val_data['y_document_type']),
            'num_document_types': len(target_names),
            'num_party_types': len(party_names)
        },
        'document_type_metrics': doctype_metrics,
        'party_types_metrics': party_metrics
    }
    
    with open(Config.MODEL_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {Config.MODEL_DIR / 'metadata.json'}")
    
    print("\nTraining complete!")
    return metadata


if __name__ == '__main__':
    print("=" * 60)
    print("PHASE 4: Training Document Tagging Models")
    print("=" * 60)
    print()
    
    # Check if processed data exists
    required_files = [
        Config.PROCESSED_DIR / 'documents_train.pkl',
        Config.PROCESSED_DIR / 'documents_val.pkl',
        Config.PROCESSED_DIR / 'label_encoders.pkl'
    ]
    
    for file in required_files:
        if not file.exists():
            print(f"ERROR: Required file not found: {file}")
            print("Please run process_features.py first (Phase 3)")
            sys.exit(1)
    
    Config.create_dirs()
    metadata = train_models()
    
    print("\n" + "=" * 60)
    print("Phase 4 Complete!")
    print("=" * 60)
    print("\nFinal Metrics Summary:")
    print("\nDocument Type (multi-class):")
    print(f"  Accuracy: {metadata['document_type_metrics']['accuracy']:.4f}")
    print(f"  F1 (weighted): {metadata['document_type_metrics']['f1_weighted']:.4f}")
    print("\nParty Types (multi-label):")
    print(f"  Subset Accuracy: {metadata['party_types_metrics']['subset_accuracy']:.4f}")
    print(f"  F1 (micro): {metadata['party_types_metrics']['f1_micro']:.4f}")
    print(f"  F1 (samples): {metadata['party_types_metrics']['f1_samples']:.4f}")
    print("\nModel ready for inference!")
