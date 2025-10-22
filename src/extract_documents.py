"""
Extract documents from case JSON files into a flat CSV format.
Memory-safe batch processing implementation.
"""
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
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
    
    Args:
        cases_dir: Directory containing case_*.json files
        output_csv: Path to output CSV file
    """
    json_files = sorted(list(cases_dir.glob('case_*.json')))
    print(f"Found {len(json_files)} case files in {cases_dir}")
    
    if len(json_files) == 0:
        print(f"WARNING: No case files found in {cases_dir}")
        return
    
    total_docs = 0
    
    # Process in batches
    for i in tqdm(range(0, len(json_files), Config.BATCH_SIZE), desc="Processing batches"):
        batch_files = json_files[i:i + Config.BATCH_SIZE]
        batch_docs = []
        
        for json_file in batch_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    case_json = json.load(f)
                
                case_id = json_file.stem  # e.g., 'case_10000'
                docs = extract_documents_from_case(case_json, case_id)
                batch_docs.extend(docs)
                
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {json_file}: {e}")
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        # Write batch to CSV (append mode after first batch)
        if batch_docs:
            df_batch = pd.DataFrame(batch_docs)
            mode = 'w' if i == 0 else 'a'
            header = (i == 0)
            df_batch.to_csv(output_csv, mode=mode, header=header, index=False)
            total_docs += len(batch_docs)
    
    print(f"Saved {total_docs} documents to {output_csv}")


# Main execution
if __name__ == '__main__':
    Config.create_dirs()
    
    # Extract training documents
    print("=" * 60)
    print("PHASE 2: Extracting Documents from Training Cases")
    print("=" * 60)
    
    output_file = Config.PROCESSED_DIR / 'documents_all.csv'
    
    print(f"\nInput directory: {Config.CASES_TRAINING_DIR}")
    print(f"Output file: {output_file}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Max text length: {Config.MAX_TEXT_LENGTH} characters\n")
    
    process_json_files_in_batches(
        Config.CASES_TRAINING_DIR,
        output_file
    )
    
    print("\n" + "=" * 60)
    print("Document extraction complete!")
    print("=" * 60)
    
    # Show summary statistics
    if output_file.exists():
        df = pd.read_csv(output_file)
        print(f"\nSummary Statistics:")
        print(f"  Total documents: {len(df)}")
        print(f"  Unique cases: {df['case_id'].nunique()}")
        print(f"  Documents with document_type tags: {df['document_type'].notna().sum()}")
        print(f"  Documents with party_types tags: {df['party_types'].notna().sum()}")
        print(f"\nDocument type distribution:")
        print(df['document_type'].value_counts().head(10))
