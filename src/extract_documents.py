"""Extract documents from case JSON files into a flat CSV format."""
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import Config

def extract_documents_from_case(case_json: dict, case_id: str) -> list[dict]:
    """
    Extract all documents from a single case JSON.
    
    IMPORTANT: Documents are at case_json['documents'], NOT case_json['case_data']['documents']
    
    Returns list of dicts, one per document.
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
    """Process all JSON files in batches. Memory-safe."""
    json_files = sorted(list(cases_dir.glob('case_*.json')))
    print(f"Found {len(json_files)} case files in {cases_dir}")
    
    if len(json_files) == 0:
        print(f"Warning: No case files found in {cases_dir}")
        return
    
    total_docs = 0
    
    # Process in batches
    for i in tqdm(range(0, len(json_files), Config.BATCH_SIZE), desc="Processing batches"):
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
                print(f"\nError processing {json_file}: {e}")
        
        # Write batch to CSV (append mode)
        if batch_docs:
            df_batch = pd.DataFrame(batch_docs)
            mode = 'w' if i == 0 else 'a'
            header = (i == 0)
            df_batch.to_csv(output_csv, mode=mode, header=header, index=False)
            total_docs += len(batch_docs)
    
    print(f"\nExtraction complete!")
    print(f"   Total documents extracted: {total_docs}")
    print(f"   Saved to: {output_csv}")

def main():
    """Main execution"""
    print("=" * 60)
    print("PHASE 2: DOCUMENT EXTRACTION")
    print("=" * 60)
    print()
    
    # Create output directory
    Config.create_dirs()
    
    # Extract training documents
    print("Extracting documents from training cases...")
    output_file = Config.PROCESSED_DIR / 'documents_all.csv'
    
    process_json_files_in_batches(
        Config.CASES_TRAINING_DIR,
        output_file
    )
    
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE")
    print("=" * 60)
    print(f"\nNext step: Run EDA notebook to analyze the data")
    print(f"  jupyter notebook src/eda.ipynb")

if __name__ == '__main__':
    main()
