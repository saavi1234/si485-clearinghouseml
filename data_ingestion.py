"""
Data Ingestion Script for Civil Rights Clearinghouse Cases

This script fetches case metadata and associated documents from the 
Clearinghouse API and saves them as individual JSON files for ML training.

Usage:
    python data_ingestion.py --case-id <id> --output-dir <dir>
    python data_ingestion.py --case-ids-file <path> --output-dir <dir> [--refetch]
    
Examples:
    # Fetch single case
    python data_ingestion.py --case-id 12345 --output-dir training
    
    # Fetch from text file with case IDs
    python data_ingestion.py --case-ids-file case_ids.txt --output-dir training
    
    # Re-fetch to update existing cases
    python data_ingestion.py --case-ids-file case_ids.txt --output-dir production --refetch
"""

import json
import os
import re
import time
import argparse
from datetime import datetime
from pathlib import Path
import requests
from typing import Optional, List, Dict, Any


# Configuration
API_TOKEN = "11a7d2a0a5c673e0d4391cb578563eb43d629a49"
HEADERS = {
    "Authorization": f"Token {API_TOKEN}",
    "User-Agent": "Chrome v22.2 Linux Ubuntu"
}
CASE_API_URL = "https://clearinghouse.net/api/v2/case/"
DOCUMENTS_API_URL = "https://clearinghouse.net/api/v2/documents/"
OUTPUT_DIR = "cases"
FAILED_CASES_LOG = "failed_cases.json"


class ClearinghouseDataIngestion:
    """Handles fetching and storing case and document data from Clearinghouse API"""
    
    def __init__(self, output_dir: str = OUTPUT_DIR):
        # Ensure base cases directory structure exists
        self._ensure_cases_structure()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)  # Create all parent directories
        self.failed_cases = []
        self.successful_cases = []
        self.skipped_cases = []
    
    def _ensure_cases_structure(self):
        """Ensure cases folder with training and production subfolders exist"""
        cases_dir = Path("cases")
        training_dir = cases_dir / "training"
        production_dir = cases_dir / "production"
        
        # Create all directories if they don't exist
        training_dir.mkdir(parents=True, exist_ok=True)
        production_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_case_ids_from_txt(self, txt_path: str) -> List[int]:
        """Extract case IDs from a text file (one ID per line)"""
        case_ids = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and line.isdigit():
                    case_ids.append(int(line))
        print(f"📋 Found {len(case_ids)} case IDs in {txt_path}")
        return case_ids
    
    def fetch_with_retry(
        self, 
        url: str, 
        params: Dict[str, Any], 
        max_retries: int = 5, 
        base_delay: float = 1.5
    ) -> Optional[Dict]:
        """
        Fetch data from API with exponential backoff retry logic
        
        Args:
            url: API endpoint URL
            params: Query parameters
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries (doubles each time)
            
        Returns:
            JSON response dict or None if failed
        """
        delay = base_delay
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(
                    url,
                    params=params,
                    headers=HEADERS,
                    timeout=30,
                )
                
                if resp.status_code == 200:
                    return resp.json()
                
                if resp.status_code == 429:
                    print(f"⚠️  Rate limit hit, waiting {delay:.1f}s (attempt {attempt}/{max_retries})")
                elif 500 <= resp.status_code < 600:
                    print(f"⚠️  Server error {resp.status_code}, retrying in {delay:.1f}s (attempt {attempt}/{max_retries})")
                else:
                    print(f"❌ HTTP {resp.status_code}: {resp.text[:200]}")
                    return None
                
                time.sleep(delay)
                delay *= 2
                
            except requests.exceptions.Timeout:
                print(f"⏱️  Timeout, retrying in {delay:.1f}s (attempt {attempt}/{max_retries})")
                time.sleep(delay)
                delay *= 2
            except requests.exceptions.RequestException as e:
                print(f"❌ Request error: {e}")
                return None
        
        print(f"❌ Failed after {max_retries} retries")
        return None
    
    def fetch_case_data(self, case_id: int) -> Optional[Dict]:
        """Fetch case metadata and tags"""
        print(f"  📄 Fetching case metadata for case {case_id}...")
        response = self.fetch_with_retry(
            CASE_API_URL,
            params={"case_id": case_id}
        )
        
        if response and "results" in response:
            results = response["results"]
            if results:
                return results[0]
        
        return None
    
    def fetch_case_documents(self, case_id: int) -> List[Dict]:
        """Fetch all documents associated with a case"""
        print(f"  📚 Fetching documents for case {case_id}...")
        response = self.fetch_with_retry(
            DOCUMENTS_API_URL,
            params={"case": case_id}
        )
        
        if response:
            # Handle both dict with "results" and direct list responses
            if isinstance(response, dict) and "results" in response:
                documents = response["results"]
            elif isinstance(response, list):
                documents = response
            else:
                print(f"  ⚠️  Unexpected response format")
                return []
            
            print(f"  ✓ Found {len(documents)} documents")
            return documents
        
        print(f"  ⚠️  No documents found")
        return []
    
    def save_case(self, case_id: int, case_data: Dict, documents: List[Dict]) -> bool:
        """
        Save combined case and document data to individual JSON file
        
        Args:
            case_id: The case ID number
            case_data: Full case metadata from API
            documents: List of document objects from API
            
        Returns:
            True if saved successfully, False otherwise
        """
        output_path = self.output_dir / f"case_{case_id}.json"
        
        combined_data = {
            "case_id": case_id,
            "fetched_at": datetime.now().isoformat(),
            "case_data": case_data,
            "documents": documents,
            "document_count": len(documents)
        }
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(combined_data, f, indent=2, ensure_ascii=False)
            print(f"  ✅ Saved to {output_path}")
            return True
        except Exception as e:
            print(f"  ❌ Failed to save: {e}")
            return False
    
    def case_already_fetched(self, case_id: int) -> bool:
        """Check if case data already exists"""
        output_path = self.output_dir / f"case_{case_id}.json"
        return output_path.exists()
    
    def check_case_changed(self, case_id: int, new_case_data: Dict, new_documents: List[Dict]) -> bool:
        """
        Check if case data has changed since last fetch
        
        Args:
            case_id: The case ID
            new_case_data: Newly fetched case data
            new_documents: Newly fetched documents
            
        Returns:
            True if data has changed, False if identical
        """
        output_path = self.output_dir / f"case_{case_id}.json"
        
        if not output_path.exists():
            return True  # New case, consider it "changed"
        
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                old_data = json.load(f)
            
            # Compare case data (excluding fetch timestamp)
            old_case = old_data.get("case_data", {})
            old_docs = old_data.get("documents", [])
            
            # Simple comparison: check if content differs
            # Note: This compares the full data structures
            case_changed = old_case != new_case_data
            docs_changed = old_docs != new_documents
            
            return case_changed or docs_changed
            
        except Exception as e:
            print(f"  ⚠️  Error reading old file: {e}")
            return True  # If we can't read old file, assume changed
    
    def process_case(self, case_id: int, skip_existing: bool = True, refetch: bool = False) -> bool:
        """
        Process a single case: fetch case data, documents, and save
        
        Args:
            case_id: The case ID to process
            skip_existing: If True, skip cases that already have files
            refetch: If True, check for changes and update if changed
            
        Returns:
            True if successful, False otherwise
        """
        print(f"\n{'='*60}")
        print(f"Processing Case ID: {case_id}")
        print(f"{'='*60}")
        
        # Check if already exists (only if not refetching)
        if not refetch and skip_existing and self.case_already_fetched(case_id):
            print(f"  ⏭️  Case {case_id} already exists, skipping...")
            self.skipped_cases.append(case_id)
            return True
        
        # Fetch case metadata
        case_data = self.fetch_case_data(case_id)
        if not case_data:
            print(f"  ❌ Failed to fetch case data for {case_id}")
            self.failed_cases.append({
                "case_id": case_id,
                "reason": "Failed to fetch case data",
                "timestamp": datetime.now().isoformat()
            })
            return False
        
        # Fetch documents
        documents = self.fetch_case_documents(case_id)
        
        # Check if data changed (only if refetching)
        if refetch and self.case_already_fetched(case_id):
            if not self.check_case_changed(case_id, case_data, documents):
                print(f"  ℹ️  Case {case_id} unchanged, skipping...")
                self.skipped_cases.append(case_id)
                return True
            else:
                print(f"  🔄 Case {case_id} has changes, updating...")
        
        # Save combined data
        if self.save_case(case_id, case_data, documents):
            self.successful_cases.append(case_id)
            return True
        else:
            self.failed_cases.append({
                "case_id": case_id,
                "reason": "Failed to save data",
                "timestamp": datetime.now().isoformat()
            })
            return False
    
    def save_failed_cases_log(self):
        """Save failed cases log"""
        if self.failed_cases:
            with open(FAILED_CASES_LOG, "w", encoding="utf-8") as f:
                json.dump(self.failed_cases, f, indent=2)
            print(f"\n⚠️  Saved {len(self.failed_cases)} failed cases to {FAILED_CASES_LOG}")
    
    def run(self, case_ids: List[int], skip_existing: bool = True, rate_limit_delay: float = 0.3, refetch: bool = False):
        """
        Main execution method to process all cases
        
        Args:
            case_ids: List of case IDs to process
            skip_existing: Whether to skip already-fetched cases
            rate_limit_delay: Delay between requests (seconds)
            refetch: Whether to check for changes and re-fetch updated cases
        """
        print("\n" + "="*60)
        print("🚀 CLEARINGHOUSE DATA INGESTION")
        print("="*60)
        
        total_cases = len(case_ids)
        
        print(f"\n📊 Configuration:")
        print(f"   • Total cases to process: {total_cases}")
        print(f"   • Output directory: {self.output_dir}")
        print(f"   • Skip existing: {skip_existing}")
        print(f"   • Re-fetch mode: {'ON (checking for updates)' if refetch else 'OFF'}")
        print(f"   • Rate limit delay: {rate_limit_delay}s")
        
        # Process each case
        start_time = time.time()
        for idx, case_id in enumerate(case_ids, 1):
            print(f"\n[{idx}/{total_cases}] ", end="")
            self.process_case(case_id, skip_existing=skip_existing, refetch=refetch)
            
            # Rate limiting - be polite to the API
            if idx < total_cases:
                time.sleep(rate_limit_delay)
        
        # Final summary
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print("📊 INGESTION COMPLETE")
        print("="*60)
        print(f"✅ Successful: {len(self.successful_cases)}")
        print(f"⏭️  Skipped (already existed): {len(self.skipped_cases)}")
        print(f"❌ Failed: {len(self.failed_cases)}")
        print(f"⏱️  Total time: {elapsed_time/60:.2f} minutes")
        print(f"📁 Data saved to: {self.output_dir.absolute()}")
        
        # Save failed cases log
        self.save_failed_cases_log()
        
        if self.failed_cases:
            print(f"\n⚠️  Check {FAILED_CASES_LOG} for failed case details")
            print(f"   You can retry failed cases later")


def main():
    """Main entry point with command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Fetch case data from Civil Rights Clearinghouse API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch single case
  python data_ingestion.py --case-id 12345 --output-dir training
  
  # Fetch from text file with case IDs
  python data_ingestion.py --case-ids-file case_ids.txt --output-dir training
  
  # Re-fetch to check for updates
  python data_ingestion.py --case-ids-file case_ids.txt --output-dir production --refetch
        """
    )
    
    # Create mutually exclusive group for input
    input_group = parser.add_mutually_exclusive_group(required=True)
    
    input_group.add_argument(
        '--case-id',
        type=int,
        help='Single case ID to fetch'
    )
    
    input_group.add_argument(
        '--case-ids-file',
        type=str,
        help='Path to text file with case IDs (one per line)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=OUTPUT_DIR,
        help=f'Output directory for case files (default: {OUTPUT_DIR}). Use "training" or "production" for organized data.'
    )
    
    parser.add_argument(
        '--refetch',
        action='store_true',
        help='Re-fetch all cases and update if changed (default: skip existing)'
    )
    
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=1.0,
        help='Delay between API requests in seconds (default: 1.0)'
    )
    
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Force re-fetch all cases regardless of changes (overwrites existing)'
    )
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    if output_dir in ["training", "production"]:
        output_dir = f"cases/{output_dir}"
    
    # Create ingestion instance with specified output directory
    ingestion = ClearinghouseDataIngestion(output_dir=output_dir)
    
    # Determine case IDs to process
    case_ids = None
    if args.case_id:
        case_ids = [args.case_id]
        print(f"📌 Processing single case ID: {args.case_id}")
    elif args.case_ids_file:
        case_ids = ingestion.extract_case_ids_from_txt(args.case_ids_file)
    
    # Run ingestion
    ingestion.run(
        case_ids=case_ids,
        skip_existing=not args.no_skip,  # Skip unless --no-skip is set
        rate_limit_delay=args.rate_limit,
        refetch=args.refetch
    )


if __name__ == "__main__":
    main()
