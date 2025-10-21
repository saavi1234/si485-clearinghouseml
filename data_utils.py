"""
Utility functions for working with ingested Clearinghouse data

This module provides helpers for loading, exploring, and validating 
the case data that was fetched by data_ingestion.py
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter


class DataExplorer:
    """Helper class for exploring and analyzing ingested case data"""
    
    def __init__(self, cases_dir: str = "cases"):
        self.cases_dir = Path(cases_dir)
        self.case_files = list(self.cases_dir.glob("case_*.json"))
        
    def load_case(self, case_id: int) -> Optional[Dict]:
        """Load a single case by ID"""
        case_file = self.cases_dir / f"case_{case_id}.json"
        if case_file.exists():
            with open(case_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def load_all_cases(self) -> List[Dict]:
        """Load all cases into memory (warning: may be large)"""
        cases = []
        for case_file in self.case_files:
            with open(case_file, 'r', encoding='utf-8') as f:
                cases.append(json.load(f))
        return cases
    
    def get_case_ids(self) -> List[int]:
        """Get list of all case IDs that have been fetched"""
        case_ids = []
        for case_file in self.case_files:
            # Extract ID from filename like "case_12345.json"
            case_id = int(case_file.stem.split("_")[1])
            case_ids.append(case_id)
        return sorted(case_ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the dataset"""
        total_cases = len(self.case_files)
        total_documents = 0
        cases_with_no_documents = 0
        document_counts = []
        
        for case_file in self.case_files:
            with open(case_file, 'r', encoding='utf-8') as f:
                case_data = json.load(f)
                doc_count = case_data.get("document_count", 0)
                total_documents += doc_count
                document_counts.append(doc_count)
                if doc_count == 0:
                    cases_with_no_documents += 1
        
        stats = {
            "total_cases": total_cases,
            "total_documents": total_documents,
            "avg_documents_per_case": total_documents / total_cases if total_cases > 0 else 0,
            "cases_with_no_documents": cases_with_no_documents,
            "max_documents_in_case": max(document_counts) if document_counts else 0,
            "min_documents_in_case": min(document_counts) if document_counts else 0,
        }
        
        return stats
    
    def print_stats(self):
        """Print dataset statistics"""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("📊 DATASET STATISTICS")
        print("="*60)
        print(f"Total Cases: {stats['total_cases']}")
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Avg Documents per Case: {stats['avg_documents_per_case']:.2f}")
        print(f"Cases with No Documents: {stats['cases_with_no_documents']}")
        print(f"Max Documents in a Case: {stats['max_documents_in_case']}")
        print(f"Min Documents in a Case: {stats['min_documents_in_case']}")
        print("="*60)
    
    def inspect_case(self, case_id: int):
        """Print detailed information about a specific case"""
        case = self.load_case(case_id)
        if not case:
            print(f"❌ Case {case_id} not found")
            return
        
        case_data = case.get("case_data", {})
        documents = case.get("documents", [])
        
        print("\n" + "="*60)
        print(f"📋 CASE {case_id} DETAILS")
        print("="*60)
        print(f"Case Name: {case_data.get('name', 'N/A')}")
        print(f"Court: {case_data.get('court', 'N/A')}")
        print(f"Fetched At: {case.get('fetched_at', 'N/A')}")
        print(f"\n📚 Documents: {len(documents)}")
        
        for idx, doc in enumerate(documents, 1):
            print(f"\n  Document {idx}:")
            print(f"    Title: {doc.get('title', 'N/A')}")
            print(f"    Type: {doc.get('document_type', 'N/A')}")
            text = doc.get('text', '')
            if text:
                print(f"    Text Length: {len(text)} characters")
                print(f"    Text Preview: {text[:100]}...")
            else:
                print(f"    Text: [No text content]")
        
        # Show available case-level tags/fields
        print(f"\n🏷️  Case Data Fields Available:")
        for key in sorted(case_data.keys()):
            value = case_data[key]
            if isinstance(value, list):
                print(f"    • {key}: [{len(value)} items]")
            elif isinstance(value, dict):
                print(f"    • {key}: {{dict with {len(value)} keys}}")
            else:
                print(f"    • {key}: {type(value).__name__}")
        
        print("="*60)
    
    def get_tag_distribution(self, tag_field: str, level: str = "case") -> Dict[str, int]:
        """
        Get distribution of values for a specific tag field
        
        Args:
            tag_field: Name of the field to analyze (e.g., 'court', 'facility_type')
            level: Either 'case' or 'document'
        
        Returns:
            Dictionary mapping values to counts
        """
        values = []
        
        for case_file in self.case_files:
            with open(case_file, 'r', encoding='utf-8') as f:
                case = json.load(f)
                
                if level == "case":
                    case_data = case.get("case_data", {})
                    field_value = case_data.get(tag_field)
                    if field_value is not None:
                        if isinstance(field_value, list):
                            values.extend(field_value)
                        else:
                            values.append(field_value)
                
                elif level == "document":
                    for doc in case.get("documents", []):
                        field_value = doc.get(tag_field)
                        if field_value is not None:
                            if isinstance(field_value, list):
                                values.extend(field_value)
                            else:
                                values.append(field_value)
        
        return dict(Counter(values))
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Validate the integrity of the fetched data
        
        Returns:
            Dictionary with validation results
        """
        issues = []
        cases_without_case_data = []
        cases_without_documents = []
        documents_without_body = []
        
        for case_file in self.case_files:
            with open(case_file, 'r', encoding='utf-8') as f:
                case = json.load(f)
                case_id = case.get("case_id")
                
                # Check case data exists
                if not case.get("case_data"):
                    cases_without_case_data.append(case_id)
                
                # Check documents
                documents = case.get("documents", [])
                if not documents:
                    cases_without_documents.append(case_id)
                
                # Check document bodies
                for doc in documents:
                    if not doc.get("text"):
                        documents_without_body.append({
                            "case_id": case_id,
                            "document_title": doc.get("title", "N/A")
                        })
        
        validation_results = {
            "total_cases_checked": len(self.case_files),
            "cases_without_case_data": cases_without_case_data,
            "cases_without_documents": cases_without_documents,
            "documents_without_body": documents_without_body,
            "validation_passed": (
                len(cases_without_case_data) == 0 and 
                len(documents_without_body) == 0
            )
        }
        
        return validation_results
    
    def print_validation_results(self):
        """Print validation results"""
        results = self.validate_data()
        
        print("\n" + "="*60)
        print("✅ DATA VALIDATION RESULTS")
        print("="*60)
        print(f"Total Cases Checked: {results['total_cases_checked']}")
        print(f"\n⚠️  Cases Without Case Data: {len(results['cases_without_case_data'])}")
        if results['cases_without_case_data']:
            print(f"   {results['cases_without_case_data'][:10]}")
        
        print(f"\nℹ️  Cases Without Documents: {len(results['cases_without_documents'])}")
        if results['cases_without_documents']:
            print(f"   (This may be normal for some cases)")
        
        print(f"\n⚠️  Documents Without Body Text: {len(results['documents_without_body'])}")
        if results['documents_without_body']:
            print(f"   {results['documents_without_body'][:5]}")
        
        if results['validation_passed']:
            print("\n✅ All critical validations passed!")
        else:
            print("\n⚠️  Some validation issues found - review above")
        print("="*60)


def retry_failed_cases(failed_cases_log: str = "failed_cases.json"):
    """
    Helper function to retry cases that failed during ingestion
    
    Args:
        failed_cases_log: Path to the failed cases JSON log
    """
    from data_ingestion import ClearinghouseDataIngestion
    
    try:
        with open(failed_cases_log, 'r') as f:
            failed_cases = json.load(f)
    except FileNotFoundError:
        print(f"❌ No failed cases log found at {failed_cases_log}")
        return
    
    if not failed_cases:
        print("✅ No failed cases to retry")
        return
    
    print(f"🔄 Retrying {len(failed_cases)} failed cases...")
    
    ingestion = ClearinghouseDataIngestion()
    case_ids = [case["case_id"] for case in failed_cases]
    
    for idx, case_id in enumerate(case_ids, 1):
        print(f"\n[{idx}/{len(case_ids)}] Retrying case {case_id}")
        ingestion.process_case(case_id, skip_existing=False)
        time.sleep(0.3)
    
    ingestion.save_progress()
    print("\n✅ Retry complete!")


if __name__ == "__main__":
    import sys
    
    # Get cases directory from command line or use default
    if len(sys.argv) > 1:
        cases_dir = sys.argv[1]
    else:
        cases_dir = "cases"
    
    print(f"\n📁 Exploring data in: {cases_dir}")
    
    # Example usage
    explorer = DataExplorer(cases_dir=cases_dir)
    
    # Print overall statistics
    explorer.print_stats()
    
    # Validate data
    explorer.print_validation_results()
    
    # Inspect first case (if any exist)
    case_ids = explorer.get_case_ids()
    if case_ids:
        print(f"\n📋 Inspecting first case as example...")
        explorer.inspect_case(case_ids[0])
    
    print(f"\n💡 Tip: You can specify a different directory:")
    print(f"   python data_utils.py cases/training")
    print(f"   python data_utils.py cases/production")
