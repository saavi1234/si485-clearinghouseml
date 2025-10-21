# Clearinghouse ML Project - Data Ingestion

This directory contains scripts for fetching and managing legal case data from the Civil Rights Clearinghouse Litigation database.

## 📁 Project Structure

```
.
├── data_ingestion.py           # Main script to fetch cases and documents from API
├── data_utils.py               # Utility functions for exploring and validating data
├── cases/                      # Case data organized by purpose
│   ├── training/               # Human-tagged cases for ML training/testing
│   │   ├── case_12345.json
│   │   ├── case_12346.json
│   │   └── ...
│   └── production/             # New cases to be tagged by ML models
│       ├── case_45678.json
│       ├── case_45679.json
│       └── ...
├── failed_cases.json           # Log of cases that failed to fetch (if any)
├── ingestion_progress.json     # Progress tracking and summary
└── clearinghouse_case_links_by_collections(without 5679 and 5666).csv
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install requests pandas
```

### 2. Run Data Ingestion

```bash
# Fetch training data
python data_ingestion.py --csv training_cases.csv --output-dir training

# Fetch production data
python data_ingestion.py --csv production_cases.csv --output-dir production

# Re-fetch to check for updates
python data_ingestion.py --csv production_cases.csv --output-dir production --refetch
```

**Features:**
- ✅ **Automatic retry** with exponential backoff for rate limits and server errors
- ✅ **Resume capability** - skips already-fetched cases
- ✅ **Change detection** - re-fetch mode only updates changed cases
- ✅ **Separate training/production** - organize data by purpose
- ✅ **Progress tracking** - saves detailed logs
- ✅ **Error handling** - gracefully handles failures and logs them

### 3. Explore the Data

```bash
# Explore training data
python data_utils.py cases/training

# Explore production data
python data_utils.py cases/production
```

This will display:
- Dataset statistics (total cases, documents, etc.)
- Data validation results
- Detailed inspection of a sample case

## 📊 Data Structure

Each case file (`cases/case_XXXXX.json`) has this structure:

```json
{
  "case_id": 12345,
  "fetched_at": "2025-10-20T10:30:00",
  "case_data": {
    "name": "Case Name",
    "court": "Court Name",
    "facility_type": [...],
    "causes": [...],
    "constitutional_clause": [...],
    "case_defendants": [...],
    ... (all case-level fields and tags)
  },
  "documents": [
    {
      "title": "Document Title",
      "document_type": "Document Type",
      "text": "Full document text...",
      ... (all document-level fields and tags)
    },
    ...
  ],
  "document_count": 5
}
```

## 🔧 Advanced Usage

### Command-Line Arguments

```bash
python data_ingestion.py --help
```

**Available options:**
- `--csv PATH` - Specify CSV file with case links
- `--output-dir DIR` - Specify output directory (e.g., "training" or "production")
- `--refetch` - Check for updates and re-fetch changed cases
- `--rate-limit SECONDS` - Set delay between requests (default: 1.0)
- `--no-skip` - Force re-fetch all cases (overwrites existing)

### Re-fetch Updated Cases

Legal cases can change over time (new documents, status updates, etc.). Use `--refetch` to check for and update changed cases:

```bash
python data_ingestion.py --csv production_cases.csv --output-dir production --refetch
```

This will:
- Fetch all cases from the API
- Compare with existing local files
- Only save cases that have changed
- Skip unchanged cases (faster!)

### Force Re-fetch Everything

To completely re-download all cases (overwrites existing):

```bash
python data_ingestion.py --csv cases.csv --output-dir training --no-skip
```

### Adjust Rate Limiting

If you're hitting rate limits:

```bash
python data_ingestion.py --csv cases.csv --output-dir training --rate-limit 2.0
```

### Load and Analyze Data

```python
from data_utils import DataExplorer

explorer = DataExplorer()

# Get statistics
stats = explorer.get_stats()
print(f"Total documents: {stats['total_documents']}")

# Load a specific case
case = explorer.load_case(12345)

# Inspect a case in detail
explorer.inspect_case(12345)

# Get all case IDs
case_ids = explorer.get_case_ids()

# Get tag distribution for analysis
court_distribution = explorer.get_tag_distribution('court', level='case')
print(court_distribution)
```

## 📝 API Endpoints Used

- **Cases:** `https://clearinghouse.net/api/v2/case/?case_id={case_id}`
  - Returns case metadata, tags, and summary
  
- **Documents:** `https://clearinghouse.net/api/v2/documents/?case={case_id}`
  - Returns all documents for a case with full text and tags

## ⚙️ Configuration

Edit these variables in `data_ingestion.py`:

```python
OUTPUT_DIR = "cases"           # Where to save case files
CSV_PATH = "..."               # Path to CSV with case links
FAILED_CASES_LOG = "..."       # Where to log failures
```

## 🎯 Next Steps for ML Pipeline

Once data ingestion is complete:

1. **Data Preparation**: Create train/test splits
2. **Feature Extraction**: Extract text features from case and document bodies
3. **Label Encoding**: Convert tags to ML-compatible format
4. **Model Training**: Train separate models for case-level and document-level tagging

## 📧 Troubleshooting

**Rate Limiting Issues:**
- Increase `rate_limit_delay` in `data_ingestion.py`
- The script automatically retries with exponential backoff

**Missing Documents:**
- Some cases legitimately have no documents
- Check `data_utils.print_validation_results()` to see which cases

**Failed Cases:**
- Check `failed_cases.json` for details
- Use `retry_failed_cases()` to retry them

**Memory Issues:**
- Load cases one at a time instead of using `load_all_cases()`
- Process cases in batches for ML training

## 📄 License

This project is for educational/research purposes using the Clearinghouse API.
