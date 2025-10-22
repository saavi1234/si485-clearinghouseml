import requests
import time

# Case type collections
case_type_dict = {
    5033: "Mental Health (Facility)",
    5034: "Prison Conditions",
    5035: "Nursing Home Conditions",
    5036: "Child Welfare",
    5038: "Intellectual Disability (Facility)",
    5039: "Policing",
    5041: "School Desegregation",
    5042: "Election/Voting Rights",
    5043: "Immigration and/or the Border",
    5044: "Disability Rights",
    5045: "Equal Employment",
    5046: "Criminal Justice (Other)",
    5047: "Indigent Defense",
    5048: "Fair Housing/Lending/Insurance",
    5049: "Education",
    5050: "Speech and Religious Freedom",
    5051: "Public Benefits/Government Services",
    5052: "Public Accommodations/Contracting",
    5053: "National Security",
    5054: "Presidential/Gubernatorial Authority",
    5055: "Environmental Justice",
    5056: "Jail Conditions",
    5057: "Public Housing",
    5058: "Juvenile Institution",
    7811: "Reproductive Issues"
}

headers = {
    'Authorization': 'Token 11a7d2a0a5c673e0d4391cb578563eb43d629a49',
    'User-Agent': 'Chrome v22.2 Linux Ubuntu'
}

def fetch_cases_for_type(type_code, type_name, delay=1.0, max_cases=50, existing_ids=None):
    """
    Fetch cases for a specific case type (limited to max_cases DISTINCT case IDs).
    Includes retry logic and delay to avoid rate limiting.
    If existing_ids is provided, will try to fetch max_cases NEW cases not in existing_ids.
    """
    url = f"https://clearinghouse.net/api/v2/case/?case_type={type_code}"
    local_records = []
    seen_case_ids = set()  # Track distinct case IDs
    if existing_ids is None:
        existing_ids = set()
    page_num = 0
    max_retries = 5
    
    print(f"Fetching cases for: {type_name} (ID: {type_code})")
    
    while url and len(seen_case_ids) < max_cases:
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Add delay between requests to avoid rate limiting
                if page_num > 0 or retry_count > 0:
                    time.sleep(delay)
                
                response = requests.get(url, headers=headers, timeout=60)
                
                if response.status_code == 429:  # Rate limited
                    wait_time = int(response.headers.get('Retry-After', 10))
                    print(f"  Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                
                if response.status_code == 504:  # Gateway timeout
                    print(f"  Server timeout (504), retry {retry_count + 1}/{max_retries}")
                    retry_count += 1
                    time.sleep(5)
                    continue
                    
                if response.status_code != 200:
                    print(f"  Error {response.status_code} for {type_name}")
                    retry_count += 1
                    time.sleep(3)
                    continue
                
                data = response.json()
                results = data.get("results", [])
                
                # Extract minimal case information - only distinct case IDs
                for case in results:
                    case_id = case.get("id", "")
                    # Only add if we haven't seen this case ID before AND it's not in existing_ids
                    if case_id and case_id not in seen_case_ids and case_id not in existing_ids:
                        seen_case_ids.add(case_id)
                        local_records.append(case_id)
                        
                        # Stop if we have enough distinct NEW cases
                        if len(seen_case_ids) >= max_cases:
                            break
                
                page_num += 1
                url = data.get("next")
                
                # Stop if we have enough cases
                if len(seen_case_ids) >= max_cases:
                    url = None
                
                break  # Success, exit retry loop
                
            except requests.exceptions.Timeout:
                retry_count += 1
                print(f"  Timeout for {type_name}, retry {retry_count}/{max_retries}")
                time.sleep(5)
            except Exception as e:
                print(f"  Error for {type_name}: {e}")
                retry_count += 1
                time.sleep(3)
        
        if retry_count >= max_retries:
            print(f"  Max retries reached for {type_name}, skipping remaining pages")
            break
    
    print(f"  ✓ Completed {type_name}: {len(seen_case_ids)} distinct cases")
    return local_records

# Fetch cases sequentially with delays (safer than parallel for rate limiting)
print("=" * 60)
print("Fetching cases by type from Clearinghouse API")
print("Collecting 50 distinct cases per type")
print("=" * 60)
print()

case_ids = set()  # Use set to collect unique case IDs across all types
total_types = len(case_type_dict)

for idx, (type_code, type_name) in enumerate(case_type_dict.items(), 1):
    print(f"\n[{idx}/{total_types}] Processing: {type_name}")
    try:
        # Fetch 50 new unique cases that aren't already in our set
        type_case_ids = fetch_cases_for_type(type_code, type_name, delay=1.5, max_cases=50, existing_ids=case_ids)
        
        # Add all new cases to our set
        case_ids.update(type_case_ids)
        print(f"  New unique cases from this type: {len(type_case_ids)}")
        print(f"  Total unique case IDs collected so far: {len(case_ids)}")
        
        # Add a delay between different case types to be extra safe
        if idx < total_types:
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user.")
        break
    except Exception as e:
        print(f"  Unexpected error: {e}")
        continue

print("\n" + "=" * 60)
print("Data Collection Complete")
print("=" * 60)

# Create text file with case IDs
if len(case_ids) > 0:
    output_file = "case_ids.txt"
    # Convert set to sorted list for consistent ordering
    case_ids_list = sorted(list(case_ids))
    
    with open(output_file, 'w') as f:
        for case_id in case_ids_list:
            f.write(f"{case_id}\n")
    
    print(f"\nTotal unique case IDs collected: {len(case_ids)}")
    print(f"\n✅ Saved {len(case_ids)} unique case IDs to {output_file}")
    
    # Show first few IDs
    print("\nFirst 10 case IDs:")
    for i, case_id in enumerate(case_ids_list[:10], 1):
        print(f"  {i}. {case_id}")
else:
    print("\n⚠️  No case IDs collected. Check API authentication or connection.")
