#!/usr/bin/env python3
"""
Advanced test for CSVAnalyzer with ID-based lookups.
"""

import sys
sys.path.insert(0, '/home/moz/Projects/olympic25/olympic_contest')

from solution import CSVAnalyzer
import json


def test_id_lookup():
    """Test ID-based lookup functionality."""
    
    analyzer = CSVAnalyzer(max_preview_rows=5)
    
    # Test URL with actual IDs
    test_url = "https://raw.githubusercontent.com/datasets/gdp/master/data/gdp.csv"
    
    print("=" * 80)
    print("Testing CSVAnalyzer - ID Lookup Scenarios")
    print("=" * 80)
    
    # Scenario 1: Lookup by Country Code (acts as ID)
    print("\n🔍 Scenario 1: Lookup specific countries by Country Code")
    print("-" * 80)
    countries_to_find = ["USA", "IRN", "CHN", "DEU"]
    result = analyzer.lookup(test_url, "Country Code", countries_to_find)
    
    print(f"Requested IDs: {countries_to_find}")
    print(f"Matches found: {result['match_count']}")
    print(f"Field used: {result.get('field_used')}")
    
    if result.get('matched_records'):
        print("\nSample matched records:")
        for record in result['matched_records'][:3]:
            print(f"  {record}")
    
    if result.get('error'):
        print(f"Error: {result['error']}")
    
    # Scenario 2: Case-insensitive field name
    print("\n\n🔍 Scenario 2: Case-insensitive field lookup")
    print("-" * 80)
    result2 = analyzer.lookup(test_url, "country code", ["USA"])  # lowercase field name
    print(f"Field used: {result2.get('field_used')}")
    print(f"Matches: {result2['match_count']}")
    
    # Scenario 3: Non-existent field
    print("\n\n🔍 Scenario 3: Non-existent field")
    print("-" * 80)
    result3 = analyzer.lookup(test_url, "NonExistentField", ["test"])
    print(f"Error: {result3.get('error')}")
    
    # Scenario 4: Query for general information
    print("\n\n❓ Scenario 4: General queries about the data")
    print("-" * 80)
    questions = [
        "How many rows?",
        "What columns exist?",
        "How many columns are there?"
    ]
    
    for question in questions:
        answer = analyzer.query(test_url, question)
        print(f"Q: {question}")
        print(f"A: {answer}\n")
    
    # Scenario 5: Combined workflow - first summarize, then lookup
    print("\n📊 Scenario 5: Complete workflow")
    print("-" * 80)
    print("Step 1: Get summary")
    summary = analyzer.summarize(test_url)
    print(f"  - Total rows: {summary['row_count']}")
    print(f"  - Columns: {summary['headers']}")
    
    print("\nStep 2: Lookup based on discovered structure")
    if 'Country Code' in summary['headers']:
        lookup_result = analyzer.lookup(test_url, "Country Code", ["IRN", "IRQ"])
        print(f"  - Found {lookup_result['match_count']} records for Iran and Iraq")
        
        if lookup_result.get('matched_records'):
            # Show unique years for these countries
            years = set()
            for record in lookup_result['matched_records']:
                years.add(record.get('Year', ''))
            print(f"  - Years covered: {sorted(years)[:10]}")
    
    print("\n" + "=" * 80)
    print("✅ All ID lookup scenarios tested!")
    print("=" * 80)


def test_with_sample_data():
    """Test with inline CSV data (simulated)."""
    print("\n\n" + "=" * 80)
    print("Testing with Sample Scenarios")
    print("=" * 80)
    
    analyzer = CSVAnalyzer()
    
    # Example: User prompt might say "find person with ID 123"
    print("\n📝 Example use case:")
    print("Prompt: 'در فایل CSV به آدرس [URL]، اطلاعات شخص با ID برابر 456 را پیدا کن'")
    print("\nHow agent would handle:")
    print("1. Extract URL from prompt")
    print("2. Extract field name: 'ID'")
    print("3. Extract value: '456'")
    print("4. Call: analyzer.lookup(url, 'ID', ['456'])")
    print("5. Return matched record(s)")


if __name__ == "__main__":
    test_id_lookup()
    test_with_sample_data()
