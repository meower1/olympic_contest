#!/usr/bin/env python3
"""
Test script for CSVAnalyzer tool.
"""

import sys
sys.path.insert(0, '/home/moz/Projects/olympic25/olympic_contest')

from solution import CSVAnalyzer
import json


def test_csv_analyzer():
    """Test the CSVAnalyzer with various operations."""
    
    analyzer = CSVAnalyzer(max_preview_rows=3)
    
    # Test with a sample CSV URL (you can replace with actual test URL)
    test_url = "https://raw.githubusercontent.com/datasets/gdp/master/data/gdp.csv"
    
    print("=" * 80)
    print("Testing CSVAnalyzer Tool")
    print("=" * 80)
    
    # Test 1: Summarize
    print("\n📊 Test 1: Summarize CSV")
    print("-" * 80)
    summary = analyzer.summarize(test_url)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    # Test 2: Query
    print("\n❓ Test 2: Query CSV")
    print("-" * 80)
    questions = [
        "How many rows are in the CSV?",
        "What are the columns?",
        "Show me sample data"
    ]
    for q in questions:
        print(f"\nQuestion: {q}")
        answer = analyzer.query(test_url, q)
        print(f"Answer: {answer}")
    
    # Test 3: Lookup (if applicable)
    print("\n🔍 Test 3: Lookup by ID")
    print("-" * 80)
    if summary.get('headers') and len(summary['headers']) > 0:
        first_field = summary['headers'][0]
        print(f"Looking up using field: {first_field}")
        
        # Get some sample values
        if summary.get('sample_rows'):
            sample_values = [row.get(first_field) for row in summary['sample_rows'][:2]]
            lookup_result = analyzer.lookup(test_url, first_field, sample_values)
            print(json.dumps(lookup_result, indent=2, ensure_ascii=False))
    
    # Test 4: Generic run method
    print("\n🚀 Test 4: Generic run() method")
    print("-" * 80)
    result = analyzer.run(operation="summarize", source=test_url)
    print(f"Headers: {result.get('headers')}")
    print(f"Row count: {result.get('row_count')}")
    
    print("\n" + "=" * 80)
    print("✅ All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_csv_analyzer()
