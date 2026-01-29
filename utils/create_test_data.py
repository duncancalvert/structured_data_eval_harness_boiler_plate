"""Utility function to create test Excel files for the evaluation harness.

This module generates test data files with intentional variations to test similarity metrics.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def create_test_data_files(data_dir: Optional[str] = None) -> tuple[Path, Path]:
    """Create test Excel files for the evaluation harness.

    Args:
        data_dir: Directory where data files should be created. If None, uses 'data' folder
                 relative to project root (assumes script is run from project root).

    Returns:
        Tuple of (generated_data_path, ground_truth_path)

    Raises:
        Exception: If file creation fails (e.g., missing openpyxl dependency)
    """
    # Determine data directory
    if data_dir is None:
        # Try to find project root by looking for common markers
        current_dir = Path.cwd()
        # Check if we're in project root (has eval_harness folder) or in a subdirectory
        if (current_dir / "eval_harness").exists():
            data_path = current_dir / "data"
        elif (current_dir.parent / "eval_harness").exists():
            data_path = current_dir.parent / "data"
        else:
            # Fallback: assume we're in project root
            data_path = current_dir / "data"
    else:
        data_path = Path(data_dir)

    # Ensure data directory exists
    data_path.mkdir(parents=True, exist_ok=True)

    # Create generated_data DataFrame
    generated_data = pd.DataFrame(
        {
            "key": [
                "key_001",
                "key_002",
                "key_003",
                "key_004",
                "key_005",
            ],
            "name": [
                "Alice Johnson Stevens",
                "Bob Smith Carlos",
                "Charlie Brown Green",
                "Diana Prince",
                "Edward Norton",
            ],
            "email": [
                "alice.johnson@email.com",
                "bob.smith@email.com",
                "charlie.brown@email.com",
                "diana.prince@email.com",
                "edward.norton@email.com",
            ],
            "address": [
                "123 Main Street, New York, NY 10001",
                "456 Oak Avenue, London, UK SW1A 1AA",
                "789 Pine Road, Paris, France 75001",
                "321 Elm Street, Los Angeles, CA 90001",
                "654 Maple Drive, Chicago, IL 60601",
            ],
            "phone": [
                "+1-555-0101",
                "+44-20-7946-0958",
                "+33-1-42-86-83-26",
                "+1-555-0202",
                "+1-555-0303",
            ],
            "description": [
                "Software engineer with 5 years of experience",
                "Marketing manager specializing in digital campaigns",
                "Data scientist working on machine learning projects",
                "Product designer creating user interfaces",
                "Business analyst focusing on data insights",
            ],
        }
    )

    # Create ground_truth DataFrame with some variations
    # Ground truth has more rows and some slight differences to test robustness
    ground_truth = pd.DataFrame(
        {
            "key": [
                "key_001",  # Matches Alice Johnson
                "key_002",  # Matches Bob Smith
                "key_003",  # Matches Charlie Brown
                "key_004",  # Matches Diana Prince
                "key_005",  # Matches Edward Norton
                "key_006",  # Extra row in ground truth (no match in generated)
                "key_007",  # Extra row in ground truth (no match in generated)
            ],
            "name": [
                "Alice Johnson",  # Exact match
                "Bob Smith III",  # Exact match
                "Charlie Brown Eq",  # Exact match
                "Diana Prince Junior",  # Exact match
                "Edward Norton",  # Exact match
                "Frank Miller Jones",  # Extra row in ground truth
                "Grace Lee",  # Extra row in ground truth
            ],
            "email": [
                "alice.johnson@email.com",  # Exact match
                "bob.smith@email.com",  # Exact match
                "charlie.brown@email.com",  # Exact match
                "diana.prince@email.com",  # Exact match
                "edward.norton@email.com",  # Exact match
                "frank.miller@email.com",  # Extra row
                "grace.lee@email.com",  # Extra row
            ],
            "address": [
                "123 Main St, New York, NY 10003",  # Slight variation (St vs Street)
                "456 Oak Avenue, San Francisco, USA SW1A 1AA",  # Exact match
                "789 Pine Road, Paris, France 75001",  # Exact match
                "321 Elm Street, Los Angeles, CA 90001",  # Exact match
                "654 Maple Drive, Chicago, IL 60601",  # Exact match
                "987 Cedar Lane, Boston, MA 02101",  # Extra row
                "147 Birch Court, Seattle, WA 98101",  # Extra row
            ],
            "phone": [
                "+1-555-0101",  # Exact match
                "+44 20 7946 0958",  # Different format (spaces vs hyphens)
                "+33-1-42-86-83-26",  # Exact match
                "+1-555-0202",  # Exact match
                "+1-555-0303",  # Exact match
                "+1-555-0404",  # Extra row
                "+1-555-0505",  # Extra row
            ],
            "description": [
                "Software engineer with 5 years experience",  # Slight variation (no "of")
                "Marketing manager specializing in digital marketing campaigns",  # Variation
                "Data scientist working on ML projects",  # Abbreviation (ML vs machine learning)
                "Product designer creating UI",  # Abbreviation (UI vs user interfaces)
                "Business analyst focusing on data insights",  # Exact match
                "Consultant providing strategic advice",  # Extra row
                "Developer building web applications",  # Extra row
            ],
        }
    )

    # Save Excel files
    generated_xlsx_path = data_path / "generated_data.xlsx"
    ground_truth_xlsx_path = data_path / "ground_truth.xlsx"

    try:
        generated_data.to_excel(generated_xlsx_path, index=False)
        print(f"Created {generated_xlsx_path} ({len(generated_data)} rows)")
    except Exception as e:
        error_msg = f"Error creating Excel file {generated_xlsx_path}: {e}"
        if "openpyxl" in str(e).lower():
            error_msg += "\nMake sure openpyxl is installed: pip install openpyxl"
        raise Exception(error_msg) from e

    try:
        ground_truth.to_excel(ground_truth_xlsx_path, index=False)
        print(f"Created {ground_truth_xlsx_path} ({len(ground_truth)} rows)")
    except Exception as e:
        error_msg = f"Error creating Excel file {ground_truth_xlsx_path}: {e}"
        if "openpyxl" in str(e).lower():
            error_msg += "\nMake sure openpyxl is installed: pip install openpyxl"
        raise Exception(error_msg) from e

    print("\n" + "=" * 60)
    print("Test data files created successfully!")
    print("=" * 60)
    print("\nFiles created:")
    print(f"  - {generated_xlsx_path} ({len(generated_data)} rows)")
    print(f"  - {ground_truth_xlsx_path} ({len(ground_truth)} rows)")
    print("\nNote: Ground truth has some intentional variations to test similarity metrics:")
    print("  - Some addresses use abbreviations (St vs Street)")
    print("  - Phone numbers have different formatting")
    print("  - Descriptions have minor wording differences")
    print("  - Ground truth has 2 extra rows (Frank Miller, Grace Lee)")
    print("\nKey column:")
    print("  - Each row has a unique 'key' column (key_001, key_002, etc.)")
    print("  - Matching keys between datasets indicate corresponding rows")
    print("  - Keys key_001 through key_005 match between datasets")
    print("  - Keys key_006 and key_007 exist only in ground truth")
    print("\nUse KEY_COLUMN='key' in the evaluation notebook for key-based matching.")

    return generated_xlsx_path, ground_truth_xlsx_path


if __name__ == "__main__":
    # Allow running as a script
    create_test_data_files()
