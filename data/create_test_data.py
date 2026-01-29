"""Script to create test Excel files for the evaluation harness.

This script reads the existing CSV files and converts them to Excel format.

Run this script after installing dependencies:
    uv sync
    uv run python data/create_test_data.py
"""

import pandas as pd
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read CSV files and convert to Excel
generated_data = pd.read_csv(os.path.join(script_dir, "generated_data.csv"))
generated_data.to_excel(os.path.join(script_dir, "generated_data.xlsx"), index=False)
print("Created data/generated_data.xlsx")

ground_truth = pd.read_csv(os.path.join(script_dir, "ground_truth.csv"))
ground_truth.to_excel(os.path.join(script_dir, "ground_truth.xlsx"), index=False)
print("Created data/ground_truth.xlsx")

print("\nTest data files created successfully!")
print("\nFiles created:")
print("  - data/generated_data.xlsx (5 rows)")
print("  - data/ground_truth.xlsx (7 rows - includes 2 extra rows)")
print("  - data/generated_data.csv")
print("  - data/ground_truth.csv")
print("\nNote: Ground truth has some intentional variations to test similarity metrics:")
print("  - Some addresses use abbreviations (St vs Street)")
print("  - Phone numbers have different formatting")
print("  - Descriptions have minor wording differences")
