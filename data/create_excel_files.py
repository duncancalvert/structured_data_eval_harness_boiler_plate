"""Simple script to create Excel files from CSV files using openpyxl."""

from openpyxl import Workbook
import csv
import os

def csv_to_excel(csv_path, excel_path):
    """Convert CSV file to Excel file."""
    wb = Workbook()
    ws = wb.active
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            ws.append(row)
    
    wb.save(excel_path)
    print(f"Created {excel_path}")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Convert CSV files to Excel
csv_to_excel(
    os.path.join(script_dir, "generated_data.csv"),
    os.path.join(script_dir, "generated_data.xlsx")
)

csv_to_excel(
    os.path.join(script_dir, "ground_truth.csv"),
    os.path.join(script_dir, "ground_truth.xlsx")
)

print("\nExcel files created successfully!")
