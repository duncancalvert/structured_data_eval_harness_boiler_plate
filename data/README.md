# Test Data Files

This folder contains test data files for the structured data evaluation harness.

## Files

- **`generated_data.csv`** / **`generated_data.xlsx`**: Sample generated data with 5 rows
- **`ground_truth.csv`** / **`ground_truth.xlsx`**: Sample ground truth data with 7 rows

## Data Structure

Both files contain the following columns:
- `name`: Person's full name
- `email`: Email address
- `address`: Full address string
- `phone`: Phone number (various formats)
- `description`: Job description or bio text

## Differences Between Generated and Ground Truth

The ground truth file has been intentionally designed to test the robustness of the evaluation metrics:

1. **Size difference**: Ground truth has 7 rows vs 5 rows in generated data
2. **Text variations**:
   - Addresses: Some use abbreviations ("St" vs "Street")
   - Phone numbers: Different formatting (spaces vs hyphens)
   - Descriptions: Minor wording differences and abbreviations ("ML" vs "machine learning", "UI" vs "user interfaces")

## Creating Excel Files

If you only have CSV files, run the creation script to generate Excel files:

```bash
uv run python data/create_test_data.py
```

Or manually convert using pandas:
```python
import pandas as pd
df = pd.read_csv('data/generated_data.csv')
df.to_excel('data/generated_data.xlsx', index=False)
```

## Usage Example

```python
from eval_harness.evaluator import StructuredDataEvaluator

evaluator = StructuredDataEvaluator(
    generated_data_path="data/generated_data.xlsx",
    ground_truth_path="data/ground_truth.xlsx",
)

results = evaluator.evaluate_all_columns()
summary = evaluator.get_summary_report(results)
print(summary)
```
