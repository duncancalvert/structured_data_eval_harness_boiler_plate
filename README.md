# Structured Data Eval Harness

A Python evaluation harness for comparing generated structured data against ground truth data using various NLP text distance metrics.

## Features
* Multiple NLP Metrics: 
   * Cosine similarity
   * Levenshtein distance
   * Edit distance
   * Jaccard similarity
* Robust Data Handling: Handles mismatched dataset sizes, missing values, and different column names
* Flexible Matching Strategies: Index-based, all-pairs, or truncate matching

## Installation

### Prerequisites

* Python 3.9 or higher
* [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/duncancalvert/structured_data_eval_harness_boiler_plate.git
```

2. **Create and activate a virtual env**
```bash
python3 -m venv .venv
```

3. **Install uv:**
```bash
pip install uv
```

4. **Install dependencies** 

PROD Dependencies
```bash
uv sync
```
PROD + DEV dependecies
```bash
uv sync --extra dev
```

4. **Set up environment variables:**
```bash
cp .env.example .env
```

Edit `.env` and update the paths to your data files:

```bash
GENERATED_DATA_PATH=path/to/your/generated_data.xlsx
GROUND_TRUTH_PATH=path/to/your/ground_truth.xlsx
```

## Usage

### Using the Eval Harness

1. **Configure paths** in the first cell or use environment variables from `.env`

### Using Python Scripts

```python
from eval_harness.evaluator import StructuredDataEvaluator

# Initialize evaluator
evaluator = StructuredDataEvaluator(
    generated_data_path="generated_data.xlsx",
    ground_truth_path="ground_truth.xlsx",
    column_mapping=None,  # Optional: {"gen_col": "gt_col"}
    match_strategy="index",  # "index", "all_pairs", or "truncate"
)

# Evaluate all columns
results = evaluator.evaluate_all_columns()

# Get summary report
summary_df = evaluator.get_summary_report(results)
print(summary_df)

# Evaluate specific column
column_results = evaluator.evaluate_column("column_name")

# Evaluate with specific metrics only
results = evaluator.evaluate_all_columns(
    metrics=["cosine_similarity", "edit_distance"]
)
```

## Data Format

### Generated Data File

Your `generated_data` file should be a CSV or Excel file with columns containing the data you want to evaluate:

```
| name    | age | city      |
|---------|-----|-----------|
| Alice   | 25  | New York  |
| Bob     | 30  | London    |
| Charlie | 35  | Paris     |
```

### Ground Truth File

Your `ground_truth` file should have corresponding columns (can have different names if you provide a mapping):

```
| name    | age | city      |
|---------|-----|-----------|
| Alice   | 25  | New York  |
| Bob     | 30  | London    |
| Charlie | 35  | Paris     |
| David   | 40  | Tokyo     |  # Extra rows are OK
```

**Note**: The ground truth file can be larger or smaller than the generated data file. The evaluator handles size mismatches gracefully.

## Matching Strategies

- **`index`** (default): Matches rows by index position. Missing values are handled with NaN.
- **`all_pairs`**: Compares every generated value with every ground truth value (useful for finding best matches).
- **`truncate`**: Truncates the longer dataset to match the shorter one.

## Available Metrics

- **Cosine Similarity**: Measures similarity using TF-IDF vectorization
- **Levenshtein Distance**: Number of character edits needed
- **Edit Distance**: Normalized edit distance (0-1 scale)
- **Jaccard Similarity**: Set-based similarity using n-grams (unigrams and bigrams)

## Development

### Running Tests

```bash
pytest
```

Or with coverage:
```bash
pytest --cov=utils --cov=eval_harness
```

### Linting

```bash
ruff check .
```

### Formatting

```bash
ruff format .
```

## Configuration

### Ruff Configuration

Ruff is configured in `pyproject.toml`. Key settings:
- Line length: 100
- Target Python version: 3.9+
- Selected rules: pycodestyle, pyflakes, isort, bugbear, comprehensions, pyupgrade

### Environment Variables

See `.env.example` for available environment variables. The main ones are:
- `GENERATED_DATA_PATH`: Path to your generated data file
- `GROUND_TRUTH_PATH`: Path to your ground truth file
