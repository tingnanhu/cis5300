# CIS 5300 Final Project: Toxic Comment Classification

## Overview

This project implements and evaluates multiple baseline models for multi-label toxic comment classification. The task is to classify online comments into six categories of toxicity: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. This is based on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) from Kaggle.

## Task Description

**Multi-label Classification Problem**: Each comment can be assigned to multiple toxic categories simultaneously. The six categories are:

1. `toxic` - General toxic comments
2. `severe_toxic` - Severely toxic comments
3. `obscene` - Obscene language
4. `threat` - Threatening language
5. `insult` - Insulting language
6. `identity_hate` - Identity-based hate speech

## Project Structure

```
cis5300/
├── cleaned/                    # Data splits
│   ├── train_split.csv         # Training data
│   ├── dev_split.csv           # Development/validation data
│   └── test_split.csv          # Test data
├── results/                    # Model predictions and evaluation results
│   ├── simple_baseline_predictions.csv
│   ├── strong_baseline_predictions.csv
│   └── RESULTS_SUMMARY.md      # Detailed results summary
├── reports/                    # Project reports and proposals
├── simple-baseline.py          # Simple baseline implementation
├── simple-baseline.md          # Simple baseline documentation
├── strong_baseline.py          # Strong baseline implementation (DistilBERT)
├── strong-baseline.md          # Strong baseline documentation
├── score.py                    # Evaluation script
├── scoring.md                  # Evaluation metric documentation
├── requirements.txt            # Python dependencies
└── .gitignore                  # Git ignore file
```

## Installation

### Prerequisites

- Python 3.7+
- pip

### Setup

1. Clone or navigate to the project directory:
```bash
cd cis5300
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.21.0` - Numerical operations
- `scikit-learn>=1.0.0` - Machine learning utilities
- `torch>=1.9.0` - PyTorch for deep learning models
- `transformers>=4.20.0` - Hugging Face transformers library
- `tqdm>=4.60.0` - Progress bars

## Data Format

### Input Files

All data files are CSV format with the following columns:

- `id`: Unique identifier for each comment
- `comment_text`: The text of the comment
- `toxic`: Binary label (0 or 1)
- `severe_toxic`: Binary label (0 or 1)
- `obscene`: Binary label (0 or 1)
- `threat`: Binary label (0 or 1)
- `insult`: Binary label (0 or 1)
- `identity_hate`: Binary label (0 or 1)

**Note**: Test files may not contain label columns.

### Data Splits

- **Training set**: `cleaned/train_split.csv` (127,656 examples)
- **Development set**: `cleaned/dev_split.csv` (15,959 examples)
- **Test set**: `cleaned/test_split.csv`

## Models

### 1. Simple Baseline

**Implementation**: `simple-baseline.py`

A majority class baseline that predicts the most common class for each toxic category based on training data distribution.

**Usage**:
```bash
python simple-baseline.py <train_file> <test_file> <output_file> [--dev-file <dev_file>]
```

**Example**:
```bash
python simple-baseline.py cleaned/train_split.csv cleaned/test_split.csv results/simple_baseline_predictions.csv --dev-file cleaned/dev_split.csv
```

**Performance**: Mean AUC-ROC ≈ 0.50 (random performance)

**Documentation**: See `simple-baseline.md` for detailed documentation.

### 2. Strong Baseline

**Implementation**: `strong_baseline.py`

A transformer-based baseline using fine-tuned DistilBERT (a distilled version of BERT) for toxic comment classification. DistilBERT is faster and smaller than BERT while maintaining most of its performance. The model uses focal loss to handle class imbalance and supports hyperparameter tuning.

**Usage**:
```bash
python strong_baseline.py <train_file> <test_file> <output_file> [options]
```

**Example**:
```bash
# Basic usage
python strong_baseline.py cleaned/train_split.csv cleaned/test_split.csv results/strong_baseline_predictions.csv

# With dev set evaluation and best model saving
python strong_baseline.py cleaned/train_split.csv cleaned/test_split.csv results/strong_baseline_predictions.csv --dev-file cleaned/dev_split.csv --save-best

# With hyperparameter tuning
python strong_baseline.py cleaned/train_split.csv cleaned/test_split.csv results/strong_baseline_predictions.csv --dev-file cleaned/dev_split.csv --tune
```

**Performance**: Mean AUC-ROC ≈ 0.99 on test set

**Documentation**: See `strong-baseline.md` for detailed documentation including all command-line options, hyperparameter tuning, and checkpointing.

## Evaluation

### Evaluation Metric

The primary evaluation metric is **Mean Column-wise AUC-ROC** (Area Under the Receiver Operating Characteristic Curve). This metric:

- Computes AUC-ROC separately for each of the six label columns
- Averages the six scores to get a single metric
- Is threshold-independent and handles class imbalance well

For more details, see `scoring.md`.

### Running Evaluation

```bash
python score.py <gold_file> <pred_file>
```

**Example**:
```bash
python score.py cleaned/test_split.csv results/simple_baseline_predictions.csv
```

### Output

The evaluation script outputs:
- **Mean AUC-ROC**: Primary metric
- **Macro-F1**: Macro-averaged F1 score
- **Macro-Precision**: Macro-averaged precision
- **Macro-Recall**: Macro-averaged recall
- **Individual AUC-ROC scores**: Per-label scores
- **Metrics JSON file**: `{pred_file}_metrics.json`

**Example Output**:
```
Mean AUC-ROC: 0.978787
Macro-F1: 0.623456
Macro-Precision: 0.712345
Macro-Recall: 0.556789

Individual AUC-ROC scores:
  toxic: 0.967539
  severe_toxic: 0.977336
  obscene: 0.984031
  threat: 0.987609
  insult: 0.978092
  identity_hate: 0.978117

Metrics saved to results/simple_baseline_predictions_metrics.json
```

## Results Summary

### Performance Comparison

| Baseline | Dev AUC-ROC | Test AUC-ROC | Improvement |
|----------|-------------|--------------|-------------|
| Simple (Majority Class) | 0.500000 | 0.500000 | Baseline |
| Strong Baseline | - | 0.992305 | +98.5% |

### Key Findings

1. **Simple Baseline**: Achieves random performance (AUC-ROC = 0.5), establishing a lower bound for model performance.

2. **Strong Baseline**: Achieves state-of-the-art performance (~0.99 AUC-ROC), showing the power of transformer-based models for toxic comment classification.

For detailed results, see `results/RESULTS_SUMMARY.md`.

## Usage Examples

### Running Simple Baseline

```bash
# Basic usage
python simple-baseline.py \
    cleaned/train_split.csv \
    cleaned/test_split.csv \
    results/simple_baseline_predictions.csv

# With dev set evaluation
python simple-baseline.py \
    cleaned/train_split.csv \
    cleaned/test_split.csv \
    results/simple_baseline_predictions.csv \
    --dev-file cleaned/dev_split.csv
```

### Running Strong Baseline

```bash
# Basic usage with default hyperparameters
python strong_baseline.py \
    cleaned/train_split.csv \
    cleaned/test_split.csv \
    results/strong_baseline_predictions.csv

# With dev set evaluation and best model saving
python strong_baseline.py \
    cleaned/train_split.csv \
    cleaned/test_split.csv \
    results/strong_baseline_predictions.csv \
    --dev-file cleaned/dev_split.csv \
    --save-best

# With hyperparameter tuning (requires dev set)
python strong_baseline.py \
    cleaned/train_split.csv \
    cleaned/test_split.csv \
    results/strong_baseline_predictions.csv \
    --dev-file cleaned/dev_split.csv \
    --tune

# With custom hyperparameters
python strong_baseline.py \
    cleaned/train_split.csv \
    cleaned/test_split.csv \
    results/strong_baseline_predictions.csv \
    --epochs 5 \
    --batch-size 32 \
    --learning-rate 3e-5
```

### Evaluating Predictions

```bash
# Evaluate simple baseline
python score.py \
    cleaned/test_split.csv \
    results/simple_baseline_predictions.csv

# Evaluate strong baseline
python score.py \
    cleaned/test_split.csv \
    results/strong_baseline_predictions.csv
```

## File Descriptions

- **`simple-baseline.py`**: Implementation of majority class baseline
- **`simple-baseline.md`**: Detailed documentation for simple baseline
- **`strong_baseline.py`**: Implementation of DistilBERT-based strong baseline
- **`strong-baseline.md`**: Detailed documentation for strong baseline including usage, hyperparameters, and checkpointing
- **`score.py`**: Evaluation script for computing Mean AUC-ROC and other metrics
- **`scoring.md`**: Documentation explaining the evaluation metric
- **`requirements.txt`**: Python package dependencies
- **`results/RESULTS_SUMMARY.md`**: Comprehensive results summary and analysis
- **`.gitignore`**: Git ignore file for Python projects

## References

1. **Kaggle Competition**: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)

2. **Evaluation Metric**: 
   - Fawcett, T. (2006). "An introduction to ROC analysis". *Pattern Recognition Letters*, 27(8), 861-874.

3. **Multi-label Classification**:
   - Tsoumakas, G., & Katakis, I. (2007). "Multi-label classification: An overview". *International Journal of Data Warehousing and Mining*, 3(3), 1-13.

## Notes

- All prediction files are in CSV format with columns: `id`, `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- Simple baseline predictions are binary (0 or 1)
- Strong baseline predictions are probabilities (0.0 to 1.0)
- The evaluation script expects probabilities for optimal AUC-ROC computation, but will work with binary predictions

## Contact

For questions or issues, please refer to the course materials or contact the course instructor.

