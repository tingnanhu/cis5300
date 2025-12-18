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
├── code/                       # All training and evaluation scripts
│   ├── simple_baseline.py      # Majority class baseline
│   ├── simple_baseline.md      # Simple baseline documentation
│   ├── strong_baseline.py      # DistilBERT baseline
│   ├── strong_baseline.md      # Strong baseline documentation
│   ├── distilbert.py           # DistilBERT with focal loss
│   ├── distilbert.md           # DistilBERT documentation
│   ├── roberta.py              # RoBERTa with focal loss
│   ├── roberta.md              # RoBERTa documentation
│   ├── score.py                # Evaluation script
│   ├── score.md                # Evaluation metrics documentation
│   └── README.md               # Step-by-step execution guide
├── data/                       # Data files
│   ├── train_split.csv         # Training data
│   ├── dev_split.csv           # Development/validation data
│   └── test_split.csv          # Test data (with gold labels)
├── output/                     # Model predictions
│   ├── roberta_tuned_pred.csv
│   ├── distilbert_tuned_pred.csv
│   ├── strong_baseline_pred.csv
│   └── README.md               # Evaluation guide
├── logs/                       # Training and tuning logs
│   ├── hparam_tuning_roberta.log
│   └── hparam_tuning_distilbert.log
├── reports/                    # Project reports and proposals
├── requirements.txt            # Python dependencies
└── README.md                   # This file
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

Install required packages:

```bash
pip install torch transformers pandas numpy scikit-learn tqdm
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.21.0` - Numerical operations
- `scikit-learn>=1.0.0` - Machine learning utilities
- `torch>=1.9.0` - PyTorch for deep learning models
- `transformers>=4.20.0` - Hugging Face transformers library
- `tqdm>=4.60.0` - Progress bars

**GPU (recommended)**: Training transformer models is much faster on GPU. The scripts automatically detect and use GPU if available.

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

- **Training set**: `data/train_split.csv` - Training data with labels
- **Development set**: `data/dev_split.csv` - Development/validation data with labels
- **Test set**: `data/test_split.csv` - Test data with gold labels (for evaluation)

## Models

This project implements four models for toxic comment classification:

### 1. Simple Baseline

**Implementation**: `code/simple_baseline.py`

A majority class baseline that predicts the most common class for each toxic category based on training data distribution. This serves as a lower bound for model performance.

**Documentation**: See `code/simple_baseline.md` for detailed documentation.

### 2. Strong Baseline

**Implementation**: `code/strong_baseline.py`

A transformer-based baseline using fine-tuned DistilBERT (a distilled version of BERT). DistilBERT is faster and smaller than BERT while maintaining most of its performance.

**Documentation**: See `code/strong_baseline.md` for detailed documentation.

### 3. DistilBERT with Focal Loss

**Implementation**: `code/distilbert.py`

DistilBERT model with focal loss to handle class imbalance. Supports automatic hyperparameter tuning and per-class threshold optimization.

**Features:**
- Focal loss with per-class alpha weighting
- Configurable gamma parameter
- Optional bias initialization
- Automatic threshold optimization

**Documentation**: See `code/distilbert.md` for detailed documentation.

### 4. RoBERTa with Focal Loss

**Implementation**: `code/roberta.py`

RoBERTa model with focal loss. RoBERTa is a robustly optimized BERT that generally achieves better performance than BERT.

**Features:**
- Focal loss with per-class alpha weighting
- Configurable gamma parameter
- Dropout rate tuning
- Optional bias initialization
- Automatic threshold optimization

**Documentation**: See `code/roberta.md` for detailed documentation.

## Quick Start

For step-by-step instructions to reproduce all results, see `code/README.md`.

**Quick example:**
```bash
# Train RoBERTa with hyperparameter tuning
python code/roberta.py \
    data/train_split.csv \
    data/test_split.csv \
    output/roberta_tuned_pred.csv \
    --dev-file data/dev_split.csv \
    --tune

# Evaluate predictions
python code/score.py output/roberta_tuned_pred.csv
```

## Evaluation

### Evaluation Metrics

The primary evaluation metric is **Macro-F1** (macro-averaged F1 score across all labels). Additional metrics include:

- **Macro-Precision**: Macro-averaged precision
- **Macro-Recall**: Macro-averaged recall
- **Mean AUC-ROC**: Threshold-independent ranking metric
- **Individual AUC-ROC scores**: Per-label scores for detailed analysis

For detailed metric definitions, see `code/score.md`.

### Running Evaluation

The evaluation script (`code/score.py`) expects prediction files that contain both predictions and gold labels:

```bash
python code/score.py <pred_file> [--thresholds THRESHOLDS]
```

**Example**:
```bash
# With default thresholds (0.5 for all labels)
python code/score.py output/roberta_tuned_pred.csv

# With custom per-class thresholds
python code/score.py output/roberta_tuned_pred.csv \
    --thresholds 0.5435,0.5204,0.5858,0.6002,0.5711,0.6004
```

**Example Output**:
```
Macro-F1: 0.675037
Macro-Precision: 0.743807
Macro-Recall: 0.625995
Mean AUC-ROC: 0.992377

Individual AUC-ROC scores:
  toxic: 0.989006
  severe_toxic: 0.991764
  obscene: 0.995291
  threat: 0.996380
  insult: 0.989678
  identity_hate: 0.992141

Metrics saved to output/roberta_tuned_pred_metrics.json
```

The script saves all metrics to a JSON file: `{pred_file}_metrics.json`

## Output Files

### Prediction Files

Each model generates CSV files in the `output/` directory with:
- `id`: Comment identifier
- `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`: Prediction probabilities (0.0 to 1.0)
- `gold_toxic`, `gold_severe_toxic`, etc.: Gold standard labels (0 or 1) for evaluation

### Metrics Files

The evaluation script generates JSON files with all metrics:
- `{pred_file}_metrics.json`: Contains `macro_f1`, `macro_precision`, `macro_recall`, `mean_auc_roc`

### Log Files

Training and tuning logs are saved to the `logs/` directory:
- `hparam_tuning_roberta.log`: RoBERTa hyperparameter tuning results
- `hparam_tuning_distilbert.log`: DistilBERT hyperparameter tuning results
- `hparam_tuning.log`: General training logs

## Documentation

### Code Documentation

- **`code/README.md`**: Step-by-step guide to reproduce all results
- **`code/simple_baseline.md`**: Simple baseline documentation
- **`code/strong_baseline.md`**: Strong baseline documentation
- **`code/distilbert.md`**: DistilBERT model documentation
- **`code/roberta.md`**: RoBERTa model documentation
- **`code/score.md`**: Evaluation metrics documentation

### Output Documentation

- **`output/README.md`**: Guide for evaluating prediction files

## Key Features

### Hyperparameter Tuning

Both DistilBERT and RoBERTa models support automatic hyperparameter tuning:
- Tests multiple combinations of `alpha_max`, `gamma`, `dropout_rate`, etc.
- Selects best hyperparameters based on dev set Macro-F1
- Saves test predictions for each combination

### Threshold Optimization

Models automatically compute optimal per-class thresholds:
- Uses precision-recall curves on training data
- Applies optimized thresholds to dev/test sets
- Improves Macro-F1 performance significantly

### Focal Loss

All transformer models use focal loss to handle class imbalance:
- Per-class alpha weighting (automatically computed)
- Configurable gamma parameter
- Focuses learning on hard-to-classify examples

## References

1. **Kaggle Competition**: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)

2. **Evaluation Metric**: 
   - Fawcett, T. (2006). "An introduction to ROC analysis". *Pattern Recognition Letters*, 27(8), 861-874.

3. **Multi-label Classification**:
   - Tsoumakas, G., & Katakis, I. (2007). "Multi-label classification: An overview". *International Journal of Data Warehousing and Mining*, 3(3), 1-13.

## Notes

- All models use random seed 42 for reproducibility
- Training times vary significantly based on hardware (GPU strongly recommended)
- Hyperparameter tuning can take 10+ hours depending on hardware
- The best hyperparameters are selected based on dev set Macro-F1 score
- Optimal thresholds are computed on training data and applied to dev/test sets
- Prediction files contain both probabilities and gold labels for self-contained evaluation

## Contact

serenagu@seas.upenn.edu
ruohanz@seas.upenn.edu
liuluyue@seas.upenn.edu
tingnan@seas.upenn.edu
