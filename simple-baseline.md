# Simple Baseline: Majority Class Baseline

## Overview

The simple baseline is a **majority class baseline** that predicts the most common class for each toxic category based on the training data. For each of the six label columns (toxic, severe_toxic, obscene, threat, insult, identity_hate), the baseline determines whether class 0 (non-toxic) or class 1 (toxic) is more frequent in the training set, and predicts that class for all test examples.

## How It Works

1. **Training Phase**: For each label column, count the occurrences of class 0 and class 1 in the training data.
2. **Prediction Phase**: For each label column, predict the majority class (the one with higher count) for all test examples.

### Example

If in the training data:
- `toxic`: 90% are class 0 (non-toxic), 10% are class 1 (toxic)
- `severe_toxic`: 99% are class 0, 1% are class 1

Then the baseline will predict:
- `toxic = 0` for all test examples
- `severe_toxic = 0` for all test examples

## Usage

### Basic Usage

```bash
python simple-baseline.py <train_file> <test_file> <output_file>
```

### With Development Set Evaluation

```bash
python simple-baseline.py <train_file> <test_file> <output_file> --dev-file <dev_file>
```

This will also evaluate the baseline on the development set and report the Mean AUC-ROC score.

### Arguments

- `train_file`: Path to training data CSV file (required)
  - Must contain: `id`, `comment_text`, and label columns
  - Label columns: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- `test_file`: Path to test data CSV file (required)
  - Must contain: `id` column
  - `comment_text` column is optional (ignored by simple baseline)
- `output_file`: Path where predictions will be saved (required)
- `--dev-file PATH`: Path to development set CSV file (optional)
  - Must contain: `id`, `comment_text`, and label columns
  - Used for evaluation only (no hyperparameter tuning needed for this baseline)

### Example

```bash
# Basic usage
python simple-baseline.py data/train.csv data/test.csv predictions/simple_baseline.csv

# With dev set evaluation
python simple-baseline.py data/train.csv data/test.csv predictions/simple_baseline.csv --dev-file data/dev.csv
```

## Input Format

### Training File (`train.csv`)
- Must contain: `id`, `comment_text`, and label columns
- Label columns: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- Each label column should contain binary values (0 or 1)
- Example format:
  ```
  id,comment_text,toxic,severe_toxic,obscene,threat,insult,identity_hate
  ed56f082116dcbd0,"Grandma Terri Should Burn in Trash...",1,0,0,0,0,0
  abc123,"This is a normal comment",0,0,0,0,0,0
  ```

### Development File (`dev.csv`) - Optional
- Must contain: `id`, `comment_text`, and label columns
- Label columns: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- Used for evaluation to report performance on dev set

### Test File (`test.csv`)
- Must contain: `id` column
- May contain `comment_text` column (will be ignored by simple baseline)
- Other columns are ignored

### Output File (`simple_baseline.csv`)
- Contains columns: `id`, `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- All predictions are binary (0 or 1)

## Sample Output

When running the script, you'll see output like:

```
Loading training data from data/train.csv...
Loading test data from data/test.csv...
Loading development data from data/dev.csv...
Computing majority class predictions...

Majority class for each label:
  toxic: 0 (143346/159571 = 89.83%)
  severe_toxic: 0 (159254/159571 = 99.80%)
  obscene: 0 (152941/159571 = 95.84%)
  threat: 0 (159252/159571 = 99.80%)
  insult: 0 (151700/159571 = 95.07%)
  identity_hate: 0 (156810/159571 = 98.27%)

Evaluating on development set...
Dev set Mean AUC-ROC: 0.500000

Saving predictions to predictions/simple_baseline.csv...
Done! Predictions saved to predictions/simple_baseline.csv
```

## Evaluation

To evaluate the simple baseline on test set:

```bash
python score.py data/test_labels.csv predictions/simple_baseline.csv
```

The evaluation script will output:
- **Mean AUC-ROC**: Primary metric (typically ~0.5 for majority class baseline)
- **Macro-F1**: Macro-averaged F1 score
- **Macro-Precision**: Macro-averaged precision
- **Macro-Recall**: Macro-averaged recall
- **Individual AUC-ROC scores**: Per-label scores
- **Metrics JSON file**: All metrics saved to `predictions/simple_baseline_metrics.json`

**Note**: The evaluation script expects probabilities (0.0 to 1.0), but the simple baseline outputs binary predictions (0 or 1). The evaluation script will still work, but the AUC-ROC will be approximately 0.5 since all examples get the same prediction.

### Example Evaluation Output

When evaluating with `score.py`, you'll see output like:

```
Mean AUC-ROC: 0.500000
Macro-F1: 0.123456
Macro-Precision: 0.234567
Macro-Recall: 0.089012

Individual AUC-ROC scores:
  toxic: 0.500000
  severe_toxic: 0.500000
  obscene: 0.500000
  threat: 0.500000
  insult: 0.500000
  identity_hate: 0.500000

Metrics saved to predictions/simple_baseline_metrics.json
```

## Expected Performance

The majority class baseline typically achieves a **Mean AUC-ROC score around 0.50**, which is equivalent to random guessing. This is because:

1. **Class imbalance**: Most labels have a strong majority class (often >95% are class 0)
2. **AUC-ROC interpretation**: When predicting the same class for all examples, the ROC curve becomes a diagonal line, resulting in AUC = 0.5
3. **No learning**: The baseline doesn't use any features from the comments, only the label distribution

### Why AUC-ROC ≈ 0.5?

When a classifier always predicts the same class (e.g., always predicts 0), the ROC curve becomes a straight diagonal line from (0,0) to (1,1), which has an area of 0.5. This represents a classifier that performs no better than random guessing.

## Limitations

1. **No feature usage**: The baseline completely ignores the comment text and other features
2. **Poor performance**: Achieves essentially random performance (AUC-ROC ≈ 0.5)
3. **Not useful for deployment**: Would not be effective in a real-world application

## Purpose

Despite its poor performance, the majority class baseline serves important purposes:

1. **Lower bound**: Establishes a minimum performance threshold that any real model should exceed
2. **Sanity check**: Helps verify that the evaluation metric and data pipeline are working correctly
3. **Baseline comparison**: Provides a reference point to measure improvement from more sophisticated models

## Development Set Usage

The development set is used for **evaluation only** (not hyperparameter tuning, since this baseline has no hyperparameters). This allows you to:

1. **Monitor performance**: See how the baseline performs on held-out data
2. **Compare with other baselines**: Use the same dev set to compare simple vs. strong baselines
3. **Verify evaluation pipeline**: Ensure your evaluation code works correctly before running on test set

