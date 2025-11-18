# Evaluation Metric: Mean Column-wise AUC-ROC

## Overview

This project uses **Mean Column-wise Area Under the Receiver Operating Characteristic Curve (AUC-ROC)** as the primary evaluation metric. This metric is the standard evaluation measure for the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) on Kaggle.

## Task Description

The task is a **multi-label classification problem** where each comment can be assigned to multiple toxic categories simultaneously. The six categories are:

1. `toxic`
2. `severe_toxic`
3. `obscene`
4. `threat`
5. `insult`
6. `identity_hate`

## Metric Definition

### AUC-ROC (Area Under the ROC Curve)

The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various classification thresholds. The AUC-ROC score represents the area under this curve and ranges from 0 to 1, where:

- **1.0** = Perfect classifier
- **0.5** = Random classifier
- **< 0.5** = Worse than random

The AUC-ROC is calculated as:

\[
\text{AUC-ROC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(x)) \, dx
\]

where:
- **TPR (True Positive Rate)** = \(\frac{\text{TP}}{\text{TP} + \text{FN}}\) (also called Sensitivity or Recall)
- **FPR (False Positive Rate)** = \(\frac{\text{FP}}{\text{FP} + \text{TN}}\)

### Mean Column-wise AUC-ROC

For multi-label classification, we compute the AUC-ROC separately for each of the six label columns, then take the mean:

\[
\text{Mean AUC-ROC} = \frac{1}{6} \sum_{i=1}^{6} \text{AUC-ROC}_i
\]

where \(\text{AUC-ROC}_i\) is the AUC-ROC score for the \(i\)-th label column.

## Why This Metric?

1. **Threshold-independent**: AUC-ROC evaluates the model's ability to rank examples correctly across all possible thresholds, making it suitable for comparing models without choosing a specific threshold.

2. **Handles class imbalance**: The dataset is highly imbalanced (most comments are non-toxic). AUC-ROC is less sensitive to class imbalance than metrics like accuracy.

3. **Multi-label appropriate**: By computing AUC-ROC per label and averaging, we get a single score that reflects performance across all toxic categories.

4. **Standard for the task**: This is the official metric used in the Kaggle competition, allowing for direct comparison with published results.

## Usage

### Command Line

```bash
python score.py <gold_file> <pred_file>
```

### Arguments

- `gold_file`: Path to the gold standard labels file (CSV format)
- `pred_file`: Path to the predicted labels file (CSV format)

### Output

The script outputs:
1. **Mean AUC-ROC**: Primary evaluation metric
2. **Macro-F1**: Macro-averaged F1 score across all labels
3. **Macro-Precision**: Macro-averaged precision across all labels
4. **Macro-Recall**: Macro-averaged recall across all labels
5. **Individual AUC-ROC scores**: Per-label AUC-ROC scores
6. **Metrics JSON file**: All metrics saved to `{pred_file}_metrics.json`

### Input Format

Both input files should be CSV files with the following columns:
- `id`: Comment identifier (optional, will be used for alignment if present)
- `toxic`: Binary label (0 or 1) or probability (0.0 to 1.0)
- `severe_toxic`: Binary label or probability
- `obscene`: Binary label or probability
- `threat`: Binary label or probability
- `insult`: Binary label or probability
- `identity_hate`: Binary label or probability

**Note**: The gold standard file should contain binary labels (0 or 1). The prediction file should contain probabilities (values between 0.0 and 1.0) for optimal AUC-ROC computation.

### Example

```bash
# Example usage
python score.py data/test_labels.csv predictions/baseline_predictions.csv
```

### Example Output

```
Mean AUC-ROC: 0.854321
Macro-F1: 0.623456
Macro-Precision: 0.712345
Macro-Recall: 0.556789

Individual AUC-ROC scores:
  toxic: 0.912345
  severe_toxic: 0.876543
  obscene: 0.890123
  threat: 0.765432
  insult: 0.901234
  identity_hate: 0.780123

Metrics saved to predictions/baseline_predictions_metrics.json
```

The metrics JSON file contains:
```json
{
  "mean_auc_roc": 0.854321,
  "macro_f1": 0.623456,
  "macro_precision": 0.712345,
  "macro_recall": 0.556789
}
```

## Implementation Details

The evaluation script:
1. Loads both gold standard and prediction CSV files
2. Aligns the dataframes by ID (if present) or by row order
3. Computes AUC-ROC for each of the six label columns using `sklearn.metrics.roc_auc_score`
4. Calculates the mean of all six AUC-ROC scores
5. Computes macro-averaged F1, precision, and recall scores:
   - Converts probability predictions to binary (threshold 0.5)
   - Computes F1, precision, and recall for each label
   - Averages across all labels (macro-averaging)
6. Outputs all metrics to stdout
7. Saves all metrics to a JSON file: `{pred_file}_metrics.json`

### Additional Metrics

In addition to Mean AUC-ROC, the script also computes:

- **Macro-F1**: Harmonic mean of precision and recall, averaged across all labels
- **Macro-Precision**: Precision averaged across all labels
- **Macro-Recall**: Recall averaged across all labels

These metrics provide complementary information to AUC-ROC:
- **AUC-ROC**: Threshold-independent ranking performance
- **Macro-F1/Precision/Recall**: Threshold-dependent classification performance (at 0.5 threshold)

## References

1. **Kaggle Competition**: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)
   - Official competition page with dataset and evaluation details

2. **AUC-ROC Metric**:
   - Fawcett, T. (2006). "An introduction to ROC analysis". *Pattern Recognition Letters*, 27(8), 861-874. [DOI: 10.1016/j.patrec.2005.10.010](https://doi.org/10.1016/j.patrec.2005.10.010)
   - Wikipedia: [Receiver Operating Characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

3. **Multi-label Classification**:
   - Tsoumakas, G., & Katakis, I. (2007). "Multi-label classification: An overview". *International Journal of Data Warehousing and Mining*, 3(3), 1-13.

4. **Jigsaw/Conversation AI**:
   - The competition was organized by Jigsaw (formerly Google's Conversation AI team) to develop better toxicity detection models. See the competition discussion forums for additional context and approaches.

## Additional Metrics (Computed but Not Primary)

While Mean AUC-ROC remains the primary evaluation metric, the script also computes:

- **Macro-F1**: Provides threshold-dependent performance assessment (computed at 0.5 threshold)
- **Macro-Precision**: Measures precision across all labels
- **Macro-Recall**: Measures recall across all labels

These metrics are useful for:
- Understanding model performance at a specific threshold
- Comparing models when threshold selection is important
- Providing complementary information to AUC-ROC

### Metrics Not Used

- **Accuracy**: Not suitable for imbalanced multi-label classification
- **Hamming Loss**: Less interpretable and not standard for this task
- **Subset Accuracy**: Too strict for multi-label problems (requires exact match on all labels)

