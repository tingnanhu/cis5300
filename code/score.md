# Evaluation Metric: Macro-F1

## Overview

This project uses **Macro-F1** as the primary evaluation metric. This metric provides threshold-dependent performance assessment across all toxic categories. The script also computes Macro-Precision, Macro-Recall, and Mean Column-wise AUC-ROC as complementary metrics.

## Task Description

The task is a **multi-label classification problem** where each comment can be assigned to multiple toxic categories simultaneously. The six categories are:

1. `toxic`
2. `severe_toxic`
3. `obscene`
4. `threat`
5. `insult`
6. `identity_hate`

## Metric Definition

### Macro-F1

For multi-label classification, we compute F1 for each of the six label columns, then take the macro average (unweighted mean) across all labels:

\[
\text{Macro-F1} = \frac{1}{6} \sum_{i=1}^{6} \text{F1}_i
\]

where:
- **F1** = \(2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}\) (harmonic mean of precision and recall)
- **Precision** = \(\frac{\text{TP}}{\text{TP} + \text{FP}}\) (positive predictive value)
- **Recall** = \(\frac{\text{TP}}{\text{TP} + \text{FN}}\) (sensitivity, true positive rate)

Macro-F1 is computed by:
1. Converting probability predictions to binary predictions using per-class thresholds (default: 0.5 for all labels, or custom thresholds via `--thresholds`)
2. Computing F1 for each label
3. Averaging across all six labels (macro-averaging)

## Why This Metric?

1. **Threshold-dependent evaluation**: Macro-F1 evaluates model performance at specific classification thresholds, which is important for practical deployment where binary decisions are needed.

2. **Per-class thresholds**: The `--thresholds` argument allows optimization of thresholds per toxic category, which is crucial for handling class imbalance and different operational requirements.

3. **Balanced performance measure**: F1 provides a balanced view that considers both precision and recall, making it suitable for imbalanced multi-label classification tasks.

4. **Multi-label appropriate**: Macro-averaging treats all labels equally, providing a balanced view of performance across all toxic categories regardless of their frequency.

5. **Interpretable and actionable**: F1 provides a clear, interpretable measure of classification performance that directly relates to real-world application needs.

## Usage

### Command Line

```bash
python score.py <pred_file> [--thresholds THRESHOLDS]
```

### Arguments

- `pred_file`: Path to the predicted labels file (CSV format). Must include both prediction probabilities and gold standard labels (see Input Format below).
- `--thresholds`: Optional comma-separated thresholds for each label (default: 0.5 for all). Order: `toxic,severe_toxic,obscene,threat,insult,identity_hate`

### Output

The script outputs:
1. **Macro-F1**: Primary evaluation metric - macro-averaged F1 score across all labels
2. **Macro-Precision**: Complementary metric - macro-averaged precision across all labels
3. **Macro-Recall**: Complementary metric - macro-averaged recall across all labels
4. **Mean AUC-ROC**: Complementary threshold-independent metric
5. **Individual AUC-ROC scores**: Per-label AUC-ROC scores
6. **Metrics JSON file**: All metrics saved to `{pred_file}_metrics.json`

### Input Format

The prediction file should be a CSV file with the following columns:
- `id`: Comment identifier (used for alignment)
- `toxic`: Prediction probability (0.0 to 1.0)
- `severe_toxic`: Prediction probability (0.0 to 1.0)
- `obscene`: Prediction probability (0.0 to 1.0)
- `threat`: Prediction probability (0.0 to 1.0)
- `insult`: Prediction probability (0.0 to 1.0)
- `identity_hate`: Prediction probability (0.0 to 1.0)
- `gold_toxic`: Gold standard binary label (0 or 1)
- `gold_severe_toxic`: Gold standard binary label (0 or 1)
- `gold_obscene`: Gold standard binary label (0 or 1)
- `gold_threat`: Gold standard binary label (0 or 1)
- `gold_insult`: Gold standard binary label (0 or 1)
- `gold_identity_hate`: Gold standard binary label (0 or 1)

**Note**: The prediction file must contain both prediction probabilities and gold standard labels. Gold labels are identified by the `gold_*` prefix. This allows for self-contained evaluation files.

### Example

```bash
# Basic usage (default thresholds: 0.5 for all labels)
python score.py output/roberta_tuned_pred.csv

# With custom per-class thresholds
python score.py output/roberta_tuned_pred.csv --thresholds 0.5435,0.5204,0.5858,0.6002,0.5711,0.6004
```

### Example Output

```
Macro-F1: 0.623456
Macro-Precision: 0.712345
Macro-Recall: 0.556789
Mean AUC-ROC: 0.854321

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
  "macro_f1": 0.623456,
  "macro_precision": 0.712345,
  "macro_recall": 0.556789,
  "mean_auc_roc": 0.854321
}
```

## Implementation Details

The evaluation script:
1. Loads the prediction CSV file containing both predictions and gold labels
2. Extracts gold labels from `gold_*` columns in the prediction file
3. Aligns predictions and gold labels by ID
4. Computes primary metric (Macro-F1):
   - Converts probability predictions to binary using per-class thresholds
   - Default threshold is 0.5 for all labels if `--thresholds` is not provided
   - Computes F1 for each label
   - Averages across all labels (macro-averaging)
5. Computes complementary metrics (Macro-Precision, Macro-Recall, Mean AUC-ROC):
   - Macro-Precision and Macro-Recall: Computed similarly to Macro-F1 but using precision and recall respectively
   - Mean AUC-ROC: Computes AUC-ROC for each of the six label columns using `sklearn.metrics.roc_auc_score`, then calculates the mean
6. Outputs all metrics to stdout
7. Saves all metrics to a JSON file: `{pred_file}_metrics.json`

### Additional Metrics

In addition to the primary metric, the script also computes:

- **Macro-Precision**: Macro-averaged precision across all labels
  - Provides information about false positive control
  - Useful for understanding precision at the selected thresholds

- **Macro-Recall**: Macro-averaged recall across all labels
  - Provides information about false negative control
  - Useful for understanding recall at the selected thresholds

- **Mean AUC-ROC**: Threshold-independent ranking performance metric
  - Computes AUC-ROC separately for each label column
  - Takes the mean across all six labels
  - Provides threshold-independent performance assessment

These metrics provide complementary information:
- **Macro-F1**: Primary threshold-dependent classification performance (uses custom thresholds if provided via `--thresholds`, otherwise 0.5 for all labels)
- **Macro-Precision/Recall**: Additional threshold-dependent metrics for detailed performance analysis
- **Mean AUC-ROC**: Threshold-independent ranking performance across all possible thresholds

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

## Threshold Selection

The `--thresholds` argument allows you to specify optimal per-class thresholds for binary classification. This is important because:

- **Class imbalance**: Different toxic categories have different base rates, so optimal thresholds may vary
- **Operational requirements**: Different applications may prioritize precision vs. recall differently
- **Model calibration**: Well-calibrated models may benefit from threshold optimization

If `--thresholds` is not provided, the default threshold of 0.5 is used for all labels. Optimal thresholds can be determined using precision-recall curves on training or validation data.

