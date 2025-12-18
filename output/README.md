# Evaluation Results

This directory contains prediction files from different models for the toxic comment classification task.

## Prediction Files

- `roberta_tuned_pred.csv`: Predictions from RoBERTa model with hyperparameter tuning
- `distilbert_tuned_pred.csv`: Predictions from DistilBERT model with hyperparameter tuning
- `strong_baseline_pred.csv`: Predictions from the strong baseline model

Each prediction file contains:
- `id`: Comment identifier
- `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`: Prediction probabilities (0.0 to 1.0)
- `gold_toxic`, `gold_severe_toxic`, `gold_obscene`, `gold_threat`, `gold_insult`, `gold_identity_hate`: Gold standard labels (0 or 1)

## Evaluating Predictions

Use the `score.py` script to evaluate predictions. The script computes Macro-F1 (primary metric), Macro-Precision, Macro-Recall, and Mean AUC-ROC.

### Basic Usage

```bash
python code/score.py output/roberta_tuned_pred.csv
```

This uses default thresholds of 0.5 for all labels.

### With Custom Thresholds

```bash
python code/score.py output/roberta_tuned_pred.csv --thresholds 0.5435,0.5204,0.5858,0.6002,0.5711,0.6004
```

The thresholds are specified in order: `toxic,severe_toxic,obscene,threat,insult,identity_hate`

### Evaluating All Models

```bash
# Evaluate RoBERTa predictions
python code/score.py output/roberta_tuned_pred.csv

# Evaluate DistilBERT predictions
python code/score.py output/distilbert_tuned_pred.csv

# Evaluate Strong Baseline predictions
python code/score.py output/strong_baseline_pred.csv
```

## Example Output

The evaluation script outputs metrics to stdout and saves them to a JSON file. Example output:

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

Metrics saved to roberta_tuned_pred_metrics.json
```

### Output Format

- **Macro-F1**: Primary evaluation metric - macro-averaged F1 score across all labels
- **Macro-Precision**: Complementary metric - macro-averaged precision across all labels
- **Macro-Recall**: Complementary metric - macro-averaged recall across all labels
- **Mean AUC-ROC**: Threshold-independent ranking metric
- **Individual AUC-ROC scores**: Per-label AUC-ROC scores for detailed analysis

### Metrics JSON File

The script also saves all metrics to a JSON file: `{pred_file}_metrics.json`

Example JSON content:
```json
{
  "macro_f1": 0.675037,
  "macro_precision": 0.743807,
  "macro_recall": 0.625995,
  "mean_auc_roc": 0.992377
}
```

## More Information

For detailed information about the evaluation metrics, see `code/score.md`.

For information about how the models were trained, see:
- `code/roberta.md` - RoBERTa model documentation
- `code/distilbert.md` - DistilBERT model documentation
- `code/strong_baseline.md` - Strong baseline documentation

