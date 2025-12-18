#!/usr/bin/env python3
"""
Evaluation script for Toxic Comment Classification task.
Computes Macro-F1 as the primary evaluation metric.
Also computes Macro-Precision, Macro-Recall, and Mean Column-wise AUC-ROC
as complementary metrics.

Usage:
    python score.py <pred_file> [--thresholds THRESHOLDS]

Arguments:
    pred_file: Path to predicted labels file (CSV format with columns:
        id, toxic, severe_toxic, obscene, threat, insult, identity_hate,
        gold_toxic, gold_severe_toxic, gold_obscene, gold_threat,
        gold_insult, gold_identity_hate)
    --thresholds: Optional comma-separated thresholds for each label
        (default: 0.5 for all).
        Order: toxic,severe_toxic,obscene,threat,insult,identity_hate

Output:
    Prints Macro-F1 (primary), Macro-Precision, Macro-Recall, and Mean AUC-ROC
    to stdout, along with individual AUC-ROC scores per label.
"""

import sys
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score
)
import json
import argparse


def load_labels(filepath):
    """
    Load labels from a CSV file.
    """
    df = pd.read_csv(filepath)
    return df


def extract_gold_labels(pred_df):
    """
    Extract gold labels from prediction file if they exist (gold_* columns).
    Returns gold_df with standard label column names, or None if not found.
    """
    label_columns = [
        'toxic', 'severe_toxic', 'obscene',
        'threat', 'insult', 'identity_hate'
    ]

    gold_columns = [f'gold_{col}' for col in label_columns]

    if all(col in pred_df.columns for col in gold_columns):
        gold_df = pred_df[['id'] + gold_columns].copy()
        gold_df.columns = ['id'] + label_columns
        return gold_df
    return None


def compute_mean_auc_roc(gold_df, pred_df):
    """
    Compute mean column-wise AUC-ROC score.
    """
    label_columns = [
        'toxic', 'severe_toxic', 'obscene',
        'threat', 'insult', 'identity_hate'
    ]

    if 'id' in gold_df.columns:
        gold_df = gold_df.set_index('id')
    if 'id' in pred_df.columns:
        pred_df = pred_df.set_index('id')

    gold_df, pred_df = gold_df.align(pred_df, join='inner', axis=0)

    # Compute AUC-ROC for each label
    auc_scores = []
    for label in label_columns:
        if label not in gold_df.columns or label not in pred_df.columns:
            msg = (f"Warning: Column '{label}' not found in one or both "
                   f"files. Skipping.")
            print(msg, file=sys.stderr)
            continue

        gold_labels = gold_df[label].values
        pred_probs = pred_df[label].values

        # Check if there are any positive examples for this label
        if len(np.unique(gold_labels)) < 2:
            msg = f"Warning: Label '{label}' has only one class. Skipping."
            print(msg, file=sys.stderr)
            continue

        try:
            auc = roc_auc_score(gold_labels, pred_probs)
            auc_scores.append(auc)
        except ValueError as e:
            msg = (f"Warning: Could not compute AUC for '{label}': {e}. "
                   f"Skipping.")
            print(msg, file=sys.stderr)
            continue

    if len(auc_scores) == 0:
        msg = ("Could not compute AUC for any label. "
               "Please check your input files.")
        raise ValueError(msg)

    mean_auc = np.mean(auc_scores)
    return mean_auc, auc_scores


def compute_macro_metrics(gold_df, pred_df, thresholds=None):
    """
    Compute macro-averaged F1, precision, and recall scores.

    Args:
        gold_df: DataFrame with gold standard labels
        pred_df: DataFrame with predicted probabilities
        thresholds: Array of thresholds for each label (default: 0.5 for all)
    """
    label_columns = [
        'toxic', 'severe_toxic', 'obscene',
        'threat', 'insult', 'identity_hate'
    ]

    if 'id' in gold_df.columns:
        gold_df = gold_df.set_index('id')
    if 'id' in pred_df.columns:
        pred_df = pred_df.set_index('id')

    gold_df, pred_df = gold_df.align(pred_df, join='inner', axis=0)

    gold_labels = gold_df[label_columns].values
    pred_probs = pred_df[label_columns].values

    if thresholds is None:
        thresholds = np.full(len(label_columns), 0.5)
    else:
        thresholds = np.array(thresholds)
        if len(thresholds) != len(label_columns):
            raise ValueError(
                f"Number of thresholds ({len(thresholds)}) must match "
                f"number of labels ({len(label_columns)})"
            )

    thresholds_array = np.array(thresholds).reshape(1, -1)
    binary_preds = (pred_probs >= thresholds_array).astype(int)

    macro_f1 = f1_score(
        gold_labels, binary_preds, average='macro', zero_division=0)
    macro_precision = precision_score(
        gold_labels, binary_preds, average='macro', zero_division=0)
    macro_recall = recall_score(
        gold_labels, binary_preds, average='macro', zero_division=0)

    return macro_f1, macro_precision, macro_recall


def main():
    parser = argparse.ArgumentParser(
        description='Evaluation script for Toxic Comment Classification task'
    )
    parser.add_argument(
        'pred_file',
        help='Path to predicted labels file (CSV with gold_* columns)'
    )
    parser.add_argument(
        '--thresholds',
        type=str,
        default=None,
        help=('Comma-separated thresholds for each label '
              '(default: 0.5 for all). '
              'Order: toxic,severe_toxic,obscene,threat,insult,identity_hate')
    )

    args = parser.parse_args()

    pred_file = args.pred_file

    thresholds = None
    if args.thresholds:
        try:
            thresholds = [
                float(t.strip()) for t in args.thresholds.split(',')
            ]
        except ValueError:
            msg = ("Error: Invalid thresholds format. "
                   "Use comma-separated numbers.")
            print(msg, file=sys.stderr)
            sys.exit(1)

    try:
        pred_df = load_labels(pred_file)

        # Extract gold labels from pred_file
        gold_df = extract_gold_labels(pred_df)

        if gold_df is None:
            msg = ("Error: No gold labels found in pred_file. "
                   "Expected gold_* columns (gold_toxic, gold_severe_toxic, "
                   "etc.).")
            print(msg, file=sys.stderr)
            sys.exit(1)

        macro_f1, macro_precision, macro_recall = compute_macro_metrics(
            gold_df, pred_df, thresholds=thresholds)

        mean_auc, individual_aucs = compute_mean_auc_roc(gold_df, pred_df)

        print(f"Macro-F1: {macro_f1:.6f}")
        print(f"Macro-Precision: {macro_precision:.6f}")
        print(f"Macro-Recall: {macro_recall:.6f}")
        print(f"Mean AUC-ROC: {mean_auc:.6f}")

        label_columns = [
            'toxic', 'severe_toxic', 'obscene',
            'threat', 'insult', 'identity_hate'
        ]
        print("\nIndividual AUC-ROC scores:")
        for i, label in enumerate(label_columns):
            if i < len(individual_aucs):
                print(f"  {label}: {individual_aucs[i]:.6f}")
        metrics = {
            'macro_f1': float(macro_f1),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'mean_auc_roc': float(mean_auc)
        }

        metrics_file = pred_file.replace('.csv', '_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {metrics_file}", file=sys.stderr)

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
