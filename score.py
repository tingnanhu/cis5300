#!/usr/bin/env python3
"""
Evaluation script for Toxic Comment Classification task.
Computes Mean Column-wise AUC-ROC score.

Usage:
    python score.py <gold_file> <pred_file>

Arguments:
    gold_file: Path to gold standard labels file (CSV format with columns:
        id, toxic, severe_toxic, obscene, threat, insult, identity_hate)
    pred_file: Path to predicted labels file (CSV format with same columns)

Output:
    Prints the mean AUC-ROC score to stdout.
"""

import sys
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score
)
import json


def load_labels(filepath):
    """
    Load labels from a CSV file.
    """
    df = pd.read_csv(filepath)
    return df


def compute_mean_auc_roc(gold_df, pred_df):
    """
    Compute mean column-wise AUC-ROC score.
    """
    # Define the label columns (excluding 'id' if present)
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


def compute_macro_metrics(gold_df, pred_df):
    """
    Compute macro-averaged F1, precision, and recall scores.
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

    # Convert predictions to binary (threshold 0.5)
    gold_labels = gold_df[label_columns].values
    pred_probs = pred_df[label_columns].values
    pred_binary = (pred_probs >= 0.5).astype(int)

    macro_f1 = f1_score(
        gold_labels, pred_binary, average='macro', zero_division=0)
    macro_precision = precision_score(
        gold_labels, pred_binary, average='macro', zero_division=0)
    macro_recall = recall_score(
        gold_labels, pred_binary, average='macro', zero_division=0)

    return macro_f1, macro_precision, macro_recall


def main():
    if len(sys.argv) != 3:
        msg = "Usage: python score.py <gold_file> <pred_file>"
        print(msg, file=sys.stderr)
        sys.exit(1)

    gold_file = sys.argv[1]
    pred_file = sys.argv[2]

    try:
        gold_df = load_labels(gold_file)
        pred_df = load_labels(pred_file)

        mean_auc, individual_aucs = compute_mean_auc_roc(gold_df, pred_df)

        macro_f1, macro_precision, macro_recall = compute_macro_metrics(
            gold_df, pred_df)

        print(f"Mean AUC-ROC: {mean_auc:.6f}")
        print(f"Macro-F1: {macro_f1:.6f}")
        print(f"Macro-Precision: {macro_precision:.6f}")
        print(f"Macro-Recall: {macro_recall:.6f}")

        label_columns = [
            'toxic', 'severe_toxic', 'obscene',
            'threat', 'insult', 'identity_hate'
        ]
        print("\nIndividual AUC-ROC scores:")
        for i, label in enumerate(label_columns):
            if i < len(individual_aucs):
                print(f"  {label}: {individual_aucs[i]:.6f}")
        metrics = {
            'mean_auc_roc': float(mean_auc),
            'macro_f1': float(macro_f1),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall)
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
