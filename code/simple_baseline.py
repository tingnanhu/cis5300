#!/usr/bin/env python3
"""
Simple Baseline: Majority Class Baseline

This baseline predicts the majority class (most common label) for each
toxic category based on the training data. For each label column, it
determines whether 0 (non-toxic) or 1 (toxic) is more common, and
predicts that class for all test examples.

Usage:
    python simple-baseline.py <train_file> <test_file> <output_file>
        [--dev-file PATH]

Arguments:
    train_file: Path to training data CSV file
        Must contain: id, comment_text, and label columns (toxic, severe_toxic,
        obscene, threat, insult, identity_hate)
    test_file: Path to test data CSV file
        Must contain: id column (comment_text column is optional and ignored)
    output_file: Path to output CSV file with predictions
    --dev-file PATH: (Optional) Path to development set CSV file for evaluation
        Must contain: id and label columns
"""

import sys
import pandas as pd
import numpy as np
import argparse

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def load_data(filepath):
    """Load CSV file."""
    return pd.read_csv(filepath)


def compute_majority_class(train_df, label_column):
    """
    Compute the majority class for a given label column.
    """
    if label_column not in train_df.columns:
        msg = f"Label column '{label_column}' not found in training data"
        raise ValueError(msg)

    value_counts = train_df[label_column].value_counts()

    majority_class = value_counts.idxmax()
    return majority_class


def predict_majority_class(test_df, train_df, label_columns):
    """
    Predict majority class for each label column.
    """
    if 'id' in test_df.columns:
        predictions = test_df[['id']].copy()
    else:
        predictions = pd.DataFrame()

    for label in label_columns:
        majority_class = compute_majority_class(train_df, label)
        predictions[label] = majority_class

    return predictions


def evaluate_predictions(predictions_df, gold_df, label_columns):
    """
    Evaluate predictions against gold standard labels.
    """
    if 'id' in predictions_df.columns:
        predictions_df = predictions_df.set_index('id')
    if 'id' in gold_df.columns:
        gold_df = gold_df.set_index('id')

    predictions_df, gold_df = predictions_df.align(
        gold_df, join='inner', axis=0)

    auc_scores = []
    for label in label_columns:
        if label not in gold_df.columns or label not in predictions_df.columns:
            continue

        gold_labels = gold_df[label].values
        pred_labels = predictions_df[label].values

        # Check if we have any positive examples
        if len(np.unique(gold_labels)) < 2:
            continue

        try:
            pred_probs = np.where(pred_labels == 1, 0.51, 0.49)
            auc = roc_auc_score(gold_labels, pred_probs)
            auc_scores.append(auc)
        except ValueError:
            continue

    if len(auc_scores) == 0:
        return None

    return np.mean(auc_scores)


def main():
    parser = argparse.ArgumentParser(
        description='Simple baseline: Majority class baseline')
    parser.add_argument(
        'train_file',
        help='Path to training data CSV file with labels')
    parser.add_argument('test_file', help='Path to test data CSV file')
    parser.add_argument(
        'output_file',
        help='Path to output CSV file with predictions')
    parser.add_argument(
        '--dev-file', type=str, default=None,
        help='Path to development set CSV file for evaluation')

    args = parser.parse_args()

    try:
        # Load data
        print(f"Loading training data from {args.train_file}...",
              file=sys.stderr)
        train_df = load_data(args.train_file)

        print(f"Loading test data from {args.test_file}...", file=sys.stderr)
        test_df = load_data(args.test_file)

        dev_df = None
        if args.dev_file:
            print(f"Loading development data from {args.dev_file}...",
                  file=sys.stderr)
            dev_df = load_data(args.dev_file)

        label_columns = [
            'toxic', 'severe_toxic', 'obscene',
            'threat', 'insult', 'identity_hate'
        ]

        if 'id' not in test_df.columns:
            # Create sequential IDs if not present.
            test_df['id'] = range(len(test_df))
            msg = ("Warning: 'id' column not found in test file. "
                   "Created sequential IDs.")
            print(msg, file=sys.stderr)

        print("Computing majority class predictions...", file=sys.stderr)
        predictions = predict_majority_class(test_df, train_df, label_columns)

        # Print statistics
        print("\nMajority class for each label:", file=sys.stderr)
        for label in label_columns:
            majority = compute_majority_class(train_df, label)
            count = (train_df[label] == majority).sum()
            total = len(train_df)
            percentage = (count / total) * 100
            msg = (f"  {label}: {majority} "
                   f"({count}/{total} = {percentage:.2f}%)")
            print(msg, file=sys.stderr)

        if dev_df is not None:
            if not SKLEARN_AVAILABLE:
                print("\nWarning: sklearn not available. Skipping dev set "
                      "evaluation.", file=sys.stderr)
                print("You can evaluate manually using: "
                      "python score.py cleaned/dev_split.csv "
                      "results/simple_baseline_dev_predictions.csv",
                      file=sys.stderr)
            else:
                print("\nEvaluating on development set...", file=sys.stderr)
                dev_predictions = predict_majority_class(
                    dev_df, train_df, label_columns)
                dev_score = evaluate_predictions(
                    dev_predictions, dev_df, label_columns)
                if dev_score is not None:
                    print(f"Dev set Mean AUC-ROC: {dev_score:.6f}",
                          file=sys.stderr)
                else:
                    print("Warning: Could not compute dev set score.",
                          file=sys.stderr)

        print(f"\nSaving predictions to {args.output_file}...",
              file=sys.stderr)
        predictions.to_csv(args.output_file, index=False)

        print(f"Done! Predictions saved to {args.output_file}",
              file=sys.stderr)

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
