#!/usr/bin/env python3
"""
BERT Baseline: DistilBERT for Toxic Comment Classification

This baseline uses DistilBERT (a distilled version of BERT) for multi-label
toxic comment classification. DistilBERT is faster and smaller than BERT while
maintaining most of its performance.

Usage:
    python bert_baseline.py <train_file> <test_file> <output_file> [options]

Arguments:
    train_file: Path to training data CSV file
        Must contain: id, comment_text, and label columns (toxic, severe_toxic,
        obscene, threat, insult, identity_hate)
    test_file: Path to test data CSV file
        Must contain: id and comment_text columns
    output_file: Path to output CSV file with predictions

Options:
    --dev-file PATH: Path to development set CSV file for evaluation/tuning
        Must contain: id, comment_text, and label columns
    --epochs N: Number of training epochs (default: 3)
    --batch-size N: Batch size for training (default: 16)
    --learning-rate FLOAT: Learning rate (default: 2e-5)
    --max-length N: Maximum sequence length (default: 256)
    --model-name STR: HuggingFace model name (default: distilbert-base-uncased)
    --save-dir PATH: Directory to save model checkpoints (default: checkpoints)
    --save-best: Save best model based on dev set performance
    --resume PATH: Path to checkpoint directory to resume training from
    --tune: Enable hyperparameter tuning on dev set (requires --dev-file)
"""

import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    DistilBertConfig,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import roc_auc_score
import argparse
from tqdm import tqdm
import os
import csv
from sklearn.metrics import f1_score, precision_score, recall_score


class ToxicCommentDataset(Dataset):
    """Dataset class for toxic comment classification."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        """
        Initialize dataset.

        Args:
            texts: List of comment texts
            labels: Array of labels (n_samples, n_labels)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }


def preprocess_text(text):
    """Basic text preprocessing."""
    if pd.isna(text):
        return ""
    return str(text).strip()


def load_data(filepath):
    """
    Load CSV file with robust error handling.
    """
    try:
        return pd.read_csv(filepath, quoting=csv.QUOTE_MINIMAL)
    except pd.errors.ParserError:
        try:
            return pd.read_csv(
                filepath,
                quoting=csv.QUOTE_MINIMAL,
                on_bad_lines='skip',
                engine='python'
            )
        except TypeError:
            return pd.read_csv(
                filepath,
                quoting=csv.QUOTE_MINIMAL,
                error_bad_lines=False,
                warn_bad_lines=True,
                engine='python'
            )

def focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    Focal Loss wrapper for BCE-with-logits for multi-label classification.
    """
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction='none'
    )
    probs = torch.sigmoid(logits)

    pt = torch.where(targets == 1, probs, 1 - probs)
    focal_weight = alpha * (1 - pt) ** gamma

    loss = focal_weight * bce_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def train_model(
    train_df, label_columns, tokenizer, model, device,
    epochs=3, batch_size=16, learning_rate=2e-5, max_length=256,
    dev_df=None, save_dir=None, save_best=False
):
    """
    Train DistilBERT model for multi-label classification.
    """
    train_texts = train_df['comment_text'].apply(preprocess_text).tolist()
    train_labels = train_df[label_columns].values.astype(float)

    train_dataset = ToxicCommentDataset(
        train_texts, train_labels, tokenizer, max_length)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    model.train()
    best_dev_score = -1
    best_model_state = None

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}...", file=sys.stderr)
        total_loss = 0

        progress_bar = tqdm(train_loader, desc="Training", file=sys.stderr)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            loss = focal_loss_with_logits(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"  Average loss: {avg_loss:.4f}", file=sys.stderr)

        if dev_df is not None:
            dev_score = evaluate_model(
                model, dev_df, label_columns, tokenizer, device,
                batch_size=batch_size, max_length=max_length)
            print(f"  Dev AUC-ROC: {dev_score:.6f}", file=sys.stderr)
            if dev_score > best_dev_score:
                best_dev_score = dev_score
                print("  -> New best dev score!", file=sys.stderr)
                if save_best:
                    best_model_state = model.state_dict().copy()

        if save_dir:
            checkpoint_path = os.path.join(
                save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dev_score': best_dev_score,
                'avg_loss': avg_loss,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}", file=sys.stderr)

    if save_best and best_model_state is not None:
        msg = (f"\nRestoring best model "
               f"(dev AUC-ROC: {best_dev_score:.6f})...")
        print(msg, file=sys.stderr)
        model.load_state_dict(best_model_state)
        if save_dir:
            best_model_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_dev_score': best_dev_score,
            }, best_model_path)
            model_save_path = os.path.join(save_dir, 'best_model_hf')
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            print(f"Best model saved: {best_model_path}", file=sys.stderr)
            msg = f"Best model (HuggingFace format) saved: {model_save_path}"
            print(msg, file=sys.stderr)

    return model


def tune_hyperparameters(
    train_df, dev_df, label_columns, tokenizer, device,
    max_length=256, save_dir=None
):
    """
    Tune hyperparameters (Learning Rate and Dropout) on development set.
    """
    print("Tuning Learning Rate and Dropout on development set...", file=sys.stderr)

    learning_rate_options = [2e-5, 5e-5]
    dropout_rate_options = [0.1, 0.3]
    
    batch_size = 32
    epochs = 5

    best_score = -1
    best_params = {
        'learning_rate': 2e-5,
        'dropout_rate': 0.1,
    }

    total_combinations = (len(learning_rate_options) * len(dropout_rate_options))
    current = 0

    for learning_rate in learning_rate_options:
        for dropout_rate in dropout_rate_options:
            current += 1
            print(f"  [{current}/{total_combinations}] "
                  f"Testing lr={learning_rate}, dropout={dropout_rate}...",
                  file=sys.stderr)

            config = DistilBertConfig.from_pretrained(
                'distilbert-base-uncased',
                num_labels=len(label_columns),
                problem_type="multi_label_classification",
                seq_classif_dropout=dropout_rate
            )

            model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                config=config
            )
            model.to(device)

            model = train_model(
                train_df, label_columns, tokenizer, model, device,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_length=max_length,
                dev_df=None,
                save_dir=None,
                save_best=False
            )

            score = evaluate_model(
                model, dev_df, label_columns, tokenizer, device,
                batch_size=batch_size,
                max_length=max_length)

            print(f"    Dev AUC-ROC: {score:.6f}", file=sys.stderr)

            if score > best_score:
                best_score = score
                best_params = {
                    'learning_rate': learning_rate,
                    'dropout_rate': dropout_rate
                }
                print(f"    -> New best! (AUC-ROC: {score:.6f})",
                      file=sys.stderr)

            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\nBest hyperparameters: {best_params} "
          f"(Dev AUC-ROC: {best_score:.6f})", file=sys.stderr)

    return (best_params['learning_rate'], best_params['dropout_rate'])


def evaluate_model(
    model, df, label_columns, tokenizer, device,
    batch_size=16, max_length=256
):
    """
    Evaluate model and return mean AUC-ROC, plus Macro-F1, Precision, Recall.
    """
    model.eval()

    texts = df['comment_text'].apply(preprocess_text).tolist()
    labels = df[label_columns].values.astype(float)

    dataset = ToxicCommentDataset(texts, labels, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", file=sys.stderr):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].numpy()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()

            predictions.append(probs)
            true_labels.append(batch_labels)


    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    auc_scores = []
    for i, label in enumerate(label_columns):
        try:
            auc = roc_auc_score(true_labels[:, i], predictions[:, i])
            auc_scores.append(auc)
        except ValueError:
            continue

    mean_auc = np.mean(auc_scores) if auc_scores else 0.0

    binary_preds = (predictions >= 0.5).astype(int)

    macro_f1 = f1_score(true_labels, binary_preds, average='macro', zero_division=0)
    macro_precision = precision_score(true_labels, binary_preds, average='macro', zero_division=0)
    macro_recall = recall_score(true_labels, binary_preds, average='macro', zero_division=0)

    print(f"  Macro-F1:      {macro_f1:.6f}", file=sys.stderr)
    print(f"  Macro-Precision: {macro_precision:.6f}", file=sys.stderr)
    print(f"  Macro-Recall:  {macro_recall:.6f}", file=sys.stderr)

    return mean_auc


def predict(
    model, test_df, label_columns, tokenizer, device,
    batch_size=16, max_length=256
):
    """
    Make predictions on test data.
    """
    model.eval()

    texts = test_df['comment_text'].apply(preprocess_text).tolist()
    dummy_labels = np.zeros((len(texts), len(label_columns)))

    dataset = ToxicCommentDataset(texts, dummy_labels, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", file=sys.stderr):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()

            predictions.append(probs)

    predictions = np.concatenate(predictions, axis=0)

    output_df = pd.DataFrame()
    if 'id' in test_df.columns:
        output_df['id'] = test_df['id'].values
    else:
        output_df['id'] = range(len(test_df))

    for i, label in enumerate(label_columns):
        output_df[label] = predictions[:, i]

    return output_df


def main():
    parser = argparse.ArgumentParser(
        description=('BERT baseline: DistilBERT for '
                     'toxic comment classification'))
    parser.add_argument(
        'train_file',
        help='Path to training data CSV file')
    parser.add_argument('test_file', help='Path to test data CSV file')
    parser.add_argument('output_file', help='Path to output CSV file')
    parser.add_argument(
        '--dev-file', type=str, default=None,
        help='Path to development set CSV file')
    parser.add_argument(
        '--epochs', type=int, default=3,
        help='Number of training epochs (default: 3)')
    parser.add_argument(
        '--batch-size', type=int, default=16,
        help='Batch size for training (default: 16)')
    parser.add_argument(
        '--learning-rate', type=float, default=2e-5,
        help='Learning rate (default: 2e-5)')
    parser.add_argument(
        '--max-length', type=int, default=256,
        help='Maximum sequence length (default: 256)')
    parser.add_argument(
        '--model-name', type=str, default='distilbert-base-uncased',
        help='HuggingFace model name (default: distilbert-base-uncased)')
    parser.add_argument(
        '--save-dir', type=str, default='checkpoints',
        help='Directory to save model checkpoints (default: checkpoints)')
    parser.add_argument(
        '--save-best', action='store_true',
        help='Save best model based on dev set performance')
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint directory to resume training from')
    parser.add_argument(
        '--tune', action='store_true',
        help='Enable hyperparameter tuning on dev set (requires --dev-file)')

    args = parser.parse_args()

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}", file=sys.stderr)

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

        if 'comment_text' not in train_df.columns:
            msg = "Training file must contain 'comment_text' column"
            raise ValueError(msg)
        if 'comment_text' not in test_df.columns:
            msg = "Test file must contain 'comment_text' column"
            raise ValueError(msg)

        label_columns = [
            'toxic', 'severe_toxic', 'obscene',
            'threat', 'insult', 'identity_hate'
        ]

        missing_labels = [
            label for label in label_columns
            if label not in train_df.columns]
        if missing_labels:
            msg = f"Training file missing label columns: {missing_labels}"
            raise ValueError(msg)

        learning_rate = args.learning_rate
        batch_size = args.batch_size
        epochs = args.epochs
        dropout_rate = 0.2

        tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)

        if args.tune:
            if dev_df is None:
                raise ValueError("--tune requires --dev-file to be specified")
            if 'comment_text' not in dev_df.columns:
                msg = "Dev file must contain 'comment_text' column"
                raise ValueError(msg)
            
            learning_rate, dropout_rate = tune_hyperparameters(
                train_df, dev_df, label_columns, tokenizer, device,
                max_length=args.max_length,
                save_dir=None 
            )

        start_epoch = 0
        
        if args.resume:
            print(f"Resuming from checkpoint: {args.resume}...", file=sys.stderr)
            checkpoint_pattern = os.path.join(args.resume, 'checkpoint_epoch_*.pt')
            import glob
            checkpoints = glob.glob(checkpoint_pattern)
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=os.path.getctime)
                print(f"Loading checkpoint: {latest_checkpoint}...", file=sys.stderr)
                checkpoint = torch.load(latest_checkpoint, map_location=device)

                config = DistilBertConfig.from_pretrained(
                    args.model_name,
                    num_labels=len(label_columns),
                    problem_type="multi_label_classification",
                    seq_classif_dropout=dropout_rate 
                )
                model = DistilBertForSequenceClassification.from_pretrained(
                    args.model_name,
                    config=config
                )
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                start_epoch = checkpoint['epoch']
                print(f"Resumed from epoch {start_epoch}", file=sys.stderr)
            else:
                print("Warning: No checkpoint found, starting from scratch", file=sys.stderr)
                config = DistilBertConfig.from_pretrained(
                    args.model_name,
                    num_labels=len(label_columns),
                    problem_type="multi_label_classification",
                    seq_classif_dropout=dropout_rate
                )
                model = DistilBertForSequenceClassification.from_pretrained(
                    args.model_name,
                    config=config
                )
                model.to(device)
        else:
            print(f"Loading model: {args.model_name} with dropout {dropout_rate}...",
                  file=sys.stderr)
            config = DistilBertConfig.from_pretrained(
                args.model_name,
                num_labels=len(label_columns),
                problem_type="multi_label_classification",
                seq_classif_dropout=dropout_rate
            )
            model = DistilBertForSequenceClassification.from_pretrained(
                args.model_name,
                config=config
            )
            model.to(device)

        print(f"\nTraining final model with {epochs} epochs, "
              f"batch_size={batch_size}, "
              f"learning_rate={learning_rate}, "
              f"dropout_rate={dropout_rate}...",
              file=sys.stderr)
              
        model = train_model(
            train_df, label_columns, tokenizer, model, device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=args.max_length,
            dev_df=dev_df,
            save_dir=args.save_dir,
            save_best=args.save_best
        )

        print("\nMaking predictions on test set...", file=sys.stderr)
        predictions = predict(
            model, test_df, label_columns, tokenizer, device,
            batch_size=args.batch_size,
            max_length=args.max_length
        )

        print(f"Saving predictions to {args.output_file}...", file=sys.stderr)
        predictions.to_csv(args.output_file, index=False)

        msg = f"Done! Predictions saved to {args.output_file}"
        print(msg, file=sys.stderr)
        print("Predictions are probabilities (0.0 to 1.0) for each "
              "toxic category.", file=sys.stderr)

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
