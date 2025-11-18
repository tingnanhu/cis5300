#!/usr/bin/env python3
"""
BERT Baseline: BERT for Toxic Comment Classification

This baseline uses BERT (not DistilBERT) for multi-label
toxic comment classification.

Usage:
    python bert_baseline.py <train_file> <test_file> <output_file> [options]
"""

import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import roc_auc_score
import argparse
import csv
from tqdm import tqdm
import os
from sklearn.metrics import f1_score, precision_score, recall_score


class ToxicCommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
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
    if pd.isna(text):
        return ""
    return str(text).strip()


def load_data(filepath):
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

            if save_best and dev_score > best_dev_score:
                best_dev_score = dev_score
                best_model_state = model.state_dict().copy()
                print("  -> New best model!", file=sys.stderr)

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
            print(f"  Saved checkpoint: {checkpoint_path}", file=sys.stderr)

    if save_best and best_model_state is not None:
        print(f"\nRestoring best model (AUC-ROC={best_dev_score:.6f})",
              file=sys.stderr)
        model.load_state_dict(best_model_state)

        if save_dir:
            best_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({'model_state_dict': model.state_dict()}, best_path)

            hf_path = os.path.join(save_dir, 'best_model_hf')
            model.save_pretrained(hf_path)
            tokenizer.save_pretrained(hf_path)

            print(f"Saved best model to {best_path}", file=sys.stderr)

    return model


def tune_hyperparameters(
    train_df, dev_df, label_columns, tokenizer, device,
    max_length=256, save_dir=None
):
    print("Tuning hyperparameters...", file=sys.stderr)

    lr_options = [1e-5, 2e-5, 3e-5]
    batch_options = [8, 16, 32]
    epoch_options = [2, 3]

    best_score = -1
    best_params = None

    total = len(lr_options) * len(batch_options) * len(epoch_options)
    counter = 0

    for lr in lr_options:
        for bs in batch_options:
            for ep in epoch_options:
                counter += 1
                print(f"[{counter}/{total}] lr={lr}, batch={bs}, ep={ep}",
                      file=sys.stderr)

                model = BertForSequenceClassification.from_pretrained(
                    tokenizer.name_or_path,
                    num_labels=len(label_columns),
                    problem_type="multi_label_classification"
                ).to(device)

                model = train_model(
                    train_df, label_columns, tokenizer, model, device,
                    epochs=ep,
                    batch_size=bs,
                    learning_rate=lr,
                    max_length=max_length,
                    dev_df=None,
                    save_dir=None,
                    save_best=False
                )

                score = evaluate_model(
                    model, dev_df, label_columns, tokenizer, device,
                    batch_size=bs, max_length=max_length)

                print(f"Dev AUC-ROC: {score:.6f}", file=sys.stderr)

                if score > best_score:
                    best_score = score
                    best_params = (lr, bs, ep)

                del model
                torch.cuda.empty_cache()

    print(f"Best params: {best_params} AUC={best_score:.6f}", file=sys.stderr)
    return best_params


def evaluate_model(
    model, df, label_columns, tokenizer, device,
    batch_size=16, max_length=256
):
    model.eval()

    texts = df['comment_text'].apply(preprocess_text).tolist()
    labels = df[label_columns].values.astype(float)

    dataset = ToxicCommentDataset(texts, labels, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds, trues = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", file=sys.stderr):
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].numpy()

            logits = model(input_ids, attention_mask=attn).logits
            probs = torch.sigmoid(logits).cpu().numpy()

            preds.append(probs)
            trues.append(batch_labels)

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    aucs = []
    for i in range(len(label_columns)):
        try:
            aucs.append(roc_auc_score(trues[:, i], preds[:, i]))
        except ValueError:
            pass
    mean_auc = np.mean(aucs) if aucs else 0.0

    hard = (preds >= 0.5).astype(int)
    macro_f1 = f1_score(trues, hard, average='macro', zero_division=0)
    macro_prec = precision_score(trues, hard, average='macro', zero_division=0)
    macro_rec = recall_score(trues, hard, average='macro', zero_division=0)

    print(f"Macro-F1: {macro_f1:.6f}", file=sys.stderr)
    print(f"Macro-Precision: {macro_prec:.6f}", file=sys.stderr)
    print(f"Macro-Recall: {macro_rec:.6f}", file=sys.stderr)

    return mean_auc


def predict(
    model, test_df, label_columns, tokenizer, device,
    batch_size=16, max_length=256
):
    model.eval()

    texts = test_df['comment_text'].apply(preprocess_text).tolist()
    dummy = np.zeros((len(texts), len(label_columns)))

    dataset = ToxicCommentDataset(texts, dummy, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", file=sys.stderr):
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)

            logits = model(input_ids, attention_mask=attn).logits
            probs = torch.sigmoid(logits).cpu().numpy()

            preds.append(probs)

    preds = np.concatenate(preds)

    out = pd.DataFrame()
    out['id'] = test_df['id'] if 'id' in test_df else range(len(test_df))

    for i, label in enumerate(label_columns):
        out[label] = preds[:, i]

    return out


def main():
    parser = argparse.ArgumentParser(
        description='BERT baseline for toxic comment classification')
    parser.add_argument('train_file')
    parser.add_argument('test_file')
    parser.add_argument('output_file')

    parser.add_argument('--dev-file', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--model-name', type=str,
                        default='bert-base-uncased')

    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--save-best', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--tune', action='store_true')

    args = parser.parse_args()

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}", file=sys.stderr)

        train_df = load_data(args.train_file)
        test_df = load_data(args.test_file)

        dev_df = load_data(args.dev_file) if args.dev_file else None

        label_columns = [
            'toxic', 'severe_toxic', 'obscene',
            'threat', 'insult', 'identity_hate'
        ]

        tokenizer = BertTokenizer.from_pretrained(args.model_name)

        start_epoch = 0

        if args.resume:
            import glob
            pat = os.path.join(args.resume, 'checkpoint_epoch_*.pt')
            ckpts = glob.glob(pat)

            if ckpts:
                latest = max(ckpts, key=os.path.getctime)
                print(f"Resuming from {latest}", file=sys.stderr)

                checkpoint = torch.load(latest, map_location=device)

                model = BertForSequenceClassification.from_pretrained(
                    args.model_name,
                    num_labels=len(label_columns),
                    problem_type="multi_label_classification"
                )
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                start_epoch = checkpoint['epoch']

            else:
                print("No checkpoint found. Starting new model.", file=sys.stderr)
                model = BertForSequenceClassification.from_pretrained(
                    args.model_name,
                    num_labels=len(label_columns),
                    problem_type="multi_label_classification"
                ).to(device)

        else:
            print(f"Loading model: {args.model_name}...", file=sys.stderr)
            model = BertForSequenceClassification.from_pretrained(
                args.model_name,
                num_labels=len(label_columns),
                problem_type="multi_label_classification"
            ).to(device)

        learning_rate = args.learning_rate
        batch_size = args.batch_size
        epochs = args.epochs

        if args.tune:
            if dev_df is None:
                raise ValueError("--tune requires --dev-file")
            learning_rate, batch_size, epochs = tune_hyperparameters(
                train_df, dev_df, label_columns, tokenizer, device,
                max_length=args.max_length
            )
        elif args.save_best and dev_df is None:
            print("Warning: --save-best requires --dev-file. Ignoring.",
                  file=sys.stderr)
            args.save_best = False

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

        preds = predict(
            model, test_df, label_columns, tokenizer, device,
            batch_size=args.batch_size,
            max_length=args.max_length
        )

        preds.to_csv(args.output_file, index=False)

        print(f"Done! Saved to {args.output_file}", file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
