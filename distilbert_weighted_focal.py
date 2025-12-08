# !/usr/bin/env python3
"""
DistilBERT for Toxic Comment Classification
...
"""

import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_curve
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
import random
from sklearn.metrics import f1_score, precision_score, recall_score
import logging


# Dataset
class ToxicCommentDataset(Dataset):
    """Dataset class for toxic comment classification."""

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


# Utilities
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


# Focal loss with per-class alpha and gamma
def focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    logits: (B, C)
    targets: (B, C)
    alpha: float or Tensor(C,)
    gamma: focal loss gamma
    reduction: 'mean'|'sum'|'none'
    """
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )

    probs = torch.sigmoid(logits).clamp(min=1e-6, max=1-1e-6)
    pt = torch.where(targets == 1, probs, 1 - probs)

    # Alpha for positives, 1-alpha for negatives
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.to(logits.device)
        alpha = alpha.view(1, -1)  # (1, C)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    else:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    focal_weight = alpha_t * (1 - pt) ** gamma
    loss = focal_weight * bce_loss

    if reduction == 'mean':
        # Normalize by number of positives to keep the loss scale consistent
        num_positives = targets.sum()
        return loss.sum() / (num_positives + 1e-7)
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


# Initialize bias with pi clipping
def initialize_bias(model, label_counts, num_samples, pi_min=1e-4, pi_max=0.9):
    """
    Initialize the final layer bias to reflect the data imbalance.
    b = -log((1 - pi) / pi)
    pi_min, pi_max: clipping bounds for empirical positive rates
    """

    # Compute positive rate for each class and clip to avoid extreme values
    pi = torch.tensor(label_counts / num_samples, dtype=torch.float32)
    print(pi)
    pi = pi.clamp(min=pi_min, max=pi_max)
    # Set bias so that sigmoid(bias) ≈ pi (the positive rate)
    bias_values = -torch.log((1 - pi) / pi)
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'bias'):
        with torch.no_grad():
            device = model.classifier.bias.device
            model.classifier.bias.copy_(bias_values.to(device))
            print(f"Initialized output bias values: {bias_values.numpy()}", file=sys.stderr)


def compute_training_thresholds(
    model, train_df, label_columns, tokenizer, device,
    batch_size=32, max_length=256, log=None
):
    """
    Compute optimal per-class thresholds on TRAINING data using PR curves.
    The returned thresholds are applied later on DEV/TEST for binary prediction.
    """
    model.eval()
    texts = train_df['comment_text'].apply(preprocess_text).tolist()
    labels = train_df[label_columns].values.astype(float)

    dataset = ToxicCommentDataset(texts, labels, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Thresholds", file=sys.stderr):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()

            preds.append(probs)
            trues.append(batch['labels'].cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    thresholds = []
    for i in range(len(label_columns)):
        try:
            p, r, t = precision_recall_curve(trues[:, i], preds[:, i])
            # Find threshold that maximizes F1 score
            f1 = (2 * p * r) / (p + r + 1e-9)
            best = t[np.argmax(f1[:-1])]
        except:
            best = 0.5  # fallback
        thresholds.append(best)

    thresholds = np.array(thresholds)

    if log:
        log("\n--- Optimal Training Thresholds ---")
        for label, t in zip(label_columns, thresholds):
            log(f"  {label}: threshold={t:.4f}")
        log("---------------------------------")
    
    return thresholds


# Training
def train_model(
    train_df, label_columns, tokenizer, model, device,
    epochs=3, batch_size=32, learning_rate=2e-5, max_length=256,
    dev_df=None, save_dir=None, save_best=False,
    alpha_per_class=None, gamma=2.0, max_grad_norm=1.0
):
    """
    Train DistilBERT model for multi-label classification.

    Args:
        gamma: focal loss gamma
        alpha_per_class: numpy array or list with length = num_labels
        max_grad_norm: gradient clipping value
    """
    # Dev evaluation during training uses 0.5 threshold (optimization happens after)

    train_texts = train_df['comment_text'].apply(preprocess_text).tolist()
    train_labels = train_df[label_columns].values.astype(float)

    train_dataset = ToxicCommentDataset(train_texts, train_labels, tokenizer, max_length)
    generator = torch.Generator()
    generator.manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)

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

    if alpha_per_class is None:
        num_labels = len(label_columns)
        alpha_per_class = np.full(num_labels, 0.25)
        print("Warning: alpha_per_class not provided, using default 0.25 for all classes.", file=sys.stderr)

    # Convert to tensor for use in focal loss
    alpha_tensor = torch.tensor(alpha_per_class, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}...", file=sys.stderr)
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc="Training", file=sys.stderr)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = focal_loss_with_logits(
                logits, labels, alpha=alpha_tensor, gamma=gamma, reduction="sum"
            )

            loss.backward()

            # Clip gradients to prevent exploding gradients
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"  Average loss: {avg_loss:.4f}", file=sys.stderr)

        if dev_df is not None:
            dev_score, _ = evaluate_model(
                model, dev_df, label_columns, tokenizer, device,
                batch_size=batch_size, max_length=max_length, thresholds=None
            )
            print(f"  Dev Macro-F1 (0.5 threshold): {dev_score:.6f}", file=sys.stderr)
            if dev_score > best_dev_score:
                best_dev_score = dev_score
                print("  -> New best dev score!", file=sys.stderr)
                if save_best:
                    best_model_state = model.state_dict().copy()

        if save_dir:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dev_score': best_dev_score,
                'avg_loss': avg_loss,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}", file=sys.stderr)

    # Restore best model if requested
    if save_best and best_model_state is not None:
        print(f"\nRestoring best model (dev Macro-F1: {best_dev_score:.6f})...", file=sys.stderr)
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

    return model


# Hyperparameter tuning for alpha, gamma, and bias initialization
def tune_hyperparameters(
    train_df, dev_df, test_df, label_columns, tokenizer, device,
    max_length=256, save_dir=None
):
    """
    Tune:
        - α_max (alpha clipping upper bound)
        - gamma (focal loss gamma)
        - init_bias_flag (whether to run initialize_bias or leave default biases)
 
    NOTE: For tuning, the optimal threshold is computed on the TRAINING data 
    and then applied to the DEV data for the final F1 score evaluation.
    """

    if save_dir is None:
        log_path = "hparam_tuning.log"
    else:
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, "hparam_tuning.log")

    logging.basicConfig(
        filename=log_path,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    def log(msg):
        if "threshold" not in msg and "GLOBAL METRICS" not in msg and "AUC" not in msg:
             print(msg, file=sys.stderr)
        logging.info(msg)

    log("Tuning α_max, gamma, and init_bias_flag...")

    alpha_min = 0.05
    pi_min = 1e-4
    pi_max = 0.0959
    learning_rate = 2e-5
    batch_size = 32
    epochs = 3

    alpha_max_options = [0.5, 0.75, 0.9]
    gamma_options = [0.0, 2.0, 4.0]
    init_bias_options = [True, False]
    grad_clip_options = [1.0]

    label_counts = train_df[label_columns].sum(axis=0).values
    num_samples = len(train_df)

    best_f1 = -1.0
    best_params = {}

    total = (
        len(alpha_max_options)
        * len(gamma_options)
        * len(init_bias_options)
        * len(grad_clip_options)
    )
    current = 0

    for alpha_max in alpha_max_options:
        for gamma in gamma_options:
            for init_bias_flag in init_bias_options:
                for max_grad_norm in grad_clip_options:

                    current += 1
                    log(
                        f"[{current}/{total}] α_max={alpha_max}, gamma={gamma}, "
                        f"init_bias={init_bias_flag}, grad_clip={max_grad_norm}"
                    )
                    seed = 42
                    # Reset seeds for reproducibility of this parameter combination
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)

                    # Compute per-class alpha: rare classes get higher alpha (more weight on positives)
                    positive_rates = label_counts / num_samples
                    alpha_per_class = 1.0 - positive_rates
                    alpha_per_class = np.clip(alpha_per_class, alpha_min, alpha_max)

                    config = DistilBertConfig.from_pretrained(
                        'distilbert-base-uncased',
                        num_labels=len(label_columns),
                        problem_type="multi_label_classification",
                        seq_classif_dropout=0.2,
                    )

                    model = DistilBertForSequenceClassification.from_pretrained(
                        'distilbert-base-uncased',
                        config=config
                    )

                    if init_bias_flag:
                        initialize_bias(
                            model,
                            label_counts=label_counts,
                            num_samples=num_samples,
                            pi_min=pi_min,
                            pi_max=pi_max
                        )

                    model = model.to(device)

                    trained_model = train_model(
                        train_df, label_columns, tokenizer, model, device,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        max_length=max_length,
                        dev_df=None,
                        save_dir=None,
                        save_best=False,
                        alpha_per_class=alpha_per_class,
                        gamma=gamma,
                        max_grad_norm=max_grad_norm
                    )

                    optimal_thresholds = compute_training_thresholds(
                        trained_model, train_df, label_columns, tokenizer, device,
                        batch_size=batch_size, max_length=max_length, log=log
                    )

                    f1_macro, _ = evaluate_model(
                        trained_model, dev_df, label_columns, tokenizer, device,
                        batch_size=batch_size,
                        max_length=max_length,
                        thresholds=optimal_thresholds,
                        log=log
                    )

                    log(f"  Dev Macro-F1: {f1_macro:.6f}")
                    # Generate test predictions for this hyperparameter combination
                    predictions = predict(
                        model, test_df, label_columns, tokenizer, device,
                        batch_size=batch_size,
                        max_length=max_length
                    )
                    pred_file = (
                        f"pred_alpha{alpha_max}_gamma{gamma}_bias{init_bias_flag}_"
                        f"clip{max_grad_norm}.csv"
                    )
                    pred_path = os.path.join(save_dir, pred_file) if save_dir else pred_file

                    log(f"Saving predictions to {pred_path} ...")
                    predictions.to_csv(pred_path, index=False)

                    # Track best hyperparameter combination
                    if f1_macro > best_f1:
                        best_f1 = f1_macro
                        best_params = {
                            "alpha_min": alpha_min,
                            "alpha_max": alpha_max,
                            "gamma": gamma,
                            "init_bias": init_bias_flag,
                            "pi_min": pi_min,
                            "pi_max": pi_max,
                            "max_grad_norm": max_grad_norm,
                        }
                        log("  -> New best!")

                    del model
                    del trained_model
                    torch.cuda.empty_cache()

    log("\nBest Hyperparameters:")
    for k, v in best_params.items():
        log(f"  {k}: {v}")

    log(f"Best Dev Macro-F1: {best_f1:.6f}")

    return best_params


# Evaluation
def evaluate_model(
    model, df, label_columns, tokenizer, device,
    batch_size=32, max_length=256, log=None, thresholds=None
):
    """
    Evaluate model with:
        - per-class AUC
        - macro/micro/weighted metrics
        - Uses provided per-class thresholds for binary prediction.
        
    Note: Thresholds and detailed metrics are only written to the log file via `log`.
    """

    log_path = "hparam_tuning.log"

    if not logging.getLogger().handlers:
        logging.basicConfig(
            filename=log_path,
            filemode="a",
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )

    # Create default log function if none provided
    if log is None:
        def log(msg):
            print(msg, file=sys.stderr)
            logging.info(msg)

    model.eval()
    log("\n=== Running Evaluation ===")

    texts = df['comment_text'].apply(preprocess_text).tolist()
    labels = df[label_columns].values.astype(float)

    dataset = ToxicCommentDataset(texts, labels, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", file=sys.stderr):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()

            predictions.append(probs)
            true_labels.append(batch_labels)

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    if thresholds is not None:
        if len(thresholds) != len(label_columns):
            raise ValueError("Thresholds length must match number of label columns.")
    
        # Apply per-class thresholds for binary predictions
        thresholds_array = np.array(thresholds).reshape(1, -1)
        binary_preds = (predictions >= thresholds_array).astype(int)
        log("\nUsing Per-Class Thresholds:")
        for label, t in zip(label_columns, thresholds):
            log(f"  {label}: threshold={t:.4f}")
    else:
        binary_preds = (predictions >= 0.5).astype(int)
        log("\nUsing Default Threshold: 0.5 for all classes.")

    log("\nPer-Class AUC:")
    auc_scores = {}
    for i, label in enumerate(label_columns):
        try:
            # Skip AUC if class has only one unique value (can't compute ROC)
            if len(np.unique(true_labels[:, i])) < 2:
                auc = np.nan
            else:
                auc = roc_auc_score(true_labels[:, i], predictions[:, i])
        except ValueError:
            auc = np.nan

        auc_scores[label] = auc
        log(f"  {label}: {auc:.6f}" if not np.isnan(auc) else f"  {label}: N/A")

    mean_auc = np.nanmean([v for v in auc_scores.values()])
    log(f"\nMean AUC-ROC: {mean_auc:.6f}")

    log("\nPer-Class Precision/Recall/F1:")
    for i, label in enumerate(label_columns):
        p = precision_score(true_labels[:, i], binary_preds[:, i], zero_division=0)
        r = recall_score(true_labels[:, i], binary_preds[:, i], zero_division=0)
        f = f1_score(true_labels[:, i], binary_preds[:, i], zero_division=0)
        log(f"  {label}: P={p:.4f}, R={r:.4f}, F1={f:.4f}")

    # ------------------------------------------------------------
    # Global metrics
    # ------------------------------------------------------------
    macro_f1 = f1_score(true_labels, binary_preds, average='macro', zero_division=0)
    macro_precision = precision_score(true_labels, binary_preds, average='macro', zero_division=0)
    macro_recall = recall_score(true_labels, binary_preds, average='macro', zero_division=0)

    micro_f1 = f1_score(true_labels, binary_preds, average='micro', zero_division=0)
    micro_precision = precision_score(true_labels, binary_preds, average='micro', zero_division=0)
    micro_recall = recall_score(true_labels, binary_preds, average='micro', zero_division=0)

    weighted_f1 = f1_score(true_labels, binary_preds, average='weighted', zero_division=0)
    weighted_precision = precision_score(true_labels, binary_preds, average='weighted', zero_division=0)
    weighted_recall = recall_score(true_labels, binary_preds, average='weighted', zero_division=0)

    log("\n===== GLOBAL METRICS =====")
    log(f"Macro Precision: {macro_precision:.6f}")
    log(f"Macro Recall:    {macro_recall:.6f}")
    log(f"Macro F1:        {macro_f1:.6f}")

    log(f"Micro Precision: {micro_precision:.6f}")
    log(f"Micro Recall:    {micro_recall:.6f}")
    log(f"Micro F1:        {micro_f1:.6f}")

    log(f"Weighted Precision: {weighted_precision:.6f}")
    log(f"Weighted Recall:    {weighted_recall:.6f}")
    log(f"Weighted F1:        {weighted_f1:.6f}")

    log("\n=== Evaluation Complete ===")

    # Return macro_f1 and the thresholds used (0.5 array if None was passed)
    if thresholds is not None:
        return macro_f1, thresholds
    else:
        return macro_f1, np.full(len(label_columns), 0.5)


# Predict
def predict(
    model, test_df, label_columns, tokenizer, device,
    batch_size=32, max_length=256
):
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

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
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

# Main
def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(
        description=('BERT baseline: DistilBERT for toxic comment classification'))
    parser.add_argument('train_file', help='Path to training data CSV file')
    parser.add_argument('test_file', help='Path to test data CSV file')
    parser.add_argument('output_file', help='Path to output CSV file')
    parser.add_argument('--dev-file', type=str, default=None, help='Path to development set CSV file')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate (default: 2e-5)')
    parser.add_argument('--max-length', type=int, default=256, help='Maximum sequence length (default: 256)')
    parser.add_argument('--model-name', type=str, default='distilbert-base-uncased', help='HuggingFace model name')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--save-best', action='store_true', help='Save best model based on dev set performance')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint directory to resume training from')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning on dev set (requires --dev-file)')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Set alpha_max (focal loss weighting upper '
                        'bound) when not tuning (default: 0.9)')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Set gamma (focal loss focusing parameter) '
                        'when not tuning (default: 2.0)')
    parser.add_argument('--init-bias', type=str, default=None,
                        help='Set bias initialization flag when not tuning '
                        '(True/False, default: False)')
    parser.add_argument('--grad-clip', type=float, default=None,
                        help='Set gradient clipping value when not tuning '
                        '(default: 1.0)')

    args = parser.parse_args()

    def setup_main_log(log_path):
        logging.basicConfig(
            filename=log_path,
            filemode="a",
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )

        def log(msg):
            # Filter verbose metrics from stderr output, but log everything to file
            if "threshold" not in msg and "GLOBAL METRICS" not in msg and "AUC" not in msg and "Evaluation Complete" not in msg:
                 print(msg, file=sys.stderr)
            logging.info(msg)
        return log

    log_file_path = "hparam_tuning.log"
    log = setup_main_log(log_file_path)

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}", file=sys.stderr)

        print(f"Loading training data from {args.train_file}...", file=sys.stderr)
        train_df = load_data(args.train_file)

        print(f"Loading test data from {args.test_file}...", file=sys.stderr)
        test_df = load_data(args.test_file)

        dev_df = None
        if args.dev_file:
            print(f"Loading development data from {args.dev_file}...", file=sys.stderr)
            dev_df = load_data(args.dev_file)

        if 'comment_text' not in train_df.columns:
            raise ValueError("Training file must contain 'comment_text' column")
        if 'comment_text' not in test_df.columns:
            raise ValueError("Test file must contain 'comment_text' column")

        label_columns = [
            'toxic', 'severe_toxic', 'obscene',
            'threat', 'insult', 'identity_hate'
        ]

        missing_labels = [label for label in label_columns if label not in train_df.columns]
        if missing_labels:
            raise ValueError(f"Training file missing label columns: {missing_labels}")

        label_counts = train_df[label_columns].sum(axis=0).values
        num_samples = len(train_df)
    
        # Compute per-class alpha: rare classes get higher alpha (more weight on positive examples)
        positive_rates = label_counts / num_samples
        alpha_per_class = 1.0 - positive_rates
        alpha_per_class = np.clip(alpha_per_class, 0.05, 0.75)

        learning_rate = args.learning_rate
        batch_size = args.batch_size
        epochs = args.epochs
        dropout_rate = 0.2 
        tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
    
        chosen_gamma = 2.0
        pi_min_clip = 1e-4
        pi_max_clip = 0.9
        max_grad_norm = 1.0
        init_bias_flag = False

        best_combo = None
        if args.tune:
            if dev_df is None:
                raise ValueError("--tune requires --dev-file to be specified")
            # Warn if user provided manual parameters (they'll be ignored)
            if (args.alpha is not None or args.gamma is not None or
                    args.init_bias is not None or
                    args.grad_clip is not None):
                print("Warning: --tune is enabled. Ignoring --alpha, --gamma, "
                      "--init-bias, and --grad-clip parameters.",
                      file=sys.stderr)
            best_combo = tune_hyperparameters(
                train_df, dev_df, test_df, label_columns, tokenizer, device,
                max_length=args.max_length,
                save_dir=args.save_dir
            )

            if best_combo is not None:
                print("Selected combo from tuning:", best_combo, file=sys.stderr)
                # Extract best hyperparameters from tuning results
                alpha_min = best_combo['alpha_min']
                alpha_max = best_combo['alpha_max']
                alpha_per_class = 1.0 - positive_rates
                alpha_per_class = np.clip(alpha_per_class, alpha_min, alpha_max)
                chosen_gamma = best_combo['gamma']
                pi_min_clip = best_combo['pi_min']
                pi_max_clip = best_combo['pi_max']
                max_grad_norm = best_combo['max_grad_norm']
                init_bias_flag = best_combo['init_bias']
                return
        else:
            # Use user-provided parameters or defaults when not tuning
            default_alpha_max = 0.9
            default_gamma = 2.0
            default_init_bias = False
            default_grad_clip = 1.0

            alpha_max = args.alpha if args.alpha is not None else default_alpha_max
            chosen_gamma = args.gamma if args.gamma is not None else default_gamma
            max_grad_norm = args.grad_clip if args.grad_clip is not None else default_grad_clip
            
            if args.init_bias is not None:
                init_bias_flag = args.init_bias.lower() in ('true', '1', 'yes')
            else:
                init_bias_flag = default_init_bias

            # Compute per-class alpha values
            alpha_min = 0.05
            alpha_per_class = 1.0 - positive_rates
            alpha_per_class = np.clip(alpha_per_class, alpha_min, alpha_max)

        # Reset seeds before final model training for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        print(f"Loading model: {args.model_name} with dropout {dropout_rate}...", file=sys.stderr)
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
        
        if init_bias_flag:
            initialize_bias(
                model, label_counts, num_samples, 
                pi_min=pi_min_clip, pi_max=pi_max_clip
            )

        model.to(device)

        print(f"\nTraining final model with {epochs} epochs, batch_size={batch_size}, learning_rate={learning_rate}...",
              file=sys.stderr)

        model = train_model(
            train_df, label_columns, tokenizer, model, device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=args.max_length,
            dev_df=None,
            save_dir=args.save_dir,
            save_best=args.save_best,
            alpha_per_class=alpha_per_class,
            gamma=chosen_gamma,
            max_grad_norm=max_grad_norm
        )
        
        if dev_df is not None:
            # Find best thresholds on training data, then apply to dev set
            print("\nComputing optimal per-class thresholds on TRAINING data...", file=sys.stderr)
            optimal_train_thresholds = compute_training_thresholds(
                model, train_df, label_columns, tokenizer, device,
                batch_size=args.batch_size, max_length=args.max_length,
                log=log
            )

            print("\nEvaluating on DEVELOPMENT data using TRAINING thresholds (see log file for details)...", file=sys.stderr)
            dev_score, _ = evaluate_model(
                model, dev_df, label_columns, tokenizer, device,
                batch_size=args.batch_size, max_length=args.max_length,
                thresholds=optimal_train_thresholds,
                log=log
            )
            print(f"\nFinal Dev Macro-F1 (using Training Thresholds): {dev_score:.6f}", file=sys.stderr)

        print("\nMaking predictions on test set...", file=sys.stderr)
        predictions = predict(
            model, test_df, label_columns, tokenizer, device,
            batch_size=args.batch_size,
            max_length=args.max_length
        )

        print(f"Saving predictions (probabilities) to {args.output_file}...", file=sys.stderr)
        predictions.to_csv(args.output_file, index=False)

        print(f"Done! Predictions saved to {args.output_file}", file=sys.stderr)
        print("Predictions are probabilities (0.0 to 1.0) for each toxic category.", file=sys.stderr)

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
