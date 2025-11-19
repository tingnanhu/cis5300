# Strong Baseline: DistilBERT for Toxic Comment Classification

## Overview

The strong baseline uses **DistilBERT (a distilled version of BERT)** for multi-label toxic comment classification. DistilBERT is faster and smaller than BERT while maintaining most of its performance. This transformer-based model fine-tunes a pre-trained DistilBERT model to classify comments into six toxic categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. The model uses focal loss to handle class imbalance and supports hyperparameter tuning.

## How It Works

1. **Preprocessing**: Text comments are tokenized using DistilBERT tokenizer and truncated/padded to a fixed length (default: 256 tokens)
2. **Model Architecture**: Fine-tunes a pre-trained DistilBERT model (`distilbert-base-uncased` by default) with a classification head for multi-label classification
3. **Training**: Uses focal loss to handle class imbalance, AdamW optimizer with linear warmup scheduler, and gradient clipping
4. **Prediction**: Outputs probability scores (0.0 to 1.0) for each of the six toxic categories

### Key Features

- **Focal Loss**: Addresses class imbalance by focusing learning on hard examples
- **Multi-label Classification**: Each comment can have multiple toxic labels simultaneously
- **Hyperparameter Tuning**: Supports automatic hyperparameter search
- **Checkpointing**: Saves model checkpoints and best model based on dev set performance
- **Resume Training**: Can resume training from saved checkpoints

## Usage

### Basic Usage

```bash
python strong_baseline.py <train_file> <test_file> <output_file>
```

### With Development Set Evaluation

```bash
python strong_baseline.py <train_file> <test_file> <output_file> --dev-file <dev_file>
```

### With Best Model Saving

```bash
python strong_baseline.py <train_file> <test_file> <output_file> --dev-file <dev_file> --save-best
```

This will save the model with the best dev set AUC-ROC score.

### With Hyperparameter Tuning

```bash
python strong_baseline.py <train_file> <test_file> <output_file> --dev-file <dev_file> --tune
```

This will automatically search for the best hyperparameters (learning rate, batch size, epochs) and use them for training.

### With Checkpoint Saving

```bash
python strong_baseline.py <train_file> <test_file> <output_file> --save-dir checkpoints
```

This will save checkpoints after each epoch to the specified directory.

### Resuming from Checkpoint

```bash
python strong_baseline.py <train_file> <test_file> <output_file> --resume checkpoints
```

This will resume training from the latest checkpoint in the specified directory.

### Arguments

**Required Arguments:**
- `train_file`: Path to training data CSV file
  - Must contain: `id`, `comment_text`, and label columns
  - Label columns: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- `test_file`: Path to test data CSV file
  - Must contain: `id` and `comment_text` columns
- `output_file`: Path where predictions will be saved

**Optional Arguments:**
- `--dev-file PATH`: Path to development set CSV file
  - Must contain: `id`, `comment_text`, and label columns
  - Used for evaluation and hyperparameter tuning
- `--epochs N`: Number of training epochs (default: 3)
- `--batch-size N`: Batch size for training (default: 16)
- `--learning-rate FLOAT`: Learning rate for optimizer (default: 2e-5)
- `--max-length N`: Maximum sequence length for tokenization (default: 256)
- `--model-name STRING`: Pre-trained model name (default: 'distilbert-base-uncased')
  - Options: 'distilbert-base-uncased', 'distilbert-base-cased', etc.
- `--save-dir PATH`: Directory to save checkpoints (default: 'checkpoints')
- `--save-best`: Save the best model based on dev set performance (requires --dev-file)
- `--resume PATH`: Resume training from checkpoints in the specified directory
- `--tune`: Perform hyperparameter tuning (requires --dev-file)
  - Searches over: learning rates [2e-5, 5e-5], dropout rates [0.1, 0.3]
  - Uses fixed batch_size=32 and epochs=5 during tuning

### Examples

```bash
# Basic usage with default hyperparameters
python strong_baseline.py \
    cleaned/train_split.csv \
    cleaned/test_split.csv \
    results/strong_baseline_predictions.csv

# With dev set evaluation
python strong_baseline.py \
    cleaned/train_split.csv \
    cleaned/test_split.csv \
    results/strong_baseline_predictions.csv \
    --dev-file cleaned/dev_split.csv

# With best model saving
python strong_baseline.py \
    cleaned/train_split.csv \
    cleaned/test_split.csv \
    results/strong_baseline_predictions.csv \
    --dev-file cleaned/dev_split.csv \
    --save-best

# With hyperparameter tuning
python strong_baseline.py \
    cleaned/train_split.csv \
    cleaned/test_split.csv \
    results/strong_baseline_predictions.csv \
    --dev-file cleaned/dev_split.csv \
    --tune

# With custom hyperparameters
python strong_baseline.py \
    cleaned/train_split.csv \
    cleaned/test_split.csv \
    results/strong_baseline_predictions.csv \
    --epochs 5 \
    --batch-size 32 \
    --learning-rate 3e-5 \
    --max-length 512

# With checkpoint saving
python strong_baseline.py \
    cleaned/train_split.csv \
    cleaned/test_split.csv \
    results/strong_baseline_predictions.csv \
    --save-dir checkpoints/distilbert_baseline \
    --save-best \
    --dev-file cleaned/dev_split.csv
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
- Used for evaluation and hyperparameter tuning

### Test File (`test.csv`)
- Must contain: `id` and `comment_text` columns
- Other columns are ignored

### Output File (`strong_baseline.csv`)
- Contains columns: `id`, `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- All predictions are probabilities (0.0 to 1.0)

## Model Architecture

### Base Model
- **Pre-trained Model**: DistilBERT-base-uncased (default)
  - 6 transformer layers (distilled from BERT's 12 layers)
  - 768 hidden dimensions
  - 66M parameters (60% of BERT-base)
- **Classification Head**: Linear layer mapping DistilBERT output to 6 labels (one per toxic category)
- **Dropout Rate**: Configurable via `seq_classif_dropout` (default: 0.2, tuned via --tune)

### Loss Function
- **Focal Loss**: Modified binary cross-entropy that focuses on hard examples
  - Formula: `FL = α * (1 - pt)^γ * BCE`
  - Default parameters: `α = 0.25`, `γ = 2.0`
  - Helps handle class imbalance by down-weighting easy examples

### Training Details
- **Optimizer**: AdamW with weight decay
- **Learning Rate Schedule**: Linear warmup (0 warmup steps by default)
- **Gradient Clipping**: Max norm of 1.0
- **Device**: Automatically uses CUDA if available, otherwise CPU

## Sample Output

When running the script, you'll see output like:

```
Using device: cuda
Loading model: distilbert-base-uncased with dropout 0.2...
Epoch 1/3...
Training: 100%|████████████| 7979/7979 [15:23<00:00,  8.65it/s, loss=0.234]
  Average loss: 0.2156
  Macro-F1:      0.623456
  Macro-Precision: 0.712345
  Macro-Recall:  0.556789
  Dev AUC-ROC: 0.972350
  -> New best dev score!
  Saved checkpoint: checkpoints/checkpoint_epoch_1.pt

Epoch 2/3...
Training: 100%|████████████| 7979/7979 [15:20<00:00,  8.67it/s, loss=0.189]
  Average loss: 0.1823
  Macro-F1:      0.645678
  Macro-Precision: 0.723456
  Macro-Recall:  0.578901
  Dev AUC-ROC: 0.978123
  -> New best dev score!
  Saved checkpoint: checkpoints/checkpoint_epoch_2.pt

Epoch 3/3...
Training: 100%|████████████| 7979/7979 [15:18<00:00,  8.69it/s, loss=0.165]
  Average loss: 0.1567
  Macro-F1:      0.656789
  Macro-Precision: 0.734567
  Macro-Recall:  0.589012
  Dev AUC-ROC: 0.978456
  Saved checkpoint: checkpoints/checkpoint_epoch_3.pt

Restoring best model (dev AUC-ROC: 0.978456)
Saved best model to checkpoints/best_model.pt
Predicting: 100%|████████████| 1995/1995 [02:15<00:00, 14.75it/s]
Done! Saved to results/strong_baseline_predictions.csv
```

## Evaluation

To evaluate the strong baseline on test set:

```bash
python score.py cleaned/test_split.csv results/strong_baseline_predictions.csv
```

The evaluation script will output:
- **Mean AUC-ROC**: Primary metric (typically ~0.98-0.99 for DistilBERT baseline)
- **Macro-F1**: Macro-averaged F1 score
- **Macro-Precision**: Macro-averaged precision
- **Macro-Recall**: Macro-averaged recall
- **Individual AUC-ROC scores**: Per-label scores
- **Metrics JSON file**: All metrics saved to `results/strong_baseline_predictions_metrics.json`

Note: The model's `evaluate_model` function also prints Macro-F1, Macro-Precision, and Macro-Recall during training/evaluation.

### Example Evaluation Output

When evaluating with `score.py`, you'll see output like:

```
Mean AUC-ROC: 0.992305
Macro-F1: 0.644635
Macro-Precision: 0.731427
Macro-Recall: 0.586722

Individual AUC-ROC scores:
  toxic: 0.987539
  severe_toxic: 0.991336
  obscene: 0.994031
  threat: 0.995609
  insult: 0.988092
  identity_hate: 0.988117

Metrics saved to results/strong_baseline_predictions_metrics.json
```

## Expected Performance

The DistilBERT baseline typically achieves a **Mean AUC-ROC score around 0.98-0.99**, which represents state-of-the-art performance for this task. This is because:

1. **Pre-trained Representations**: DistilBERT provides rich contextualized word embeddings learned from large-scale text data (distilled from BERT)
2. **Fine-tuning**: The model is fine-tuned on the specific toxic comment classification task
3. **Focal Loss**: Effectively handles the severe class imbalance in the dataset
4. **Multi-label Architecture**: Properly models the multi-label nature of the task
5. **Efficiency**: DistilBERT is faster and more memory-efficient than BERT while maintaining similar performance

## Hyperparameter Tuning

When using `--tune`, the script searches over:

- **Learning Rates**: [2e-5, 5e-5]
- **Dropout Rates**: [0.1, 0.3]

Total combinations: 2 × 2 = 4 configurations

During tuning, the script uses fixed hyperparameters:
- **Batch Size**: 32
- **Epochs**: 5

The script evaluates each configuration on the dev set and selects the one with the highest AUC-ROC score. The best learning rate and dropout rate are then used for the final model training.

**Note**: Hyperparameter tuning requires a development set (`--dev-file`).

## Checkpointing

### Checkpoint Format

Checkpoints are saved as PyTorch state dictionaries containing:
- `epoch`: Current epoch number
- `model_state_dict`: Model parameters
- `optimizer_state_dict`: Optimizer state
- `scheduler_state_dict`: Learning rate scheduler state
- `best_dev_score`: Best dev set AUC-ROC score seen so far
- `avg_loss`: Average training loss for the epoch

### Best Model Saving

When using `--save-best`, the script saves:
- `best_model.pt`: PyTorch checkpoint with best model state
- `best_model_hf/`: Hugging Face format model and tokenizer (can be loaded with `from_pretrained()`)

## Hardware Requirements

- **GPU**: Recommended for training (CUDA-compatible GPU with at least 8GB VRAM)
- **CPU**: Can run but will be significantly slower
- **Memory**: At least 16GB RAM recommended
- **Storage**: ~300MB for model checkpoints, ~500MB for pre-trained DistilBERT model

### Training Time Estimates

- **Single epoch**: ~10-15 minutes on GPU (batch_size=16, ~128K training examples)
- **Full training (3 epochs)**: ~30-45 minutes on GPU
- **Hyperparameter tuning**: ~3-5 hours on GPU (4 configurations, 5 epochs each)

## Limitations

1. **Computational Cost**: Requires significant computational resources (GPU recommended)
2. **Training Time**: Takes longer to train than simpler baselines
3. **Memory Usage**: Requires substantial memory for large batch sizes
4. **Sequence Length**: Limited by max_length parameter (default 256 tokens)
5. **Overfitting**: May overfit on small datasets without proper regularization

## Tips for Best Results

1. **Use GPU**: Training is 10-20x faster on GPU
2. **Hyperparameter Tuning**: Use `--tune` to find optimal hyperparameters for your dataset
3. **Save Best Model**: Use `--save-best` to keep the best performing model
4. **Checkpointing**: Use `--save-dir` to save checkpoints for long training runs
5. **Batch Size**: Adjust based on available GPU memory (larger batch sizes may improve performance)
6. **Learning Rate**: Start with default (2e-5) and adjust if needed
7. **Max Length**: Increase if your comments are longer (at cost of memory and speed)

## Troubleshooting

### Out of Memory Errors
- Reduce `--batch-size` (try 8 or 4)
- Reduce `--max-length` (try 128)
- Use gradient accumulation (not currently implemented)

### Slow Training
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce batch size if causing memory issues
- DistilBERT is already optimized for speed; consider reducing max_length if needed

### Poor Performance
- Try hyperparameter tuning with `--tune`
- Increase number of epochs
- Check data quality and preprocessing
- Ensure dev set is representative

## References

1. **BERT Paper**: Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". *NAACL-HLT 2019*.

2. **DistilBERT Paper**: Sanh, V., et al. (2019). "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter". *NeurIPS EMC2 Workshop*.

3. **Focal Loss**: Lin, T. Y., et al. (2017). "Focal Loss for Dense Object Detection". *ICCV 2017*.

4. **Transformers Library**: Hugging Face Transformers - https://huggingface.co/transformers/

5. **Kaggle Competition**: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)

