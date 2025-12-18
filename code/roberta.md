# RoBERTa Focal Loss Model - Usage Guide

This guide explains how to use the `roberta.py` script for toxic comment classification using RoBERTa with focal loss.

## What This Script Does

The script trains a RoBERTa model to classify toxic comments into six categories:
- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

The model uses **focal loss** to handle class imbalance, which helps the model focus on hard-to-classify examples. It also supports:
- Per-class alpha weighting (automatically computed from class frequencies)
- Configurable gamma parameter for focal loss
- Optional bias initialization based on class imbalance
- Automatic threshold optimization on training data
- Hyperparameter tuning with dropout rate and gradient clipping options

## Basic Usage

### Required Arguments

The script requires three positional arguments:

```bash
python roberta.py <train_file> <test_file> <output_file>
```

- `train_file`: Path to your training CSV file (must contain `comment_text` column and all six label columns)
- `test_file`: Path to your test CSV file (must contain `comment_text` column)
- `output_file`: Path where predictions will be saved (CSV format with probabilities for each class)

### Example: Basic Training

```bash
python roberta.py \
    data/train_split.csv \
    data/test_split.csv \
    output/predictions.csv
```

This will:
- Train for 3 epochs with default hyperparameters
- Use default alpha_max=0.9, gamma=2.0, no bias initialization
- Save predictions as probabilities (0.0 to 1.0) for each toxic category

## Optional Arguments

### Training Configuration

- `--epochs <int>`: Number of training epochs (default: 3)
- `--batch-size <int>`: Batch size for training (default: 32)
- `--learning-rate <float>`: Learning rate (default: 2e-5)
- `--max-length <int>`: Maximum sequence length for tokenization (default: 256)
- `--model-name <str>`: HuggingFace model name (default: 'roberta-base')
  - Options: 'roberta-base', 'roberta-large', etc.

### Model Saving

- `--save-dir <str>`: Directory to save model checkpoints (default: 'checkpoints')
- `--save-best`: Save the best model based on dev set performance (requires `--dev-file`)

### Development Set

- `--dev-file <path>`: Path to development set CSV file. If provided:
  - The script will evaluate on the dev set after training
  - Optimal per-class thresholds are computed on training data and applied to dev set
  - Detailed metrics are logged to `hparam_tuning.log`

## Two Modes of Operation

### Mode 1: Hyperparameter Tuning

When you have a development set and want to find the best hyperparameters automatically:

```bash
python roberta.py \
    data/train_split.csv \
    data/test_split.csv \
    output/predictions.csv \
    --dev-file data/dev_split.csv \
    --tune
```

**What happens:**
- The script tests multiple combinations of:
  - `alpha_max`: [0.6, 0.75, 0.9]
  - `gamma`: [0.0, 2.0, 4.0]
  - `init_bias`: [False]
  - `max_grad_norm`: [1.0]
  - `dropout_rate`: [0.1, 0.3]
- For each combination, it:
  - Trains a model for 3 epochs
  - Computes optimal thresholds on training data using precision-recall curves
  - Evaluates on dev set using those thresholds
  - Saves test predictions with a descriptive filename
- Selects the combination with the best dev Macro-F1 score
- Logs all results to `hparam_tuning.log` (or `save_dir/hparam_tuning.log` if `--save-dir` is used)
- **Note:** The script exits after tuning - it does not train a final model with the best parameters

**Important:** If you use `--tune`, any `--alpha`, `--gamma`, `--init-bias`, or `--grad-clip` arguments are ignored (with a warning).

### Mode 2: Manual Parameter Setting

When you want to set hyperparameters directly (or use defaults):

```bash
python roberta.py \
    data/train_split.csv \
    data/test_split.csv \
    output/predictions.csv \
    --dev-file data/dev_split.csv \
    --alpha 0.9 \
    --gamma 2.0 \
    --init-bias False \
    --grad-clip 1.0
```

**Parameter Descriptions:**

- `--alpha <float>`: Upper bound for per-class alpha values in focal loss (default: 0.9)
  - Lower values give more weight to positive examples
  - The actual per-class alpha is computed as `1.0 - positive_rate` and clipped between 0.05 and this value
  
- `--gamma <float>`: Focal loss focusing parameter (default: 2.0)
  - Higher values focus more on hard examples
  - `gamma=0.0` is equivalent to weighted binary cross-entropy
  
- `--init-bias <str>`: Whether to initialize output layer bias based on class imbalance (default: False)
  - Use `True`, `1`, or `yes` to enable
  - When enabled, bias is set to `-log((1 - pi) / pi)` where `pi` is the positive rate (clipped between 1e-4 and 0.9)
  
- `--grad-clip <float>`: Maximum gradient norm for clipping (default: 1.0)
  - Prevents exploding gradients during training
  - Set to 0 to disable gradient clipping

**If you don't specify these parameters**, the script uses the defaults:
- `alpha_max = 0.9`
- `gamma = 2.0`
- `init_bias = False`
- `grad_clip = 1.0`

## Complete Example: Full Training with Dev Evaluation

```bash
python roberta.py \
    data/train_split.csv \
    data/test_split.csv \
    output/final_predictions.csv \
    --dev-file data/dev_split.csv \
    --epochs 5 \
    --batch-size 16 \
    --learning-rate 2e-5 \
    --save-dir checkpoints \
    --save-best \
    --alpha 0.9 \
    --gamma 2.0 \
    --init-bias True \
    --grad-clip 1.0
```

This will:
1. Train for 5 epochs with batch size 16
2. Use the specified hyperparameters
3. Save checkpoints to `checkpoints/` directory
4. Save the best model (based on dev Macro-F1) to `checkpoints/best_model.pt`
5. Compute optimal thresholds on training data
6. Evaluate on dev set using those thresholds
7. Generate predictions on test set
8. Log detailed metrics to `hparam_tuning.log`

## Output Files

### Predictions File (`output_file`)
- CSV file with columns: `id`, `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- Values are probabilities (0.0 to 1.0) for each class
- Can be converted to binary predictions using thresholds (typically 0.5 or optimized thresholds)

### Log File (`hparam_tuning.log`)
- Contains detailed evaluation metrics:
  - Per-class AUC-ROC scores
  - Per-class Precision, Recall, F1
  - Macro/Micro/Weighted metrics
  - Optimal thresholds (when computed)
- Appends to existing log file

### Checkpoints (if `--save-dir` is used)
- `checkpoint_epoch_N.pt`: Model state after each epoch
- `best_model.pt`: Best model state (if `--save-best` is used)
- `best_model_hf/`: HuggingFace-compatible model directory

### Tuning Outputs (if `--tune` is used)
- `pred_alpha{alpha}_gamma{gamma}_bias{flag}_clip{clip}_dropout{rate}.csv`: Test predictions for each hyperparameter combination
- Saved in `save_dir/` if specified, otherwise in current directory

## Data Format Requirements

### Training File
Must contain:
- `comment_text`: The text of each comment
- `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`: Binary labels (0 or 1)
- Optionally: `id` column

### Test File
Must contain:
- `comment_text`: The text of each comment
- Optionally: `id` column (if present, will be included in output)

### Development File (if using `--dev-file`)
Same format as training file with all label columns.

## Key Features

### Focal Loss
The model uses focal loss to address class imbalance:
- **Alpha weighting**: Automatically computed per class based on positive rates
- **Gamma parameter**: Controls focus on hard examples (higher = more focus)
- **Formula**: `FL = -alpha * (1 - pt)^gamma * log(pt)` where `pt` is the predicted probability for the true class

### Threshold Optimization
When a dev set is provided, the script:
1. Computes optimal per-class thresholds on training data using precision-recall curves
2. Applies these thresholds to dev set for evaluation
3. Uses Macro-F1 as the primary evaluation metric

### Model Architecture
- Base model: RoBERTa (Robustly Optimized BERT Pretraining Approach)
- Classification head: Multi-label binary classification layer
- Dropout: Configurable dropout rates for hidden layers, attention, and classifier

## Evaluation Metrics

The script evaluates models using:
- **Macro-F1**: Primary metric (macro-averaged F1 across all labels)
- **Macro-Precision**: Complementary metric
- **Macro-Recall**: Complementary metric
- **Mean AUC-ROC**: Threshold-independent ranking metric
- **Per-class metrics**: Individual scores for each toxic category

All metrics are logged to `hparam_tuning.log` for detailed analysis.

