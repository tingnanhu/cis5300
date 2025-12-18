# Step-by-Step Execution Guide

This guide provides step-by-step instructions to reproduce all results. For general project information, see the root `README.md`.

## Prerequisites

1. **Python 3.8+** with required packages (see root `README.md` for installation)
2. **Data files**: Training, development, and test CSV files in `data/` directory
3. **GPU (recommended)**: Scripts automatically detect and use GPU if available

## Step-by-Step Guide

### Step 1: Simple Baseline

The simple baseline predicts the majority class for each label.

```bash
python code/simple_baseline.py \
    data/train_split.csv \
    data/test_split.csv \
    output/simple_baseline_pred.csv \
    --dev-file data/dev_split.csv
```

**What it does:**
- Analyzes training data to find majority class for each label
- Predicts that class for all test examples
- Evaluates on dev set if provided

**Expected output:** Predictions saved to `output/simple_baseline_pred.csv`

### Step 2: Strong Baseline (DistilBERT)

The strong baseline uses DistilBERT with basic training.

```bash
python code/strong_baseline.py \
    data/train_split.csv \
    data/test_split.csv \
    output/strong_baseline_pred.csv \
    --dev-file data/dev_split.csv \
    --epochs 3 \
    --batch-size 32 \
    --learning-rate 2e-5
```

**What it does:**
- Fine-tunes DistilBERT on training data
- Evaluates on dev set
- Generates predictions on test set

**Expected output:** Predictions saved to `output/strong_baseline_pred.csv`

### Step 3: DistilBERT with Focal Loss and Hyperparameter Tuning

This model uses DistilBERT with focal loss and automatically tunes hyperparameters.

```bash
python code/distilbert.py \
    data/train_split.csv \
    data/test_split.csv \
    output/distilbert_tuned_pred.csv \
    --dev-file data/dev_split.csv \
    --tune
```

**What it does:**
- Tests multiple hyperparameter combinations:
  - `alpha_max`: [0.6, 0.75, 0.9]
  - `gamma`: [0.0, 2.0, 4.0]
  - `init_bias`: [True, False]
- For each combination:
  - Trains model for 3 epochs
  - Computes optimal thresholds on training data
  - Evaluates on dev set using Macro-F1
- Selects best hyperparameters based on dev Macro-F1
- Generates test predictions with best model

**Expected output:** 
- Predictions saved to `output/distilbert_tuned_pred.csv`
- Tuning results logged to `logs/hparam_tuning_distilbert.log`

**Note:** This process takes several hours depending on your hardware. The script will test 18 combinations (3 × 3 × 2).

### Step 4: RoBERTa with Focal Loss and Hyperparameter Tuning

This model uses RoBERTa with focal loss and automatically tunes hyperparameters.

```bash
python code/roberta.py \
    data/train_split.csv \
    data/test_split.csv \
    output/roberta_tuned_pred.csv \
    --dev-file data/dev_split.csv \
    --tune
```

**What it does:**
- Tests multiple hyperparameter combinations:
  - `alpha_max`: [0.6, 0.75, 0.9]
  - `gamma`: [0.0, 2.0, 4.0]
  - `dropout_rate`: [0.1, 0.3]
  - `init_bias`: [False]
  - `max_grad_norm`: [1.0]
- For each combination:
  - Trains model for 3 epochs
  - Computes optimal thresholds on training data
  - Evaluates on dev set using Macro-F1
- Selects best hyperparameters based on dev Macro-F1
- Generates test predictions with best model

**Expected output:**
- Predictions saved to `output/roberta_tuned_pred.csv`
- Tuning results logged to `logs/hparam_tuning_roberta.log`

**Note:** This process takes several hours. The script will test 36 combinations (3 × 3 × 2 × 1 × 1).


The prediction files in `output/` should already contain both predictions and gold labels.

### Step 6: Evaluate Results

Use the `score.py` script to evaluate all predictions. The primary metric is **Macro-F1**.

#### Evaluate RoBERTa Predictions

```bash
python code/score.py output/roberta_tuned_pred.csv
```

**Example output:**
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

Metrics saved to output/roberta_tuned_pred_metrics.json
```

#### Evaluate DistilBERT Predictions

```bash
python code/score.py output/distilbert_tuned_pred.csv
```

**Example output:**
```
Macro-F1: 0.681264
Macro-Precision: 0.672331
Macro-Recall: 0.698767
Mean AUC-ROC: 0.992068

Individual AUC-ROC scores:
  toxic: 0.988793
  severe_toxic: 0.988061
  obscene: 0.995195
  threat: 0.997637
  insult: 0.990034
  identity_hate: 0.992687

Metrics saved to output/distilbert_tuned_pred_metrics.json
```

#### Evaluate Strong Baseline Predictions

```bash
python code/score.py output/strong_baseline_pred.csv
```

#### Using Custom Thresholds

If you have optimal thresholds from training, you can specify them:

```bash
python code/score.py output/roberta_tuned_pred.csv \
    --thresholds 0.5435,0.5204,0.5858,0.6002,0.5711,0.6004
```

The thresholds are in order: `toxic,severe_toxic,obscene,threat,insult,identity_hate`

## Complete Workflow Example

Here's a complete example to reproduce all results:

```bash
# 1. Simple baseline
python code/simple_baseline.py \
    data/train_split.csv \
    data/test_split.csv \
    output/simple_baseline_pred.csv \
    --dev-file data/dev_split.csv

# 2. Strong baseline
python code/strong_baseline.py \
    data/train_split.csv \
    data/test_split.csv \
    output/strong_baseline_pred.csv \
    --dev-file data/dev_split.csv \
    --epochs 3

# 3. DistilBERT with tuning (takes several hours)
python code/distilbert.py \
    data/train_split.csv \
    data/test_split.csv \
    output/distilbert_tuned_pred.csv \
    --dev-file data/dev_split.csv \
    --tune

# 4. RoBERTa with tuning (takes several hours)
python code/roberta.py \
    data/train_split.csv \
    data/test_split.csv \
    output/roberta_tuned_pred.csv \
    --dev-file data/dev_split.csv \
    --tune

# 5. Evaluate all models
python code/score.py output/roberta_tuned_pred.csv
python code/score.py output/distilbert_tuned_pred.csv
python code/score.py output/strong_baseline_pred.csv
python code/score.py output/simple_baseline_pred.csv
```

## Training Without Hyperparameter Tuning

If you want to train with specific hyperparameters instead of tuning:

### DistilBERT with Manual Parameters

```bash
python code/distilbert.py \
    data/train_split.csv \
    data/test_split.csv \
    output/distilbert_pred.csv \
    --dev-file data/dev_split.csv \
    --alpha 0.9 \
    --gamma 2.0 \
    --init-bias False \
    --grad-clip 1.0 \
    --epochs 3
```

### RoBERTa with Manual Parameters

```bash
python code/roberta.py \
    data/train_split.csv \
    data/test_split.csv \
    output/roberta_pred.csv \
    --dev-file data/dev_split.csv \
    --alpha 0.9 \
    --gamma 2.0 \
    --init-bias False \
    --grad-clip 1.0 \
    --epochs 3
```

## Output Files

After running the models, you'll find:

- **Prediction files** in `output/`: CSV files with predictions and gold labels
- **Metrics files**: JSON files with evaluation metrics (generated by `score.py`)
- **Log files** in `logs/`: Training and tuning logs

For more details on output files, see the root `README.md`.

## Additional Documentation

For detailed information about each component:

- **Model Documentation**: `simple_baseline.md`, `strong_baseline.md`, `distilbert.md`, `roberta.md`
- **Evaluation**: `score.md` - Detailed evaluation metrics documentation
- **General Info**: Root `README.md` - Project overview, structure, and general information
- **Output Guide**: `output/README.md` - Guide for evaluating prediction files

