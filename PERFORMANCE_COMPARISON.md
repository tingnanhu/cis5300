# Performance Comparison: strong_baseline.py vs distilbert_weighted_focal.py

## Summary

While both scripts use similar base parameters (DistilBERT, focal loss, same training setup), `distilbert_weighted_focal.py` achieves significantly better F1 scores (0.689810 Macro-F1) compared to `strong_baseline.py`. The key differences are:

1. **Per-class alpha weighting** (most important)
2. **Optimal threshold optimization**
3. **Loss normalization strategy**
4. **Bias initialization option**

## Key Differences

### 1. Focal Loss Alpha Weighting

#### `strong_baseline.py`:
```python
# Fixed alpha for all classes
loss = focal_loss_with_logits(logits, labels)  # alpha=0.25 (default)
```

- Uses **fixed alpha=0.25** for all 6 toxicity classes
- All classes receive equal weighting regardless of class imbalance
- Rare classes (e.g., `threat`, `severe_toxic`) get the same weight as common ones

#### `distilbert_weighted_focal.py`:
```python
# Per-class alpha based on class frequency
positive_rates = label_counts / num_samples
alpha_per_class = 1.0 - positive_rates  # Rare classes get higher alpha
alpha_per_class = np.clip(alpha_per_class, 0.05, 0.75)

loss = focal_loss_with_logits(
    logits, labels, 
    alpha=alpha_tensor,  # Per-class alpha tensor
    gamma=gamma, 
    reduction="sum"
)
```

- Computes **per-class alpha** based on class frequency
- Rare classes (low positive rate) get **higher alpha** (more weight on positive examples)
- Example: If `threat` has 0.5% positive rate, alpha ≈ 0.995 (clipped to 0.75)
- If `toxic` has 10% positive rate, alpha ≈ 0.90
- This directly addresses class imbalance during training

**Impact**: This is the **most critical difference**. By giving rare classes more weight, the model learns to better detect minority classes, which directly improves Macro-F1 (which treats all classes equally).

### 2. Threshold Optimization

#### `strong_baseline.py`:
```python
# Fixed 0.5 threshold for all classes
binary_preds = (predictions >= 0.5).astype(int)
```

- Uses **0.5 threshold** for all classes
- No optimization based on class characteristics
- Suboptimal for imbalanced classes

#### `distilbert_weighted_focal.py`:
```python
# Optimal thresholds computed on training data
optimal_thresholds = compute_training_thresholds(
    trained_model, train_df, label_columns, tokenizer, device,
    batch_size=batch_size, max_length=max_length, log=log
)

# Apply per-class thresholds
thresholds_array = np.array(thresholds).reshape(1, -1)
binary_preds = (predictions >= thresholds_array).astype(int)
```

- Computes **optimal per-class thresholds** using precision-recall curves
- Finds threshold that maximizes F1 for each class on training data
- From the log: thresholds range from 0.36 (severe_toxic) to 0.55 (identity_hate)
- These thresholds are then applied to dev/test sets

**Impact**: Different classes have different optimal operating points. Using class-specific thresholds significantly improves F1 scores, especially for rare classes.

### 3. Loss Normalization

#### `strong_baseline.py`:
```python
if reduction == 'mean':
    return loss.mean()  # Simple average
```

- Normalizes by total number of examples
- Can be dominated by common classes

#### `distilbert_weighted_focal.py`:
```python
if reduction == 'mean':
    # Normalize by number of positives to keep the loss scale consistent
    num_positives = targets.sum()
    return loss.sum() / (num_positives + 1e-7)
```

- Normalizes by **number of positive examples**
- Keeps loss scale consistent across batches with different positive ratios
- Prevents rare classes from being overwhelmed by common classes

**Impact**: Ensures rare classes contribute meaningfully to the loss signal.

### 4. Bias Initialization

#### `strong_baseline.py`:
- No bias initialization
- Model starts with default (usually zero) biases

#### `distilbert_weighted_focal.py`:
```python
if init_bias_flag:
    initialize_bias(
        model, label_counts, num_samples, 
        pi_min=pi_min_clip, pi_max=pi_max_clip
    )
```

- Optionally initializes output layer biases based on class frequencies
- Sets bias so that `sigmoid(bias) ≈ positive_rate`
- Helps model start closer to the right calibration

**Impact**: While the best configuration in the log used `init_bias=False`, this feature can help in some cases.

### 5. Evaluation Metric Focus

#### `strong_baseline.py`:
- Primary metric: **Mean AUC-ROC**
- F1 is computed but not optimized
- Uses fixed 0.5 threshold

#### `distilbert_weighted_focal.py`:
- Primary metric: **Macro-F1** with optimized thresholds
- Hyperparameter tuning optimizes for Macro-F1
- Thresholds optimized to maximize F1

**Impact**: Direct optimization for F1 leads to better F1 scores.

## Quantitative Impact

From the tuning log, the best configuration achieved:
- **Macro-F1: 0.689810** (with `alpha_max=0.5`, `gamma=2.0`, `init_bias=False`)

Key observations from the log:
1. **Thresholds vary significantly**: 
   - `severe_toxic`: 0.3646 (very low - rare class)
   - `toxic`: 0.4717 (moderate)
   - `identity_hate`: 0.5546 (higher)

2. **Per-class F1 scores**:
   - `toxic`: 0.8416
   - `severe_toxic`: 0.5294 (improved by low threshold)
   - `threat`: 0.5263 (improved by threshold optimization)
   - `obscene`: 0.8465
   - `insult`: 0.7770
   - `identity_hate`: 0.6181

3. **Without threshold optimization**, rare classes would have much lower F1 scores.

## Why These Differences Matter for F1

1. **Macro-F1 treats all classes equally**: It's the average of per-class F1 scores. Rare classes have equal weight, so improving them directly improves Macro-F1.

2. **Per-class alpha weighting**: By giving rare classes more weight during training, the model learns better representations for them, improving their individual F1 scores.

3. **Threshold optimization**: Rare classes often need lower thresholds (more lenient) to achieve good recall. Using 0.5 for all classes hurts rare classes disproportionately.

4. **Loss normalization**: Normalizing by positives ensures rare classes contribute meaningfully to the loss, preventing the model from ignoring them.

## Conclusion

The performance difference is **not** due to different base parameters, but rather due to:

1. **Class-imbalance handling**: Per-class alpha weighting directly addresses the severe class imbalance
2. **Post-processing optimization**: Optimal threshold selection maximizes F1 for each class
3. **Training signal balance**: Loss normalization ensures all classes contribute to learning

These techniques are specifically designed to improve performance on imbalanced multi-label classification tasks, which is why `distilbert_weighted_focal.py` achieves significantly better F1 scores despite using the same base model architecture and similar hyperparameters.

