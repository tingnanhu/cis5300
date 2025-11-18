# Milestone 2 Results Summary

## Date: $(date)

## Simple Baseline Results

### Predictions Generated
-  **Test set predictions**: `results/simple_baseline_predictions.csv`
-  **Dev set predictions**: `results/simple_baseline_dev_predictions.csv`

### Training Statistics
From the training data (`cleaned/train_split.csv`):
- Total training examples: 127,656
- Majority class for each label:
  - `toxic`: 0 (115,418/127,656 = 90.41%)
  - `severe_toxic`: 0 (126,382/127,656 = 99.00%)
  - `obscene`: 0 (120,922/127,656 = 94.72%)
  - `threat`: 0 (127,252/127,656 = 99.68%)
  - `insult`: 0 (121,393/127,656 = 95.09%)
  - `identity_hate`: 0 (126,545/127,656 = 99.13%)

### Expected Performance
The simple baseline predicts the majority class (0) for all labels and all examples. This should result in:
- **Mean AUC-ROC ≈ 0.50** (random performance)
- This is expected because predicting the same class for all examples creates a diagonal ROC curve

### Evaluation Results
 **Evaluation completed successfully**

**Dev Set Performance:**
- Mean AUC-ROC: **0.500000**
- Individual scores:
  - toxic: 0.500000
  - severe_toxic: 0.500000
  - obscene: 0.500000
  - threat: 0.500000
  - insult: 0.500000
  - identity_hate: 0.500000

**Test Set Performance:**
- Mean AUC-ROC: **0.500000**
- Individual scores:
  - toxic: 0.500000
  - severe_toxic: 0.500000
  - obscene: 0.500000
  - threat: 0.500000
  - insult: 0.500000
  - identity_hate: 0.500000

**Analysis:** As expected, the simple baseline achieves random performance (AUC-ROC = 0.5) because it predicts the same class (0) for all examples across all labels. This confirms the baseline is working correctly and establishes a lower bound for model performance.

## Strong Baseline Results

 **Strong baseline completed successfully**

### Predictions Generated
-  **Test set predictions (default hyperparameters)**: `results/strong_baseline_predictions.csv`
-  **Test set predictions (tuned hyperparameters)**: `results/strong_baseline_tuned_predictions.csv`

### Hyperparameter Tuning Results
The hyperparameter tuning tested 9 combinations:
- `max_features`: [5000, 10000, 20000]
- `C`: [0.1, 1.0, 10.0]

**Best hyperparameters found:**
- `max_features`: 10000
- `C`: 1.0
- **Dev AUC-ROC**: 0.972350

Note: The best hyperparameters are actually the default values, indicating the defaults work well for this dataset.

### Performance Results

**Dev Set Performance (default hyperparameters):**
- Mean AUC-ROC: **0.972350**

**Test Set Performance (default hyperparameters):**
- Mean AUC-ROC: **0.978787**
- Individual scores:
  - toxic: 0.967539
  - severe_toxic: 0.977336
  - obscene: 0.984031
  - threat: 0.987609
  - insult: 0.978092
  - identity_hate: 0.978117

**Test Set Performance (tuned hyperparameters):**
- Mean AUC-ROC: **0.978787** (same as default, since best params = defaults)
- Individual scores: Same as above

**Analysis:** The strong baseline achieves excellent performance (~0.98 AUC-ROC), which is a huge improvement over the simple baseline (0.50). This demonstrates that using TF-IDF features and logistic regression is highly effective for this toxic comment classification task.

## Files Generated

### Predictions
- `results/simple_baseline_predictions.csv` - Simple baseline predictions for test set
- `results/simple_baseline_dev_predictions.csv` - Simple baseline predictions for dev set
- `results/strong_baseline_predictions.csv` - Strong baseline predictions for test set (default hyperparameters)
- `results/strong_baseline_tuned_predictions.csv` - Strong baseline predictions for test set (tuned hyperparameters)

### Output Logs
- `results/simple_baseline_output.txt` - Console output from simple baseline runs
- `results/strong_baseline_output.txt` - Console output from strong baseline run
- `results/strong_baseline_tuned_output.txt` - Console output from strong baseline tuning run

### Evaluation Results
- `results/simple_baseline_dev_scores.txt` -  Simple baseline dev set evaluation results
- `results/simple_baseline_test_scores.txt` -  Simple baseline test set evaluation results
- `results/strong_baseline_test_scores.txt` -  Strong baseline test set evaluation results
- `results/strong_baseline_tuned_test_scores.txt` -  Strong baseline (tuned) test set evaluation results

## Summary

1.  **sklearn installation fixed** - sklearn 1.7.2 installed for arm64 architecture

2.  **Simple baseline completed** - Generated predictions and evaluated on both dev and test sets
   - Performance: 0.50 AUC-ROC (random, as expected)

3.  **Strong baseline completed** - Generated predictions with both default and tuned hyperparameters
   - Performance: 0.978787 AUC-ROC on test set (excellent performance!)

4.  **All evaluations completed** - Both baselines evaluated and results saved

## Performance Comparison

| Baseline | Dev AUC-ROC | Test AUC-ROC | Improvement |
|----------|-------------|--------------|-------------|
| Simple (Majority Class) | 0.500000 | 0.500000 | Baseline |
| Strong (Logistic Regression + TF-IDF) | 0.972350 | 0.978787 | +95.8% |

The strong baseline shows a **95.8% improvement** over the simple baseline, demonstrating the effectiveness of using text features (TF-IDF) and machine learning (logistic regression) for toxic comment classification.

## Notes

- All prediction files are in CSV format with columns: `id`, `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- Simple baseline predictions are binary (0 or 1)
- Strong baseline predictions will be probabilities (0.0 to 1.0)
- The evaluation script (`score.py`) expects probabilities, but will work with binary predictions (though AUC-ROC will be ~0.5 for constant predictions)

