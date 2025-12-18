#!/usr/bin/env python3
"""
Compare RoBERTa errors with strong baseline to identify improvements and regressions.
"""

import pandas as pd
import numpy as np

def main():
    # Load predictions
    roberta_df = pd.read_csv('../output/roberta_tuned_pred.csv')
    baseline_df = pd.read_csv('../output/strong_baseline_pred.csv')
    
    # Load wrong predictions for RoBERTa
    wrong_roberta = pd.read_csv('wrong_predictions_roberta.csv')
    
    # Merge to compare (only merge prediction columns, keep gold from roberta)
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Select only prediction columns from baseline
    baseline_pred_cols = ['id'] + label_cols
    baseline_pred_df = baseline_df[baseline_pred_cols].copy()
    baseline_pred_df.columns = ['id'] + [f'{l}_baseline' for l in label_cols]
    
    # Select prediction and gold columns from roberta
    roberta_cols = ['id'] + label_cols + [f'gold_{l}' for l in label_cols]
    roberta_pred_df = roberta_df[roberta_cols].copy()
    roberta_pred_df.columns = ['id'] + [f'{l}_roberta' for l in label_cols] + [f'gold_{l}' for l in label_cols]
    
    # Merge
    merged = roberta_pred_df.merge(baseline_pred_df, on='id')
    
    # Load comment text from test file
    try:
        test_df = pd.read_csv('../data/test_split.csv')
        if 'comment_text' in test_df.columns:
            merged = merged.merge(test_df[['id', 'comment_text']], on='id', how='left')
        else:
            merged['comment_text'] = ''
    except:
        merged['comment_text'] = ''
    
    # Convert predictions to binary (threshold 0.5)
    roberta_correct = {}
    baseline_correct = {}
    roberta_wrong = {}
    baseline_wrong = {}
    
    for label in label_cols:
        gold = merged[f'gold_{label}'].values
        roberta_pred = (merged[f'{label}_roberta'] >= 0.5).astype(int).values
        baseline_pred = (merged[f'{label}_baseline'] >= 0.5).astype(int).values
        
        roberta_correct[label] = (roberta_pred == gold).sum()
        baseline_correct[label] = (baseline_pred == gold).sum()
        roberta_wrong[label] = (roberta_pred != gold).sum()
        baseline_wrong[label] = (baseline_pred != gold).sum()
    
    print("="*80)
    print("COMPARISON: RoBERTa vs Strong Baseline")
    print("="*80)
    
    print("\n## Error Counts by Label\n")
    print(f"{'Label':<20} {'RoBERTa Errors':<20} {'Baseline Errors':<20} {'Improvement':<15}")
    print("-" * 75)
    
    for label in label_cols:
        roberta_err = roberta_wrong[label]
        baseline_err = baseline_wrong[label]
        improvement = baseline_err - roberta_err
        improvement_pct = (improvement / baseline_err * 100) if baseline_err > 0 else 0
        print(f"{label:<20} {roberta_err:<20} {baseline_err:<20} {improvement:>5} ({improvement_pct:>5.1f}%)")
    
    # Find cases where RoBERTa is correct but baseline is wrong
    print("\n## Cases Where RoBERTa Correctly Identifies Toxicity That Baseline Misses\n")
    
    roberta_better = []
    for idx, row in merged.iterrows():
        for label in label_cols:
            gold = int(row[f'gold_{label}'])
            roberta_pred = int((row[f'{label}_roberta'] >= 0.5))
            baseline_pred = int((row[f'{label}_baseline'] >= 0.5))
            
            # RoBERTa correct, baseline wrong, and it's a positive case (toxic)
            if gold == 1 and roberta_pred == 1 and baseline_pred == 0:
                comment = str(row.get('comment_text', ''))[:200]
                roberta_better.append({
                    'id': row['id'],
                    'label': label,
                    'comment': comment,
                    'gold': gold,
                    'roberta_pred': roberta_pred,
                    'baseline_pred': baseline_pred
                })
                if len(roberta_better) >= 10:
                    break
        if len(roberta_better) >= 10:
            break
    
    for i, case in enumerate(roberta_better[:5], 1):
        print(f"\nExample {i} - {case['label']}:")
        comment = case['comment']
        if len(comment) > 150:
            comment = comment[:150] + "..."
        print(f"Comment: {comment}")
        print(f"Gold: {case['gold']}, RoBERTa: {case['roberta_pred']}, Baseline: {case['baseline_pred']}")
    
    # Find cases where baseline is correct but RoBERTa is wrong
    print("\n## Cases Where Baseline Correctly Identifies Toxicity That RoBERTa Misses\n")
    
    baseline_better = []
    for idx, row in merged.iterrows():
        for label in label_cols:
            gold = int(row[f'gold_{label}'])
            roberta_pred = int((row[f'{label}_roberta'] >= 0.5))
            baseline_pred = int((row[f'{label}_baseline'] >= 0.5))
            
            # Baseline correct, RoBERTa wrong, and it's a positive case (toxic)
            if gold == 1 and baseline_pred == 1 and roberta_pred == 0:
                comment = str(row.get('comment_text', ''))[:200]
                baseline_better.append({
                    'id': row['id'],
                    'label': label,
                    'comment': comment,
                    'gold': gold,
                    'roberta_pred': roberta_pred,
                    'baseline_pred': baseline_pred
                })
                if len(baseline_better) >= 10:
                    break
        if len(baseline_better) >= 10:
            break
    
    for i, case in enumerate(baseline_better[:5], 1):
        print(f"\nExample {i} - {case['label']}:")
        comment = case['comment']
        if len(comment) > 150:
            comment = comment[:150] + "..."
        print(f"Comment: {comment}")
        print(f"Gold: {case['gold']}, RoBERTa: {case['roberta_pred']}, Baseline: {case['baseline_pred']}")
    
    # Summary
    print("\n## Summary\n")
    print(f"Cases where RoBERTa is better: {len(roberta_better)} (found in sample)")
    print(f"Cases where Baseline is better: {len(baseline_better)} (found in sample)")
    
    total_roberta_errors = sum(roberta_wrong.values())
    total_baseline_errors = sum(baseline_wrong.values())
    total_improvement = total_baseline_errors - total_roberta_errors
    
    print(f"\nTotal errors:")
    print(f"  RoBERTa: {total_roberta_errors}")
    print(f"  Baseline: {total_baseline_errors}")
    print(f"  Improvement: {total_improvement} ({total_improvement/total_baseline_errors*100:.1f}% reduction)")

if __name__ == "__main__":
    main()

