#!/usr/bin/env python3
"""
Error analysis script for RoBERTa predictions.
Categorizes and analyzes types of errors made by the model.
"""

import pandas as pd
import numpy as np
import re
import sys
from collections import defaultdict

def categorize_error_type(comment, gold_labels, pred_labels):
    """Categorize the type of error based on comment content and label mismatch."""
    if pd.isna(comment) or comment is None:
        comment = ""
    comment = str(comment)
    comment_lower = comment.lower()
    words = comment.split()
    word_count = len(words)
    
    # Get mismatched labels
    mismatches = []
    label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    for i, label in enumerate(label_names):
        if gold_labels[i] != pred_labels[i]:
            mismatches.append((label, gold_labels[i], pred_labels[i]))
    
    error_type = []
    
    # False Negative: Model predicted 0 but should be 1
    # False Positive: Model predicted 1 but should be 0
    
    for label, gold, pred in mismatches:
        if gold == 1 and pred == 0:
            error_type.append(f'FN_{label}')
        elif gold == 0 and pred == 1:
            error_type.append(f'FP_{label}')
    
    # Categorize by content characteristics (priority order - most specific first)
    content_categories = []
    
    # Check for severity confusion (toxic vs severe_toxic mismatch)
    toxic_idx = 0
    severe_toxic_idx = 1
    has_severity_confusion = False
    # Check if there's a mismatch in toxic or severe_toxic
    toxic_mismatch = gold_labels[toxic_idx] != pred_labels[toxic_idx]
    severe_mismatch = gold_labels[severe_toxic_idx] != pred_labels[severe_toxic_idx]
    
    # If both are mismatched, it's likely severity confusion
    if toxic_mismatch and severe_mismatch:
        has_severity_confusion = True
    # Or if one is predicted when the other should be
    elif (gold_labels[toxic_idx] == 0 and pred_labels[severe_toxic_idx] == 1) or \
         (gold_labels[severe_toxic_idx] == 0 and pred_labels[toxic_idx] == 1):
        has_severity_confusion = True
    
    # Check for category boundary confusion (obscene/insult/toxic confusion)
    obscene_idx = 2
    insult_idx = 4
    has_category_confusion = False
    if (gold_labels[obscene_idx] != pred_labels[obscene_idx] and 
        (gold_labels[insult_idx] != pred_labels[insult_idx] or gold_labels[toxic_idx] != pred_labels[toxic_idx])):
        has_category_confusion = True
    elif (gold_labels[insult_idx] != pred_labels[insult_idx] and 
          gold_labels[obscene_idx] != pred_labels[obscene_idx]):
        has_category_confusion = True
    
    # Long comments (keep this category)
    if word_count > 100:
        content_categories.append('long_comment')
    
    # Identity-based hate (slurs)
    slurs = ['nigger', 'fag', 'faggot', 'kike', 'chink', 'spic', 'wetback', 'retard']
    has_slur = any(slur in comment_lower for slur in slurs)
    if has_slur:
        content_categories.append('identity_based_hate')
    
    # Threat/violence
    threat_words = ['kill', 'die', 'murder', 'death', 'suicide', 'threat', 'violence', 'harm']
    has_threat = any(word in comment_lower for word in threat_words)
    if has_threat:
        content_categories.append('threat_violence')
    
    # Profanity/obscenity
    profanity_words = ['fuck', 'shit', 'damn', 'bitch', 'ass', 'bastard', 'crap']
    has_profanity = any(word in comment_lower for word in profanity_words)
    if has_profanity:
        content_categories.append('profanity_obscenity')
    
    # Mild toxicity (subtle insults)
    mild_insults = ['idiot', 'stupid', 'moron', 'dumb', 'fool', 'clown', 'twit']
    has_mild_insult = any(word in comment_lower for word in mild_insults)
    if has_mild_insult:
        content_categories.append('mild_toxicity')
    
    # Context-dependent (political, academic, religious/ethnic references)
    context_words = ['jew', 'jewish', 'israel', 'muslim', 'islam', 'christian', 'political', 
                     'academic', 'research', 'article', 'wikipedia', 'edit']
    has_context = any(word in comment_lower for word in context_words)
    if has_context and not has_slur:  # Don't double-count identity-based hate
        content_categories.append('context_dependent')
    
    # Short comments
    if word_count < 5:
        content_categories.append('short_comment')
    
    # Severity confusion (if not already categorized)
    if has_severity_confusion and not content_categories:
        content_categories.append('severity_confusion')
    elif has_severity_confusion:
        # Add as secondary category if primary exists
        content_categories.append('severity_confusion')
    
    # Category boundary confusion (if not already categorized)
    if has_category_confusion and not content_categories:
        content_categories.append('category_boundary')
    elif has_category_confusion:
        # Add as secondary category if primary exists
        content_categories.append('category_boundary')
    
    # If no specific category found, use general categories
    if not content_categories:
        if has_severity_confusion:
            content_categories.append('severity_confusion')
        elif has_category_confusion:
            content_categories.append('category_boundary')
        else:
            content_categories.append('other')
    
    return error_type, content_categories, mismatches

def main():
    # Load wrong predictions
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'wrong_predictions_roberta.csv'
    
    df = pd.read_csv(input_file)
    
    print(f"Total wrong predictions: {len(df)}")
    print(f"Total test examples: ~15,960 (estimated)")
    print(f"Error rate: {len(df)/15960*100:.2f}%")
    print("\n" + "="*80)
    
    # Analyze errors by type
    error_stats = defaultdict(int)
    content_stats = defaultdict(int)
    label_error_counts = defaultdict(lambda: {'FN': 0, 'FP': 0})
    
    examples_by_category = defaultdict(list)
    
    label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    for idx, row in df.iterrows():
        try:
            gold_labels = [
                int(float(row['toxic'])) if not pd.isna(row['toxic']) else 0,
                int(float(row['severe_toxic'])) if not pd.isna(row['severe_toxic']) else 0,
                int(float(row['obscene'])) if not pd.isna(row['obscene']) else 0,
                int(float(row['threat'])) if not pd.isna(row['threat']) else 0,
                int(float(row['insult'])) if not pd.isna(row['insult']) else 0,
                int(float(row['identity_hate'])) if not pd.isna(row['identity_hate']) else 0
            ]
            pred_labels = [
                int(float(row['toxic_pred'])) if not pd.isna(row['toxic_pred']) else 0,
                int(float(row['severe_toxic_pred'])) if not pd.isna(row['severe_toxic_pred']) else 0,
                int(float(row['obscene_pred'])) if not pd.isna(row['obscene_pred']) else 0,
                int(float(row['threat_pred'])) if not pd.isna(row['threat_pred']) else 0,
                int(float(row['insult_pred'])) if not pd.isna(row['insult_pred']) else 0,
                int(float(row['identity_hate_pred'])) if not pd.isna(row['identity_hate_pred']) else 0
            ]
            
            comment_text = str(row['comment_text']) if not pd.isna(row['comment_text']) else ""
            
            error_types, content_cats, mismatches = categorize_error_type(
                comment_text, gold_labels, pred_labels
            )
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
        
        for et in error_types:
            error_stats[et] += 1
        
        for cc in content_cats:
            content_stats[cc] += 1
        
        for label, gold, pred in mismatches:
            if gold == 1 and pred == 0:
                label_error_counts[label]['FN'] += 1
            elif gold == 0 and pred == 1:
                label_error_counts[label]['FP'] += 1
        
        # Store examples (limit to 3 per category)
        for cc in content_cats:
            if len(examples_by_category[cc]) < 3:
                examples_by_category[cc].append({
                    'comment': str(row['comment_text'])[:200],
                    'gold': gold_labels,
                    'pred': pred_labels,
                    'mismatches': mismatches
                })
    
    # Print error statistics by label
    print("\n## Error Statistics by Label\n")
    print(f"{'Label':<20} {'False Negatives':<20} {'False Positives':<20} {'Total Errors':<15}")
    print("-" * 75)
    
    for label in label_names:
        fn = label_error_counts[label]['FN']
        fp = label_error_counts[label]['FP']
        total = fn + fp
        print(f"{label:<20} {fn:<20} {fp:<20} {total:<15}")
    
    # Print content-based statistics
    print("\n## Error Statistics by Content Category\n")
    sorted_content = sorted(content_stats.items(), key=lambda x: x[1], reverse=True)
    for category, count in sorted_content:
        percentage = count / len(df) * 100
        print(f"{category:<30} {count:<10} ({percentage:.1f}%)")
    
    # Print examples
    print("\n## Example Errors by Category\n")
    
    for category in sorted(examples_by_category.keys()):
        print(f"\n### {category.upper()} ({content_stats[category]} errors)")
        for i, example in enumerate(examples_by_category[category][:3], 1):
            print(f"\nExample {i}:")
            comment = example['comment']
            if len(comment) > 150:
                comment = comment[:150] + "..."
            print(f"Comment: {comment}")
            print(f"Gold labels: {dict(zip(label_names, example['gold']))}")
            print(f"Predicted:   {dict(zip(label_names, example['pred']))}")
            mismatches_str = ", ".join([f"{label}(gold={g}, pred={p})" 
                                       for label, g, p in example['mismatches']])
            print(f"Mismatches: {mismatches_str}")
    
    # Overall error patterns
    print("\n## Overall Error Patterns\n")
    
    total_fn = sum(label_error_counts[label]['FN'] for label in label_names)
    total_fp = sum(label_error_counts[label]['FP'] for label in label_names)
    
    print(f"Total False Negatives (missed toxic): {total_fn} ({total_fn/(total_fn+total_fp)*100:.1f}%)")
    print(f"Total False Positives (over-predicted): {total_fp} ({total_fp/(total_fn+total_fp)*100:.1f}%)")
    
    # Most common error types
    print("\n## Most Common Error Types\n")
    sorted_errors = sorted(error_stats.items(), key=lambda x: x[1], reverse=True)
    for error_type, count in sorted_errors[:10]:
        percentage = count / len(df) * 100
        print(f"{error_type:<30} {count:<10} ({percentage:.1f}%)")

if __name__ == "__main__":
    main()

