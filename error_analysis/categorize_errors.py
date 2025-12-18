#!/usr/bin/env python3
"""
Categorize errors using CSV module instead of pandas.
"""

import csv
from collections import defaultdict

def categorize_error(comment, gold_labels, pred_labels):
    """Categorize error type."""
    if not comment:
        comment = ""
    comment_lower = comment.lower()
    words = comment.split()
    word_count = len(words)
    
    categories = []
    
    # Long comments (keep this)
    if word_count > 100:
        categories.append('long_comment')
    
    # Identity-based hate (slurs)
    slurs = ['nigger', 'fag', 'faggot', 'kike', 'chink', 'spic', 'wetback', 'retard']
    has_slur = any(slur in comment_lower for slur in slurs)
    if has_slur:
        categories.append('identity_based_hate')
    
    # Threat/violence
    threat_words = ['kill', 'die', 'murder', 'death', 'suicide', 'threat', 'violence', 'harm']
    if any(word in comment_lower for word in threat_words):
        categories.append('threat_violence')
    
    # Profanity/obscenity
    profanity_words = ['fuck', 'shit', 'damn', 'bitch', 'ass', 'bastard', 'crap']
    if any(word in comment_lower for word in profanity_words):
        categories.append('profanity_obscenity')
    
    # Mild toxicity
    mild_insults = ['idiot', 'stupid', 'moron', 'dumb', 'fool', 'clown', 'twit']
    if any(word in comment_lower for word in mild_insults):
        categories.append('mild_toxicity')
    
    # Context-dependent
    context_words = ['jew', 'jewish', 'israel', 'muslim', 'islam', 'christian', 'political', 
                     'academic', 'research', 'article', 'wikipedia', 'edit']
    if any(word in comment_lower for word in context_words) and not has_slur:
        categories.append('context_dependent')
    
    # Short comments
    if word_count < 5:
        categories.append('short_comment')
    
    # Severity confusion (toxic vs severe_toxic)
    toxic_idx = 0
    severe_idx = 1
    if (gold_labels[toxic_idx] != pred_labels[toxic_idx] or 
        gold_labels[severe_idx] != pred_labels[severe_idx]):
        if (gold_labels[toxic_idx] != pred_labels[toxic_idx] and 
            gold_labels[severe_idx] != pred_labels[severe_idx]):
            categories.append('severity_confusion')
        elif (gold_labels[toxic_idx] == 0 and pred_labels[severe_idx] == 1) or \
             (gold_labels[severe_idx] == 0 and pred_labels[toxic_idx] == 1):
            categories.append('severity_confusion')
    
    # Category boundary (obscene/insult/toxic confusion)
    obscene_idx = 2
    insult_idx = 4
    if (gold_labels[obscene_idx] != pred_labels[obscene_idx] and 
        (gold_labels[insult_idx] != pred_labels[insult_idx] or 
         gold_labels[toxic_idx] != pred_labels[toxic_idx])):
        categories.append('category_boundary')
    elif (gold_labels[insult_idx] != pred_labels[insult_idx] and 
          gold_labels[obscene_idx] != pred_labels[obscene_idx]):
        categories.append('category_boundary')
    
    # Default to other if no category
    if not categories:
        categories.append('other')
    
    return categories

def main():
    input_file = 'wrong_predictions/wrong_predictions_distilbert.csv'
    
    content_stats = defaultdict(int)
    examples_by_category = defaultdict(list)
    total_errors = 0
    
    label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_errors += 1
            
            try:
                gold_labels = [
                    int(float(row['toxic'])) if row['toxic'] else 0,
                    int(float(row['severe_toxic'])) if row['severe_toxic'] else 0,
                    int(float(row['obscene'])) if row['obscene'] else 0,
                    int(float(row['threat'])) if row['threat'] else 0,
                    int(float(row['insult'])) if row['insult'] else 0,
                    int(float(row['identity_hate'])) if row['identity_hate'] else 0
                ]
                pred_labels = [
                    int(float(row['toxic_pred'])) if row['toxic_pred'] else 0,
                    int(float(row['severe_toxic_pred'])) if row['severe_toxic_pred'] else 0,
                    int(float(row['obscene_pred'])) if row['obscene_pred'] else 0,
                    int(float(row['threat_pred'])) if row['threat_pred'] else 0,
                    int(float(row['insult_pred'])) if row['insult_pred'] else 0,
                    int(float(row['identity_hate_pred'])) if row['identity_hate_pred'] else 0
                ]
                
                comment = row.get('comment_text', '') or ''
                
                categories = categorize_error(comment, gold_labels, pred_labels)
                
                for cat in categories:
                    content_stats[cat] += 1
                    if len(examples_by_category[cat]) < 3:
                        examples_by_category[cat].append({
                            'comment': comment[:200],
                            'gold': gold_labels,
                            'pred': pred_labels
                        })
            except Exception as e:
                continue
    
    # Print statistics
    print(f"Total errors: {total_errors}")
    print(f"\nError Statistics by Content Category:\n")
    sorted_content = sorted(content_stats.items(), key=lambda x: x[1], reverse=True)
    for category, count in sorted_content:
        percentage = count / total_errors * 100
        print(f"{category:<30} {count:<10} ({percentage:.1f}%)")
    
    # Print examples
    print("\n\nExample Errors by Category:\n")
    for category in sorted(examples_by_category.keys()):
        print(f"\n{category.upper()} ({content_stats[category]} errors)")
        for i, example in enumerate(examples_by_category[category][:3], 1):
            print(f"\nExample {i}:")
            comment = example['comment']
            if len(comment) > 150:
                comment = comment[:150] + "..."
            print(f"Comment: {comment}")
            print(f"Gold: {dict(zip(label_names, example['gold']))}")
            print(f"Pred: {dict(zip(label_names, example['pred']))}")

if __name__ == "__main__":
    main()

