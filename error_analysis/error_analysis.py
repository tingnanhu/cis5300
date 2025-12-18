from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import re
import pandas as pd
import os

#nltk.download("stopwords")

LABELS = [
    'toxic', 'severe_toxic', 'obscene',
    'threat', 'insult', 'identity_hate'
]

thresholds = {
    'toxic': 0.5,
    'severe_toxic': 0.5,
    'obscene': 0.5,
    'threat': 0.5,
    'insult': 0.5,
    'identity_hate': 0.5
}

stop_words = set(stopwords.words("english"))

def binarize_predictions_per_class(pred_df, thresholds):
    bin_df = pred_df.copy()
    if 'id' in bin_df.columns:
        bin_df = bin_df.set_index('id')
    for label in LABELS:
        bin_df[label] = (bin_df[label] >= thresholds[label]).astype(int)
    return bin_df

print("Load dataset")
pred_df = pd.read_csv("pred_distilbert.csv")
bin_pred_df = binarize_predictions_per_class(pred_df, thresholds)
gold_df = pd.read_csv("cleaned/test_split.csv")

df = gold_df.merge(
    bin_pred_df.reset_index(),
    on="id",
    suffixes=("", "_pred")
)

def plot_wordcloud(texts, title, save_path):
    text = " ".join(texts).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [w for w in text.split() if w not in stop_words]
    clean_text = " ".join(tokens)

    wc = WordCloud(
        width=900,
        height=500,
        background_color="white",
        stopwords=STOPWORDS,
        max_words=200,
        collocations=False
    ).generate(clean_text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc)
    plt.axis("off")
    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved wordcloud image to: {save_path}")

output_dir = "wordcloud_images"
os.makedirs(output_dir, exist_ok=True)

for label in LABELS:
    pred_col = f"{label}_pred"
    texts = df[df[pred_col] == 1]["comment_text"].dropna().astype(str)
    if texts.empty:
        print(f"No positive predictions for class: {label}")
        continue
    filename = f"{label}_wordcloud.png"
    save_path = os.path.join(output_dir, filename)
    plot_wordcloud(
        texts,
        title=f"Predicted {label.replace('_', ' ').title()} Word Cloud",
        save_path=save_path
    )

# Identify rows where ANY label prediction differs from gold label
wrong_mask = pd.DataFrame({
    label: df[label] != df[f"{label}_pred"] for label in LABELS
}).any(axis=1)

wrong_df = df[wrong_mask]

# Select columns to save — for example, id, comment_text, all gold labels and predictions
cols_to_save = ["id", "comment_text"] + LABELS + [f"{label}_pred" for label in LABELS]

wrong_output_path = "wrong_predictions_distilbert.csv"
wrong_df.to_csv(wrong_output_path, columns=cols_to_save, index=False)
print(f"Saved all wrong predictions to {wrong_output_path}")
