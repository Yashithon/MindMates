import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

# Load your dataset
df = pd.read_csv("disorders_dataset.csv")

# Optional: lowercase the text
df['text'] = df['text'].astype(str).str.lower()

# Find top words per mental_health_tag
tags = df['mental_health_tag'].unique()

for tag in tags:
    print(f"\n🔹 Top words for: {tag}")
    texts = df[df['mental_health_tag'] == tag]['text']
    vectorizer = CountVectorizer(stop_words='english', max_features=20)
    word_matrix = vectorizer.fit_transform(texts)
    word_counts = word_matrix.sum(axis=0).A1
    vocab = vectorizer.get_feature_names_out()
    word_freq = dict(zip(vocab, word_counts))
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    for word, count in top_words:
        print(f"{word}: {count}")
def predict_tag(text):
    text = text.lower()
    text_vec = vectorizer.transform([text])
    prediction = clf.predict(text_vec)[0]
    return prediction

# Example:
new_sentence = "I can't stop overthinking and my chest feels tight"
print("Predicted tag:", predict_tag(new_sentence))

