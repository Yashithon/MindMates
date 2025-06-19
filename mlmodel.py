from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Step 1: Preprocessing
df = pd.read_csv("disorders_dataset.csv")
df.dropna(subset=['text', 'mental_health_tag'], inplace=True)
df['text'] = df['text'].astype(str).str.lower()
df['mental_health_tag'] = df['mental_health_tag'].astype(str).str.lower()

# Step 2: Split data
X = df['text']
y = df['mental_health_tag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 4: Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, y_train)

# Step 5: Evaluate
y_pred = clf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Optional: Test single sentence
import numpy as np

def predict_tag(text):
    text = text.strip().lower()

    # Define list of inappropriate/swear words (expandable)
    banned_words = {"fuck", "shit", "damn", "bitch", "crap", "asshole", "dumb", "stupid", "kill", "die", "hell"}

    # Split input into words
    words = text.split()

    # Reject empty or too-short input
    if len(words) < 2:
        return "Please enter a complete sentence related to how you feel."

    # Reject if input is mostly profanity
    if all(word in banned_words for word in words):
        return "Please avoid using offensive or inappropriate words. Try describing how you feel instead."

    # Vectorize and predict
    text_vec = vectorizer.transform([text])
    prediction = clf.predict(text_vec)[0]
    return prediction
print(predict_tag("Guys! Itâ€™s finally my turn to announce losing 200 pounds in one day! I lost 200 pounds today!"))
# Save model and vectorizer
joblib.dump(clf, "mental_health_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
