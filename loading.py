from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("mental_health_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Final prediction function with input handling
def predict_tag(text):
    text = text.strip().lower()

    # Define a list of inappropriate words
    banned_words = {"fuck", "shit", "damn", "bitch", "crap", "asshole", "dumb", "stupid", "kill", "die", "hell"}

    words = text.split()

    # Reject too-short input
    if len(words) < 2:
        return "Please enter a complete sentence related to how you feel."

    # Reject if all words are profanity
    if all(word in banned_words for word in words):
        return "Please avoid using offensive or inappropriate language."

    # Vectorize and predict
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    return prediction

# Flask route
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        user_input = request.form.get("user_input")
        if user_input:
            prediction = predict_tag(user_input)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
