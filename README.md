# MindMates
This web app allows users to input a sentence or a short piece of text. Based on the content, it predicts which mental health category (such as depression, anxiety, etc.) the text most likely falls into. It uses a machine learning model trained on real-world forum data to analyze and classify mental health-related content.
This is a personal project that classifies user-input text into various mental health categories such as anxiety, depression, PTSD, bipolar disorder, and more. The goal is to raise awareness, encourage reflection, and provide gentle motivational support. It is not intended to replace professional care or diagnosis.

Features
Scrapes approximately 3,000 Reddit posts from mental health-related subreddits

Performs sentiment classification using a Logistic Regression model

Predicts relevant mental health tags based on user input

Offers context-aware motivational quotes for support

Includes two frontends:

A basic Flask web app (loading.py)

A fully styled Streamlit web app (app.py or similar)

Technologies and Libraries Used
Python for backend and processing

PRAW (Python Reddit API Wrapper) for Reddit scraping

NLTK for sentiment analysis (VADER)

Scikit-learn for machine learning and text vectorization

Joblib for saving/loading models

Pandas for data handling

Streamlit for user interface

Flask for alternate web deployment

Project Structure
File	Purpose
redditscrap.py-Scrapes and labels Reddit posts by subreddit
topwords.py-Extracts top words per mental health tag
mlmodel.py-Preprocesses data, trains the model, evaluates performance
loading.py-A basic Flask app for model inference
app.py (Streamlit)-Full-featured web app with UI, predictions, and quotes
disorders_dataset.csv-Final scraped dataset of ~3,000 labeled posts
mental_health_model.pkl-Trained Logistic Regression model
vectorizer.pkl-TF-IDF vectorizer used for transforming input text

Model Summary
Algorithm: Logistic Regression

Vectorization: TF-IDF

Classes: anxiety, depression, ptsd, bipolar, schizophrenia, eating_disorder

Accuracy: (refer to console output from mlmodel.py)

Input Validation: Handles profanity and minimal input gracefully

Notes
This app is for educational and awareness purposes only

The predictions are not to be taken as clinical advice

Always seek help from qualified mental health professionals for proper guidance
