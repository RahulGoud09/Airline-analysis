import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """Cleans and tokenizes input text for sentiment analysis."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Load dataset
try:
    df = pd.read_csv('Tweets.csv')
except FileNotFoundError:
    print("Error: Dataset 'Tweets.csv' not found. Please download it from Kaggle.")
    exit(1)

# Print value counts before processing
print("Original sentiment distribution:\n", df['airline_sentiment'].value_counts())

# Preprocess text
df['processed_text'] = df['text'].apply(preprocess_text)

# Convert sentiment to binary (positive: 1, negative: 0), removing 'neutral'
df = df[df['airline_sentiment'] != 'neutral']
df['sentiment_binary'] = (df['airline_sentiment'] == 'positive').astype(int)

# Print sentiment distribution after filtering
print("\nBinary sentiment distribution:\n", df['sentiment_binary'].value_counts())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], df['sentiment_binary'], 
    test_size=0.2, random_state=42, stratify=df['sentiment_binary']
)

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train logistic regression model
model = LogisticRegression(
    random_state=42, max_iter=1000, class_weight='balanced'
)
model.fit(X_train_vectorized, y_train)

# Save model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Evaluate model performance
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Sample predictions
sample_texts = [
    "Great flight, very comfortable and on time!",
    "Terrible service, delayed flight and rude staff.",
    "The flight was okay, nothing special.",
    "Amazing crew, loved the experience!",
    "Worst airline ever. Will never fly with them again!"
]

print("\nSample Predictions:")
sample_vectors = vectorizer.transform([preprocess_text(text) for text in sample_texts])
predictions = model.predict(sample_vectors)
probas = model.predict_proba(sample_vectors)

for text, pred, proba in zip(sample_texts, predictions, probas):
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = max(proba)  # Confidence of the predicted class
    print(f"\nText: {text}")
    print(f"Prediction: {sentiment} (Confidence: {confidence:.2f})")
