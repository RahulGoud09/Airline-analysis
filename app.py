from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import pickle
import re
import os
try:
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    from nltk.tokenize import word_tokenize
    import nltk
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please make sure all required packages are installed using:")
    print("pip install nltk scikit-learn pandas flask flask-bootstrap")
    exit(1)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
Bootstrap(app)  # Initialize Flask-Bootstrap

# Use environment variable for secret key
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key')

# Load the model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def preprocess_text(text):
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        airline = request.form['airline']
        review = request.form['review']
        processed_text = preprocess_text(review)
        text_vector = vectorizer.transform([processed_text])
        prediction_proba = model.predict_proba(text_vector)[0]
        
        positive_confidence = prediction_proba[1]
        negative_confidence = prediction_proba[0]
        sentiment = 'Positive' if positive_confidence > negative_confidence else 'Negative'
        confidence = max(positive_confidence, negative_confidence) * 100
        
        return render_template('predict.html', 
                             airline=airline,
                             review=review, 
                             review_text=review,
                             sentiment=sentiment,
                             confidence_score=confidence,
                             positive_conf=positive_confidence * 100,
                             negative_conf=negative_confidence * 100,
                             sentiment_class='positive' if sentiment == 'Positive' else 'negative')

if __name__ == '__main__':
    # Use environment variables for host and port
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
