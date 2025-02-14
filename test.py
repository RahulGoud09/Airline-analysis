import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

# Load the model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Test cases
test_reviews = [
    "The flight was amazing and the staff was very helpful",
    "Terrible service, delayed flight and rude staff",
    "Average experience, nothing special",
]

# Make predictions
for review in test_reviews:
    processed_text = preprocess_text(review)
    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)[0]
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    print(f"\nReview: {review}")
    print(f"Sentiment: {sentiment}") 