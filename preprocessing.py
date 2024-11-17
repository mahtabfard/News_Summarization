import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Clean and preprocess text by removing special characters and stopwords."""
    text = re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z]', ' ', text.lower()))
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

def tokenize_sentences(article):
    """Tokenizes an article into sentences."""
    return nltk.sent_tokenize(article)