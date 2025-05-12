import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def clean(text):
    """Clean the input text by removing special characters and extra whitespace."""
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()

def remove_stopwords(text):
    """Remove English stopwords from the text."""
    stop_words = stopwords.words('english')
    return ' '.join(word for word in text.split() if word not in stop_words)