# Text Document Classification using NLP, RNN and LSTM 🤖

A machine learning project that classifies text documents into categories like politics, tech, business, or sports using Natural Language Processing (NLP) and Deep Learning approaches.

## Overview
This project implements multiple text classification models, comparing traditional NLP approaches with deep learning methods to achieve optimal classification results.

## Requirements
- Python 3.7+
- scikit-learn
- tensorflow
- numpy
- pandas
- nltk
- streamlit

## Dataset 📊

The dataset contains a csv file that has the text string and the label for it

## Text Preprocessing
The following steps are applied to clean and prepare the text data:
1. Text Cleaning
   - Remove special characters
   - Convert to lowercase
2. Remove Stopwords
3. Tokenize the Text
4. Apply Stemming and Lemmatization

## Models Implemented

### Traditional NLP Approaches
I utilized several models to evaluate the performance of each model and take the best one:
1. *Count Vectorizer + Naive Bayes*
2. *Count Vectorizer with N-Grams + Naive Bayes*
3. *tfidf vectorizer + Naive Bayes*

### LSTM Model
The deep learning approach achieved 92% accuracy on the test set with the following architecture:
- Embedding layer
- 2x LSTM layers
- 2x Dense layers
- Adam optimizer (lr=0.001)
- Sparse categorical cross-entropy loss

Model training details:
- Epochs: 25
- Batch size: 20
- Validation split: 0.2

## Deployment 🛠️
The model is deployed using Streamlit, providing an interactive web interface where users can:
1. Input any text
2. Get instant classification results
3. View confidence scores for each category

## Usage 🚀

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
streamlit run app.py
```

### Code Example
```python
# Load and use the model
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    # Remove special characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop_words])

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Predict
text = "South Africa's Schalk Burger was named player of the year"
processed_text = preprocess_text(text)
prediction = model.predict([processed_text])[0]
print(f"Predicted Category: {prediction}")
```

## Contact 📩
- **Author**: Saher Mohammed
- **Email**: sahermuhamed176@gmail.com
- **LinkedIn**: [Saher's LinkedIn](https://www.linkedin.com/in/sahermuhamed/)

## License
This project is licensed under the MIT License - see the LICENSE file for details.


