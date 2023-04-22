import os
import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

class TextProcessing:

    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.pipeline = None
        self.lb = None

    def load_data(self):
        return pd.read_excel(self.data_file_path)

    @staticmethod
    def preprocess_text(text):
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits
        text = re.sub(r'\W|\d', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        # Join the tokens back into a string
        text = ' '.join(tokens)
        
        return text

    def split_data(self, X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000))
        ])
        self.pipeline.fit(X_train, y_train)
        self.save_model()

    def predict(self, X_test, prod=False):
        if self.pipeline is not None:
            y_pred = self.pipeline.predict(X_test)
            if prod:
                return self.label_to_onehot(y_pred)
            return y_pred
        else:
            raise ValueError("Model is not trained yet. Please train the model before making predictions.")

    def evaluate(self, y_test, y_pred):
        return classification_report(y_test, y_pred)

    def label_to_onehot(self, label, inverse=False):
        if self.lb is None:
            self.lb = LabelBinarizer()
        if inverse:
            return self.lb.inverse_transform(label)
        return self.lb.fit_transform(label)
    
    def save_model(self, model_path='text_classifier.pkl'):
        joblib.dump(self.pipeline, model_path)

    def load_model(self, model_path='text_classifier.pkl'):
        if os.path.exists(model_path):
            self.pipeline = joblib.load(model_path)
        else:
            raise FileNotFoundError("Model file not found.")
