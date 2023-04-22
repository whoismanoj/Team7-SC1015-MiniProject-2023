import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


class TextProcessing:

    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.pipeline = None

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

    def predict(self, X_test):
        if self.pipeline is not None:
            return self.pipeline.predict(X_test)
        else:
            raise ValueError("Model is not trained yet. Please train the model before making predictions.")

    def evaluate(self, y_test, y_pred):
        return classification_report(y_test, y_pred)

    def process(self):
        data = self.load_data()
        data = self.preprocess_data(data)
        X, y = data['Caption'], data['LABEL']
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        self.train_model(X_train, y_train)
        y_pred = self.predict(X_test)
        report = self.evaluate(y_test, y_pred)
        return report


if __name__ == "__main__":
    text_processing = TextProcessing('./Dataset/LabeledText.xlsx')
    classification_report = text_processing.process()
    print(classification_report)
