import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class SentimentAnalyzer:
    """Sentiment Analysis using TensorFlow (Positive/Negative/Neutral)"""
    
    def __init__(self, max_vocab_size=10000, max_seq_length=200, embedding_dim=128):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.model = None
        self.sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    def load_data(self, texts, sentiments):
        """Load and preprocess text data"""
        X_train, X_test, y_train, y_test = train_test_split(
            texts, sentiments, test_size=0.2, random_state=42
        )
        
        self.tokenizer = Tokenizer(num_words=self.max_vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(X_train)
        
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)
        
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_seq_length, padding='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_seq_length, padding='post')
        
        return X_train_pad, X_test_pad, y_train, y_test
    
    def build_model(self):
        """Build sentiment analysis model"""
        self.model = Sequential([
            layers.Embedding(self.max_vocab_size, self.embedding_dim, 
                           input_length=self.max_seq_length),
            layers.SpatialDropout1D(0.2),
            layers.Bidirectional(layers.LSTM(100, return_sequences=True)),
            layers.Dropout(0.2),
            layers.Bidirectional(layers.LSTM(100)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(3, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        """Train the model"""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history
    
    def predict_sentiment(self, texts):
        """Predict sentiment for new texts"""
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_seq_length, padding='post')
        predictions = self.model.predict(padded)
        sentiments = np.argmax(predictions, axis=1)
        
        results = []
        for i, sentiment_idx in enumerate(sentiments):
            results.append({
                'text': texts[i],
                'sentiment': self.sentiment_map[sentiment_idx],
                'confidence': float(predictions[i][sentiment_idx])
            })
        return results

if __name__ == "__main__":
    texts = [
        "I absolutely love this! Best purchase ever!",
        "This is okay, nothing special",
        "Terrible product, very disappointed",
        "Amazing quality and fast delivery",
        "It's fine, does what it says",
        "Worst experience of my life",
        "Fantastic! Highly recommend",
        "Not bad, could be better"
    ]
    
    sentiments = [2, 1, 0, 2, 1, 0, 2, 1]
    
    analyzer = SentimentAnalyzer()
    X_train, X_test, y_train, y_test = analyzer.load_data(texts, sentiments)
    analyzer.build_model()
    history = analyzer.train(X_train, y_train, X_test, y_test)
    
    new_texts = ["This is wonderful!", "It's average", "I hate it"]
    results = analyzer.predict_sentiment(new_texts)
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f})\n")