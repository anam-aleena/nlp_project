import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re

class NamedEntityRecognizer:
    """Named Entity Recognition using BiLSTM-CRF"""
    
    def __init__(self, embedding_dim=100, lstm_units=128):
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.word_tokenizer = None
        self.tag_encoder = None
        self.model = None
        self.tag_map = {
            0: 'O',
            1: 'B-PER',
            2: 'I-PER',
            3: 'B-ORG',
            4: 'I-ORG',
            5: 'B-LOC',
            6: 'I-LOC'
        }
    
    def tokenize_words(self, sentences):
        """Tokenize sentences into words"""
        word_tokenizer = {}
        word_count = 1
        
        for sentence in sentences:
            words = sentence.split()
            for word in words:
                if word not in word_tokenizer:
                    word_tokenizer[word] = word_count
                    word_count += 1
        
        self.word_tokenizer = word_tokenizer
        return word_tokenizer
    
    def prepare_data(self, sentences, tags, max_len=50):
        """Prepare sequences for NER"""
        self.tokenize_words(sentences)
        
        X = []
        y = []
        
        for sentence, tag_seq in zip(sentences, tags):
            words = sentence.split()
            tag_list = tag_seq.split()
            
            word_indices = [self.word_tokenizer.get(word, 0) for word in words]
            
            if len(word_indices) < max_len:
                word_indices.extend([0] * (max_len - len(word_indices)))
            else:
                word_indices = word_indices[:max_len]
            
            tag_indices = [list(self.tag_map.values()).index(tag) if tag in self.tag_map.values() else 0 
                          for tag in tag_list]
            
            if len(tag_indices) < max_len:
                tag_indices.extend([0] * (max_len - len(tag_indices)))
            else:
                tag_indices = tag_indices[:max_len]
            
            X.append(word_indices)
            y.append(tag_indices)
        
        return np.array(X), np.array(y)
    
    def build_model(self, vocab_size, num_tags, max_len=50):
        """Build BiLSTM model for NER"""
        model = keras.Sequential([
            layers.Embedding(vocab_size, self.embedding_dim, input_length=max_len, mask_zero=True),
            layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True, dropout=0.2)),
            layers.Dropout(0.2),
            layers.Bidirectional(layers.LSTM(self.lstm_units // 2, return_sequences=True, dropout=0.2)),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_tags, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        """Train the NER model"""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history
    
    def predict_entities(self, sentence):
        """Predict entities in a sentence"""
        words = sentence.split()
        word_indices = np.array([[self.word_tokenizer.get(word, 0) for word in words]])
        
        predictions = self.model.predict(word_indices, verbose=0)
        predicted_tags = np.argmax(predictions[0], axis=1)
        
        entities = []
        for word, tag_idx in zip(words, predicted_tags):
            tag = list(self.tag_map.values())[tag_idx] if tag_idx in self.tag_map else 'O'
            entities.append({'word': word, 'tag': tag})
        
        return entities

if __name__ == "__main__":
    sentences = [
        "John Smith works at Apple in San Francisco",
        "Mary Johnson is CEO of Microsoft",
        "Google is located in Mountain View"
    ]
    
    tags = [
        "B-PER I-PER O O B-ORG O B-LOC I-LOC",
        "B-PER I-PER O O O B-ORG",
        "B-ORG O O O B-LOC I-LOC"
    ]
    
    ner = NamedEntityRecognizer()
    X, y = ner.prepare_data(sentences, tags)
    
    vocab_size = len(ner.word_tokenizer) + 1
    num_tags = len(ner.tag_map)
    
    ner.build_model(vocab_size, num_tags)
    history = ner.train(X, y, X, y, epochs=5)
    
    test_sentence = "Steve Jobs founded Apple"
    entities = ner.predict_entities(test_sentence)
    print(f"Sentence: {test_sentence}")
    for entity in entities:
        print(f"  {entity['word']}: {entity['tag']}")