import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split

class QuestionAnsweringModel:
    """Question Answering using Attention Mechanism"""
    
    def __init__(self, embedding_dim=128, hidden_units=256):
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.tokenizer = None
        self.model = None
    
    def load_data(self, questions, contexts, answers):
        """Load and preprocess Q&A data"""
        all_texts = questions + contexts + answers
        
        self.tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(all_texts)
        
        question_seq = self.tokenizer.texts_to_sequences(questions)
        context_seq = self.tokenizer.texts_to_sequences(contexts)
        answer_seq = self.tokenizer.texts_to_sequences(answers)
        
        question_pad = pad_sequences(question_seq, maxlen=20, padding='post')
        context_pad = pad_sequences(context_seq, maxlen=100, padding='post')
        answer_pad = pad_sequences(answer_seq, maxlen=30, padding='post')
        
        return question_pad, context_pad, answer_pad
    
    def build_model(self, vocab_size=10000):
        """Build QA model with attention"""
        
        question_input = layers.Input(shape=(20,), name='question')
        question_embed = layers.Embedding(vocab_size, self.embedding_dim)(question_input)
        question_lstm = layers.LSTM(self.hidden_units, return_sequences=True)(question_embed)
        question_vec = layers.GlobalAveragePooling1D()(question_lstm)
        
        context_input = layers.Input(shape=(100,), name='context')
        context_embed = layers.Embedding(vocab_size, self.embedding_dim)(context_input)
        context_lstm = layers.LSTM(self.hidden_units, return_sequences=True)(context_embed)
        
        attention = layers.AdditiveAttention()([question_vec, context_lstm])
        attention_out = layers.GlobalAveragePooling1D()(attention)
        
        merged = layers.Concatenate()([question_vec, attention_out])
        dense1 = layers.Dense(128, activation='relu')(merged)
        dense1 = layers.Dropout(0.3)(dense1)
        answer_output = layers.Dense(vocab_size, activation='softmax')(dense1)
        
        self.model = keras.Model(
            inputs=[question_input, context_input],
            outputs=answer_output
        )
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, question_pad, context_pad, answer_pad, epochs=10, batch_size=32):
        """Train the QA model"""
        history = self.model.fit(
            [question_pad, context_pad],
            np.argmax(answer_pad, axis=1),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history
    
    def answer_question(self, question, context):
        """Generate answer for a question given context"""
        question_seq = self.tokenizer.texts_to_sequences([question])
        context_seq = self.tokenizer.texts_to_sequences([context])
        
        question_pad = pad_sequences(question_seq, maxlen=20, padding='post')
        context_pad = pad_sequences(context_seq, maxlen=100, padding='post')
        
        prediction = self.model.predict([question_pad, context_pad], verbose=0)
        answer_idx = np.argmax(prediction[0])
        
        answer = ""
        if answer_idx in self.tokenizer.index_word:
            answer = self.tokenizer.index_word[answer_idx]
        
        return answer

if __name__ == "__main__":
    questions = [
        "What is machine learning?",
        "Who invented the internet?",
        "Where is Paris?"
    ]
    
    contexts = [
        "Machine learning is a subset of artificial intelligence",
        "The internet was invented by Tim Berners-Lee",
        "Paris is the capital of France"
    ]
    
    answers = [
        "artificial intelligence",
        "Tim Berners-Lee",
        "capital France"
    ]
    
    qa_model = QuestionAnsweringModel()
    question_pad, context_pad, answer_pad = qa_model.load_data(questions, contexts, answers)
    qa_model.build_model()
    history = qa_model.train(question_pad, context_pad, answer_pad, epochs=5)
    
    test_question = "What is NLP?"
    test_context = "NLP is Natural Language Processing"
    answer = qa_model.answer_question(test_question, test_context)
    print(f"Question: {test_question}")
    print(f"Context: {test_context}")
    print(f"Answer: {answer}")