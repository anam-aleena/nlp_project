import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split

class TextSummarizer:
    """Abstractive Text Summarization using Seq2Seq"""
    
    def __init__(self, embedding_dim=128, latent_dim=256):
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.encoder_tokenizer = None
        self.decoder_tokenizer = None
        self.encoder_model = None
        self.decoder_model = None
        self.seq2seq_model = None
    
    def load_data(self, texts, summaries, max_encoder_len=100, max_decoder_len=30):
        """Load and preprocess text and summary data"""
        self.encoder_tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
        self.encoder_tokenizer.fit_on_texts(texts)
        encoder_input_data = self.encoder_tokenizer.texts_to_sequences(texts)
        encoder_input_data = pad_sequences(encoder_input_data, maxlen=max_encoder_len, padding='post')
        
        self.decoder_tokenizer = Tokenizer(num_words=2000, oov_token="<OOV>")
        self.decoder_tokenizer.fit_on_texts(summaries)
        decoder_input_data = self.decoder_tokenizer.texts_to_sequences(summaries)
        decoder_input_data = pad_sequences(decoder_input_data, maxlen=max_decoder_len, padding='post')
        
        return encoder_input_data, decoder_input_data
    
    def build_model(self, vocab_size_encoder, vocab_size_decoder, 
                   max_encoder_len=100, max_decoder_len=30):
        """Build Seq2Seq model for summarization"""
        
        encoder_inputs = layers.Input(shape=(max_encoder_len,))
        encoder_embedding = layers.Embedding(vocab_size_encoder, self.embedding_dim)(encoder_inputs)
        encoder_lstm = layers.LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]
        
        decoder_inputs = layers.Input(shape=(max_decoder_len,))
        decoder_embedding = layers.Embedding(vocab_size_decoder, self.embedding_dim)(decoder_inputs)
        decoder_lstm = layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = layers.Dense(vocab_size_decoder, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        self.seq2seq_model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.seq2seq_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        self.encoder_model = keras.Model(encoder_inputs, encoder_states)
        
        decoder_state_input_h = layers.Input(shape=(self.latent_dim,))
        decoder_state_input_c = layers.Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_states = [state_h, state_c]
        
        self.decoder_model = keras.Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )
        
        return self.seq2seq_model
    
    def train(self, encoder_input_data, decoder_input_data, epochs=10, batch_size=32):
        """Train the summarization model"""
        history = self.seq2seq_model.fit(
            [encoder_input_data, decoder_input_data],
            np.expand_dims(decoder_input_data, -1),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history
    
    def summarize(self, text, max_decoder_len=30):
        """Generate summary for input text"""
        encoder_seq = self.encoder_tokenizer.texts_to_sequences([text])
        encoder_seq = pad_sequences(encoder_seq, maxlen=100, padding='post')
        
        states_value = self.encoder_model.predict(encoder_seq, verbose=0)
        
        target_seq = np.zeros((1, max_decoder_len))
        target_seq[0, 0] = 1
        
        decoded_sentence = ""
        for i in range(1, max_decoder_len):
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq, states_value[0], states_value[1]], verbose=0
            )
            
            sampled_token_index = np.argmax(output_tokens[0, i, :])
            decoded_sentence += self.decoder_tokenizer.index_word.get(sampled_token_index, "") + " "
            
            states_value = [h, c]
        
        return decoded_sentence.strip()

if __name__ == "__main__":
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Natural language processing helps computers understand human language"
    ]
    
    summaries = [
        "fox jumps dog",
        "machine learning artificial intelligence",
        "NLP computers language"
    ]
    
    summarizer = TextSummarizer()
    encoder_input, decoder_input = summarizer.load_data(texts, summaries)
    
    summarizer.build_model(
        vocab_size_encoder=5000,
        vocab_size_decoder=2000
    )
    
    history = summarizer.train(encoder_input, decoder_input, epochs=5)