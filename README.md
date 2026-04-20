# Comprehensive NLP Project with TensorFlow

This project contains implementations of various Natural Language Processing (NLP) tasks using TensorFlow and deep learning techniques.

## Features

### 1. **Sentiment Analysis**
- 3-class classification (Negative, Neutral, Positive)
- BiLSTM architecture with embeddings
- Real-time sentiment prediction with confidence scores

### 2. **Named Entity Recognition (NER)**
- Identifies Person, Organization, Location entities
- BiLSTM model with dropout regularization
- Support for B-I-O (Beginning-Inside-Outside) tagging scheme

### 3. **Text Summarization**
- Seq2Seq with attention mechanism
- Abstractive summarization approach
- Encoder-Decoder architecture

### 4. **Question Answering**
- Context-aware QA system
- Attention mechanism for relevance
- Answer generation from context

## Project Structure

```
nlp_project/
├── sentiment_analysis.py       # Sentiment classifier
├── named_entity_recognition.py # NER model
├── text_summarization.py       # Summarizer
├── question_answering.py       # QA system
├── main_nlp_project.py         # Main runner
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/anam-aleena/nlp_project.git
cd nlp_project
```

2. **Create virtual environment (optional but recommended)**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Run all NLP tasks
```bash
python main_nlp_project.py
```

### Run individual modules

**Sentiment Analysis:**
```bash
python sentiment_analysis.py
```

**Named Entity Recognition:**
```bash
python named_entity_recognition.py
```

**Text Summarization:**
```bash
python text_summarization.py
```

**Question Answering:**
```bash
python question_answering.py
```

## Code Examples

### Sentiment Analysis
```python
from sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
X_train, X_test, y_train, y_test = analyzer.load_data(texts, sentiments)
analyzer.build_model()
analyzer.train(X_train, y_train, X_test, y_test)

results = analyzer.predict_sentiment(["This is wonderful!"])
```

### Named Entity Recognition
```python
from named_entity_recognition import NamedEntityRecognizer

ner = NamedEntityRecognizer()
entities = ner.predict_entities("John Smith works at Apple")
```

### Text Summarization
```python
from text_summarization import TextSummarizer

summarizer = TextSummarizer()
summary = summarizer.summarize("Your long text here...")
```

### Question Answering
```python
from question_answering import QuestionAnsweringModel

qa = QuestionAnsweringModel()
answer = qa.answer_question("What is AI?", "AI is Artificial Intelligence")
```

## Model Architecture

### Sentiment Analysis
- Embedding Layer → Bidirectional LSTM → LSTM → Dense (64) → Softmax

### NER
- Embedding → Bidirectional LSTM → Bidirectional LSTM → Dense (64) → Softmax

### Text Summarization
- Encoder: LSTM with attention
- Decoder: LSTM for sequence generation

### Question Answering
- Question Encoder → LSTM
- Context Encoder → LSTM
- Attention mechanism → Dense layers → Output

## Hyperparameters

Default settings (can be customized):
- `max_vocab_size`: 10,000
- `max_seq_length`: 200
- `embedding_dim`: 128
- `batch_size`: 32
- `epochs`: 10
- `lstm_units`: 128/256

## Performance Metrics

Each model tracks:
- **Accuracy**: Percentage of correct predictions
- **Loss**: Cross-entropy loss
- **Validation Metrics**: Epoch-wise performance

## Future Enhancements

- [ ] Add BERT pre-trained models
- [ ] Implement transformer architecture
- [ ] Support for multiple languages
- [ ] Production deployment ready code
- [ ] Advanced evaluation metrics (F1, precision, recall)
- [ ] Model persistence (save/load)
- [ ] Real-world datasets integration

## Dependencies

- TensorFlow 2.13.0
- NumPy 1.24.3
- Pandas 2.0.3
- Scikit-learn 1.3.0
- Matplotlib 3.7.2
- NLTK 3.8.1

## Author
Created by: anam-aleenaml engineer

## License
This project is open source and available under the MIT License.

## Contributing
Feel free to submit issues and pull requests to improve the project!
