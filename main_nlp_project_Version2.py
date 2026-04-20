from sentiment_analysis import SentimentAnalyzer
from named_entity_recognition import NamedEntityRecognizer
from text_summarization import TextSummarizer
from question_answering import QuestionAnsweringModel
import numpy as np

def main():
    print("=" * 60)
    print("COMPREHENSIVE NLP PROJECT WITH TENSORFLOW")
    print("=" * 60)
    
    # 1. SENTIMENT ANALYSIS
    print("\n1. SENTIMENT ANALYSIS")
    print("-" * 60)
    
    texts = [
        "I absolutely love this! Best purchase ever!",
        "This is okay, nothing special",
        "Terrible product, very disappointed",
    ]
    sentiments = [2, 1, 0]
    
    analyzer = SentimentAnalyzer()
    X_train, X_test, y_train, y_test = analyzer.load_data(texts, sentiments)
    analyzer.build_model()
    analyzer.train(X_train, y_train, X_test, y_test, epochs=3)
    
    results = analyzer.predict_sentiment(["This is wonderful!", "I hate it"])
    for r in results:
        print(f"Text: {r['text']} -> Sentiment: {r['sentiment']} ({r['confidence']:.2f})")
    
    # 2. NAMED ENTITY RECOGNITION
    print("\n2. NAMED ENTITY RECOGNITION")
    print("-" * 60)
    
    sentences = [
        "John Smith works at Apple in San Francisco",
        "Mary Johnson is CEO of Microsoft",
    ]
    tags = [
        "B-PER I-PER O O B-ORG O B-LOC I-LOC",
        "B-PER I-PER O O O B-ORG",
    ]
    
    ner = NamedEntityRecognizer()
    X, y = ner.prepare_data(sentences, tags)
    vocab_size = len(ner.word_tokenizer) + 1
    num_tags = len(ner.tag_map)
    
    ner.build_model(vocab_size, num_tags)
    ner.train(X, y, X, y, epochs=3)
    
    test_sentence = "Steve Jobs founded Apple"
    entities = ner.predict_entities(test_sentence)
    print(f"Sentence: {test_sentence}")
    for entity in entities:
        print(f"  {entity['word']}: {entity['tag']}")
    
    # 3. TEXT SUMMARIZATION
    print("\n3. TEXT SUMMARIZATION")
    print("-" * 60)
    
    texts_to_summarize = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming industries",
    ]
    summaries = [
        "fox jumps dog",
        "machine learning industries",
    ]
    
    summarizer = TextSummarizer()
    encoder_input, decoder_input = summarizer.load_data(texts_to_summarize, summaries)
    summarizer.build_model(vocab_size_encoder=5000, vocab_size_decoder=2000)
    print("Text Summarization model built successfully!")
    
    # 4. QUESTION ANSWERING
    print("\n4. QUESTION ANSWERING")
    print("-" * 60)
    
    questions = [
        "What is machine learning?",
        "Who invented the internet?",
    ]
    contexts = [
        "Machine learning is a subset of artificial intelligence",
        "The internet was invented by Tim Berners-Lee",
    ]
    answers = [
        "artificial intelligence",
        "Tim Berners-Lee",
    ]
    
    qa_model = QuestionAnsweringModel()
    question_pad, context_pad, answer_pad = qa_model.load_data(questions, contexts, answers)
    qa_model.build_model()
    qa_model.train(question_pad, context_pad, answer_pad, epochs=3)
    
    test_answer = qa_model.answer_question(
        "What is NLP?",
        "NLP is Natural Language Processing"
    )
    print(f"Q: What is NLP?")
    print(f"Context: NLP is Natural Language Processing")
    print(f"A: {test_answer}")
    
    print("\n" + "=" * 60)
    print("ALL NLP TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()