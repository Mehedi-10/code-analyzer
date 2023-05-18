import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Preprocess text (e.g., remove stopwords, punctuation, etc.)
    # Example using spaCy:
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    processed_text = " ".join(tokens)
    return processed_text

def preprocess_data(problem_statements, suggested_topics):
    processed_statements = [preprocess_text(statement) for statement in problem_statements]
    processed_topics = [preprocess_text(topic) for topic in suggested_topics]
    return processed_statements, processed_topics
