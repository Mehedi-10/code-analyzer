from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_model(statements, topics):
    # Vectorize input text using TF-IDF
    vectorizer = TfidfVectorizer()
    statements_vectorized = vectorizer.fit_transform(statements)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(statements_vectorized, topics, test_size=0.2, random_state=42)

    # Train a classification model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_val, y_val)
    print(f"Validation accuracy: {accuracy}")

    # Save the trained model
    save_model(model, "models/assistant_model.pkl")
    save_vectorizer(vectorizer, "models/vectorizer.pkl")

def save_model(model, path):
    # Code to save the trained model
    pass

def save_vectorizer(vectorizer, path):
    # Code to save the TF-IDF vectorizer
    pass
