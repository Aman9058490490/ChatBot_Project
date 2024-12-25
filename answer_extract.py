import sqlite3
import json
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import defaultdict

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Connect to SQLite Database (or replace with your own database connection)
conn = sqlite3.connect("stories_new.db")  # Database name 'stories_new'
cursor = conn.cursor()

# Preprocessing function for text
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize
    tokens = [t for t in tokens if t not in stopwords.words('english')]  # Remove stopwords
    return " ".join(tokens)  # Join tokens back into a string

# Function to fetch stories and their relevant data from the database
def fetch_stories_from_db():
    cursor.execute("SELECT id, story FROM stories_new")  # Assuming 'id' and 'story' are columns
    stories_data = cursor.fetchall()
    
    return stories_data

# Function to fetch questions from the database (for training the model)
def fetch_questions_from_db():
    cursor.execute("SELECT question, story_id FROM questions_new")  # Assuming 'question' and 'story_id' are columns
    questions_data = cursor.fetchall()
    
    return questions_data

# Function to fetch the most relevant sentences for a story ID
def fetch_sentences_from_story(story_id):
    cursor.execute("SELECT story FROM stories_new WHERE id=?", (story_id,))
    story_data = cursor.fetchone()
    
    if story_data:
        story_text = story_data[0]
        sentences = sent_tokenize(story_text)
        return sentences
    return []

# Prepare questions and corresponding story labels from the database
questions = []
story_labels = []
story_texts = []
story_sentences = []

# Fetch questions and story data from the database
questions_data = fetch_questions_from_db()
stories_data = fetch_stories_from_db()

for question_data in questions_data:
    question = question_data[0]
    story_id = question_data[1]
    
    # Fetch the story text using the story_id
    story_text = next((story[1] for story in stories_data if story[0] == story_id), None)
    
    if story_text:
        # Add question-answer pairs
        questions.append(question)
        story_labels.append(story_id)  # Label with the story ID
        story_texts.append(story_text)  # Text of the story to be used as context for the question
        story_sentences.append(sent_tokenize(story_text))  # Save the sentences for later use

# Create a Naive Bayes classifier pipeline
vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
classifier = MultinomialNB()

# Create a pipeline that first vectorizes the questions, then classifies the story ID
model = make_pipeline(vectorizer, classifier)

# Train the model on the question-answer pairs (story IDs as labels)
model.fit(questions, story_labels)

# Function to predict the most relevant story ID for a new question
def predict_relevant_story(question):
    predicted_story_id = model.predict([question])[0]
    return predicted_story_id

# Function to train a Naive Bayes classifier on extracted sentences for answer prediction
def train_answer_classifier():
    answer_questions = []
    answer_labels = []
    
    # Generate question-answer pairs from sentences (for simplicity, using story ID as labels)
    for story_data in stories_data:
        story_id = story_data[0]
        sentences = sent_tokenize(story_data[1])  # Tokenize sentences from the story
        
        for question in [
            "What happened in the story?", "Who is the main character?", "What is the key event?"
        ]:
            for sentence in sentences:
                answer_questions.append(question)
                answer_labels.append(sentence)  # The sentence is the answer for the question
    
    # Create and train a Naive Bayes classifier for answering the questions
    answer_model = make_pipeline(TfidfVectorizer(preprocessor=preprocess_text), MultinomialNB())
    answer_model.fit(answer_questions, answer_labels)
    
    return answer_model

# Function to predict an answer to a question using the trained classifier
def predict_answer(question, answer_model):
    predicted_answer = answer_model.predict([question])[0]
    return predicted_answer

# Function to evaluate model accuracy using confusion matrix
def evaluate_model(answer_model):
    # Test questions and expected answers (for simplicity, using hardcoded examples)
    test_questions = [
        "What happened in the story?", "Who is the main character?", "What is the key event?"
    ]
    
    # Expected answers (this should ideally come from your dataset or user annotations)
    expected_answers = [
        "The main character is Cotton.",  # This is a dummy answer
        "The main character is Cotton.",
        "Cotton's actions changed the course of the event."
    ]
    
    predicted_answers = [predict_answer(q, answer_model) for q in test_questions]
    
    # Evaluate using confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(expected_answers, predicted_answers))
    
    print("\nClassification Report:")
    print(classification_report(expected_answers, predicted_answers))

# Example: Train the answer classifier
answer_model = train_answer_classifier()

# Function to handle user input for questions
def handle_user_questions():
    while True:
        user_question = input("Please enter your question (or type 'exit' to quit): ")
        
        # If user types 'exit', stop the loop
        if user_question.lower() == 'exit':
            print("Exiting...")
            break
        
        # Predict the most relevant story ID based on the question
        predicted_story_id = predict_relevant_story(user_question)
        
        # Extract the most relevant sentences from the selected story
        relevant_sentences = fetch_sentences_from_story(predicted_story_id)
        
        # Print relevant sentences
        print(f"\nRelevant Sentences from Story ID {predicted_story_id}:")
        for sentence in relevant_sentences:
            print(sentence)
        
        # Get an answer prediction from the answer model
        predicted_answer = predict_answer(user_question, answer_model)
        print(f"\nPredicted Answer: {predicted_answer}")
        
        # Optionally evaluate model accuracy after each question (for demo purposes)
        evaluate_model(answer_model)

# Start the question-answering process
handle_user_questions()
