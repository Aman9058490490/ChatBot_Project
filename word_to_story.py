import os
import sqlite3
import json
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Paths to files
json_path = 'D:\\Machine Learning\\Project_Details\\Chatbot\\data.json'
db_path = 'D:\\Machine Learning\\Project_Details\\Chatbot\\stories_new.db'

# Stopwords for filtering entities
stop_words = set(stopwords.words('english'))
recent_story_id = None

def create_db(filepath):
    """
    Create the SQLite database and populate it with stories and their entity mappings.
    """
    with open(filepath, 'r') as file:
        data = json.load(file)
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables if they don't already exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS stories_table (
                        id TEXT PRIMARY KEY,
                        story TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS entity_story_mapping (
                        entity TEXT,
                        story_id TEXT,
                        PRIMARY KEY (entity, story_id))''')

    # Insert stories into the database
    for i in data['data']:
        try:
            story_id = i['id']
            story_text = i['story']

            # Insert story
            cursor.execute('''
            INSERT OR IGNORE INTO stories_table (id, story) 
            VALUES (?, ?)''', (story_id, story_text))

            # Extract entities from the story
            entities = extract_entities(story_text)

            # Insert entity-story mappings
            for entity in entities:
                cursor.execute('''
                INSERT OR IGNORE INTO entity_story_mapping (entity, story_id) 
                VALUES (?, ?)''', (entity.lower(), story_id))

        except sqlite3.Error as e:
            print(f"Error inserting data for story ID {story_id}: {e}")
            continue

    conn.commit()
    conn.close()

def extract_entities(text):
    """
    Extract entities from text using basic noun identification.
    """
    words = word_tokenize(text)
    words = [word.lower() for word in words if (word.isalnum()) and (word not in stop_words)]
    pos_tags = nltk.pos_tag(words)
    # Extract nouns (basic entities)
    entities = [word for word, pos in pos_tags if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
    return set(entities)  # Return unique entities

def retrieves_stories(question):
    """
    Retrieve the most relevant story based on the input question using cosine similarity.
    """
    # Extract entities from the question
    entities = extract_entities(question)
    if not entities:
        print("No entities found in the question.")
        return None
    else:
        print("Entities are:", entities)

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query stories containing the entities
    story_scores = {}
    relevant_story_ids = set()

    # Collect story IDs that are related to the entities
    for entity in entities:
        cursor.execute("SELECT story_id FROM entity_story_mapping WHERE entity = ?", (entity.lower(),))
        results = cursor.fetchall()
        for result in results:
            story_id = result[0]
            relevant_story_ids.add(story_id)

    if not relevant_story_ids:
        print("No relevant stories found.")
        conn.close()
        return None
    
    print("Relevant Story IDs:", relevant_story_ids)

    # Get all relevant stories from the database
    stories = {}
    for story_id in relevant_story_ids:
        cursor.execute("SELECT story FROM stories_table WHERE id = ?", (story_id,))
        result = cursor.fetchone()
        if result:
            stories[story_id] = result[0]

    conn.close()

    # If no stories found, return None
    if not stories:
        print("No stories found for the relevant IDs.")
        return None

    # Combine the question and stories into a single list for consistent vocabulary
    all_texts = [question] + list(stories.values())

    # Create a TF-IDF Vectorizer and fit it on all texts (question + stories)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Separate the question vector from the story vectors
    question_tfidf = tfidf_matrix[0:1]  # The first vector is the question
    story_tfidf = tfidf_matrix[1:]      # The rest are the stories

    # Compute cosine similarity between the question and each story
    cosine_similarities = cosine_similarity(question_tfidf, story_tfidf)

    # Find the story with the highest cosine similarity score
    most_relevant_story_index = cosine_similarities.argmax()
    most_relevant_story_id = list(stories.keys())[most_relevant_story_index]

    print(f"Most relevant story ID: {most_relevant_story_id}")

    global recent_story_id
    recent_story_id = most_relevant_story_id
    return most_relevant_story_id

def retrieve_story_sentences(story_id):
    """
    Retrieve sentences of a story for the given story ID.
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query the story using the story ID
    cursor.execute("SELECT story FROM stories_table WHERE id = ?", (story_id,))
    result = cursor.fetchone()
    
    if result:
        story_text = result[0]  # Extract the story text
        # Tokenize the story into sentences
        sentences = sent_tokenize(story_text)
        conn.close()
        return sentences
    else:
        conn.close()
        print(f"No story found for story ID: {story_id}")
        return []

# Example usage
if __name__ == "__main__":
    # Step 1: Populate the database
    create_db(json_path)

    # Step 2: Retrieve the most relevant story for a given question
    question = "who is the best person?"
    relevant_story_id = retrieves_stories(question)
    if relevant_story_id:
        print(f"The story ID for the most relevant story is: {relevant_story_id}")
        sentences = retrieve_story_sentences(relevant_story_id)
        print("Sentences in the relevant story:")
        for sentence in sentences:
            print(sentence)
            