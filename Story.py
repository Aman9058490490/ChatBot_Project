import os
import sqlite3
import json
import nltk
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Paths to files
json_path = 'D:\\Machine Learning\\Project_Details\\Chatbot\\data.json'
db_path = 'D:\\Machine Learning\\Project_Details\\Chatbot\\stories_new.db'

def create_db(filepath):
    """
    Create the SQLite database and populate it with stories from the JSON file.
    """
    with open(filepath, 'r') as file:
        data = json.load(file)
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create a table if it doesn't already exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS stories_table (
        id TEXT PRIMARY KEY,
        story TEXT
    )
    ''')

    # Insert stories into the database
    for i in data['data']:
        try:
            cursor.execute('''
            INSERT OR IGNORE INTO stories_table (id, story) 
            VALUES (?, ?)
            ''', (i['id'], i['story']))
        except sqlite3.Error as e:
            print(f"Error inserting data for story ID {i['id']}: {e}")
            continue

    conn.commit()
    conn.close()

    # Debugging: Verify rows in the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM stories_table")
    row_count = cursor.fetchone()[0]
    print(f"Number of rows in the database: {row_count}")
    conn.close()

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

    # Step 2: Example story IDs to retrieve
    story_ids = ["3dr23u6we5exclen4th8uq9rb42tel", "3azhrg4cu4ktme1zh7c2ro3pn2430d"]  # Replace with actual IDs from your JSON file

    # Step 3: Retrieve sentences for each story and store them in separate lists
    all_sentences = []  # Master list to store sentences for all stories
    for story_id in story_ids:
        print(f"\nProcessing story ID: {story_id}")
        sentences = retrieve_story_sentences(story_id)
        print(sentences)
        if sentences:
            all_sentences.append(sentences)  # Append the sentences to the master list
        else:
            print(f"No sentences to display for story ID {story_id}.")
            all_sentences.append([])  # Append an empty list if no sentences are found

    # Debugging: Print the master list of sentences
    print("\nAll sentences grouped by story ID:")
    for idx, sentences in enumerate(all_sentences, start=1):
        print(f"Story {idx} sentences:")
        print(sentences)

def retrieves_stories():
    pass
