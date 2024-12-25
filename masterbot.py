import re
import os
import re
import uuid
import json
import string
import sqlite3
import nltk
from nltk.chat.util import Chat, reflections
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag, ne_chunk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from collections import defaultdict

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')


###############################################################
# paths



# Paths to files
json_path = '/content/drive/MyDrive/Datasets/Chatbot/data.json'
db_path = '/content/drive/MyDrive/Datasets/Chatbot/chatbot.db'

# Stopwords for filtering entities
stop_words = set(stopwords.words('english'))
word_to_story = {}

recent_story_id = None




####################################################################
# helper functions

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return tokens

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

def extract_named_entities_with_proper_nouns(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    entities = set()
    for word, tag in pos_tags:
        if tag in ['NNP', 'NNPS']:
            entities.add(word.lower())
    return entities

def create_db(filepath):
    """
    Create the SQLite database and populate it with stories and their entity mappings.
    """
    
    db_exists = os.path.exists(db_path) and os.path.getsize(db_path) > 0
    if db_exists:
        print("Database already exists. Skipping creation.")
        print("Building word to story mapping")
        return


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
    entities = [word for word, pos in pos_tags if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
    return set(entities)


def retrieves_stories(question,db_path):
    """
    Retrieve the most relevant story based on the input question using cosine similarity.
    """
    # Extract entities from the question
    entities = extract_entities(question)
    if not entities:
        # print("No entities found in the question.")
        return None
    # else:
        # print("Entities are:", entities)

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
        # print("No relevant stories found.")
        conn.close()
        return None
    
    # print("Relevant Story IDs:", relevant_story_ids)

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
        # print("No stories found for the relevant IDs.")
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

    # print(f"Most relevant story ID: {most_relevant_story_id}")

    global recent_story_id
    recent_story_id = most_relevant_story_id
    return most_relevant_story_id

def retrieve_story_sentences(story_id,conn):
    """
    Retrieve sentences of a story for the given story ID.
    """
    # Connect to the database
    cursor = conn.cursor()

    # Query the story using the story ID
    cursor.execute("SELECT story FROM stories_table WHERE id = ?", (story_id,))
    result = cursor.fetchone()
    
    if result:
        story_text = result[0]  # Extract the story text
        # Tokenize the story into sentences
        sentences = sent_tokenize(story_text)
        # conn.close()
        return sentences
    else:
        # conn.close()
        print(f"No story found for story ID: {story_id}")
        return []


def load_stories_from_db(conn):

    cursor = conn.cursor()
    word_story_dict = defaultdict(list)
    entity_story_mapping = defaultdict(set)
    cursor.execute("SELECT id, story FROM stories_table")
    rows = cursor.fetchall()

    for story_id, story_text in rows:
        # Preprocess story text
        words = preprocess_text(story_text)
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1

        for word, freq in word_freq.items():
            word_story_dict[word].append((story_id, freq))
            synonyms = get_synonyms(word)
            for synonym in synonyms:
                word_story_dict[synonym].append((story_id, freq))

        entities = extract_named_entities_with_proper_nouns(story_text)
        for entity in entities:
            entity_story_mapping[entity].add(story_id)

    return word_story_dict, entity_story_mapping


def add_story_to_db(story_text, word_story_dict, entity_story_mapping,conn):
    story_id = str(uuid.uuid4())
    cursor = conn.cursor()
    cursor.execute("INSERT INTO stories_table (id, story) VALUES (?, ?)", (story_id, story_text))
    conn.commit()
    
    # Preprocess the story and update word_story_dict
    words = preprocess_text(story_text)
    word_freq = defaultdict(int)
    for word in words:
        word_freq[word] += 1

    for word, freq in word_freq.items():
        word_story_dict[word].append((story_id, freq))
        synonyms = get_synonyms(word)
        for synonym in synonyms:
            word_story_dict[synonym].append((story_id, freq))

    # Extract named entities and update entity_story_mapping
    entities = extract_named_entities_with_proper_nouns(story_text)
    for entity in entities:
        entity_story_mapping[entity].add(story_id)
    
    return story_id

# def update_question_in_db(question, story_id,conn):
#     cursor = conn.cursor()
#     cursor.execute("INSERT INTO questions (question, story_id) VALUES (?, ?)", (question, story_id))
#     conn.commit()


def rank_stories_by_question(question, word_story_dict, previous_story_id=None):
    question_words = preprocess_text(question)  # Preprocess the question
    story_scores = defaultdict(int)

    # For each word in the question, check its frequency in stories
    for word in question_words:
        if word in word_story_dict:
            for story_id, freq in word_story_dict[word]:
                # Increase the score of the previous story if it appears again
                if story_id == previous_story_id:
                    story_scores[story_id] += freq * 1.5  # Boost by 1.5 times
                else:
                    story_scores[story_id] += freq  # Normal score increment

    # Sort the stories by total word frequency in descending order
    ranked_stories = sorted(story_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_stories

def any_word_or_synonym_in_current_story(unique_words, current_story_id, word_story_dict):
    for word in unique_words:
        # Check the word itself
        if word in word_story_dict:
            stories_with_word = [story_id for story_id, _ in word_story_dict[word]]
            if current_story_id in stories_with_word:
                return True  # Word found in the current story

        # Check synonyms
        synonyms = get_synonyms(word)
        for synonym in synonyms:
            if synonym in word_story_dict:
                stories_with_synonym = [story_id for story_id, _ in word_story_dict[synonym]]
                if current_story_id in stories_with_synonym:
                    return True  # Synonym found in the current story
    return False


def handle_questions(questions, word_story_dict, entity_story_mapping):
    previous_story_id = None
    resolved_story_ids = []  # To store resolved story IDs for each question

    for question in questions:
        question_words = preprocess_text(question)
        pronouns = ['he', 'she', 'it', 'they', 'them', 'his', 'her']
        contains_pronoun = any(word in pronouns for word in question_words)
        unique_words = [word for word in question_words if word not in pronouns and word not in stopwords.words('english')]
        entities_in_question = extract_named_entities_with_proper_nouns(question)

        # Multi-Entity Resolution
        if len(entities_in_question) > 1:
            possible_story_ids = None
            for entity in entities_in_question:
                if entity in entity_story_mapping:
                    entity_story_ids = entity_story_mapping[entity]
                    if possible_story_ids is None:
                        possible_story_ids = entity_story_ids
                    else:
                        possible_story_ids = possible_story_ids.intersection(entity_story_ids)
            if possible_story_ids:
                previous_story_id = list(possible_story_ids)[0]
                # update_question_in_db(question, previous_story_id)
                resolved_story_ids.append(previous_story_id)  # Append resolved story ID
                continue

        # Single Entity Resolution
        if entities_in_question:
            max_entity_story = None
            max_entity_frequency = 0

            for entity in entities_in_question:
                if entity in entity_story_mapping:
                    for story_id in entity_story_mapping[entity]:
                        entity_frequency = sum(freq for s_id, freq in word_story_dict.get(entity, []) if s_id == story_id)
                        if entity_frequency > max_entity_frequency:
                            max_entity_frequency = entity_frequency
                            max_entity_story = story_id

            if max_entity_story:
                previous_story_id = max_entity_story
                # update_question_in_db(question, previous_story_id)
                resolved_story_ids.append(previous_story_id)  # Append resolved story ID
                continue

        # Pronoun Resolution with Context Retention
        if contains_pronoun and previous_story_id:
            # update_question_in_db(question, previous_story_id)
            resolved_story_ids.append(previous_story_id)  # Append resolved story ID
            continue

        # Unique Word and Synonym Check
        if unique_words and previous_story_id:
            if any_word_or_synonym_in_current_story(unique_words, previous_story_id, word_story_dict):
                # update_question_in_db(question, previous_story_id)
                resolved_story_ids.append(previous_story_id)  # Append resolved story ID
                continue
            else:
                ranked_stories = rank_stories_by_question(question, word_story_dict, previous_story_id)
                if ranked_stories:
                    previous_story_id = ranked_stories[0][0]
                # update_question_in_db(question, previous_story_id)
                resolved_story_ids.append(previous_story_id)  # Append resolved story ID
                continue

        # Default
        ranked_stories = rank_stories_by_question(question, word_story_dict, previous_story_id)
        if ranked_stories:
            previous_story_id = ranked_stories[0][0]
        # update_question_in_db(question, previous_story_id)
        resolved_story_ids.append(previous_story_id)  # Append resolved story ID

    return resolved_story_ids  # Return the list of resolved IDs  



#########################################################################
### feature extraction


# List of auxiliary verbs
AUXILIARY_VERBS = {
    "is", "am", "are", "was", "were", "be", "being", "been",
    "have", "has", "had", "do", "does", "did",
    "shall", "will", "should", "would", "can", "could", "may", "might", "must",
    "ought", "need", "dare"
}

def extract_sentence_components(sentence):
    """
    Extract components of a sentence: subject, verbs (excluding auxiliary verbs),
    adjectives, objects, and nouns. Returns a dictionary with the extracted components.
    """
    # Tokenize and POS tag the sentence
    tokens = word_tokenize(sentence.lower())
    tagged = pos_tag(tokens)

    # Initialize components
    subject = None
    verbs = []
    adverbs = []
    adjectives = []
    nouns = []
    objects = []
    aux_verbs = []
    singular_entity = False
    multiple_entity = False

    # Iterate through the tagged tokens
    for i, (word, tag) in enumerate(tagged):
        if tag in ['NN', 'NNP', 'NNS', 'NNPS', 'PRP']:
            if subject is None:
                subject = word  # First noun/pronoun is treated as the subject
            nouns.append(word)
        if (tag == 'PRP' and word in ['they', 'we', 'you']) or (tag in ['NNS', 'NNPS']):
          multiple_entity = True
        if tag == 'PRP' and word in ['he', 'she', 'it'] or (tag in ['NN', 'NNP']):
          singular_entity = True
        if tag.startswith('VB') and word.lower() not in AUXILIARY_VERBS:
            verbs.append(word)  # Exclude auxiliary verbs
        if tag.startswith('VB') and word.lower() in AUXILIARY_VERBS:
            aux_verbs.append(word)
        if tag.startswith('RB') and word.lower() in AUXILIARY_VERBS:
            aux_verbs.append(word)
        if tag.startswith('RB') and word.lower() not in AUXILIARY_VERBS:
            adverbs.append(word)  # Exclude auxiliary verbs
        if tag.startswith('JJ'):
            adjectives.append(word)  # Adjectives

    # Determine objects (heuristically as remaining nouns after the subject)
    objects = nouns[1:] if len(nouns) > 1 else []

    # Return all components as a dictionary

    # Return all components as a dictionary
    return {
        "subject": subject,
        "verbs": verbs,
        "adverbs": adverbs,
        "adjectives": adjectives,
        "objects": objects,
        "nouns": nouns,
        "singular_entity": singular_entity,
        "multiple_entity": multiple_entity
    }
    # return subject,verbs,adverbs,adjectives,objects,nouns

def has_relationship(word1, word2):
    """
    Determines if two words have a direct or indirect relationship in WordNet.

    Returns:
        bool: True if a relationship exists, False otherwise.
    """
    # Get synsets for the two words
    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)

    # If either word has no synsets in WordNet, return False
    if not synsets1 or not synsets2:
        return False

    # Define excluded general concepts to avoid overgeneralization
    EXCLUDED_CONCEPTS = {"entity", "physical_entity", "object", "whole", "matter", "substance"}

    # Check for direct relationships
    for syn1 in synsets1:
        for syn2 in synsets2:
            # Check if the words are synonyms (direct match)
            if syn1 == syn2:
                return True

            # Check if one word is a hypernym of the other
            if syn2 in syn1.hypernyms() or syn1 in syn2.hypernyms():
                return True
            if syn1 in syn2.hypernyms() or syn1 in syn2.hypernyms():
                return True


            if syn1 in syn2.closure(lambda s: s.hyponyms()):
                return True
            if syn2 in syn1.closure(lambda s: s.hyponyms()):
                return True

            # Check if one word is a part-meronym of the other (part-whole relationship)
            if syn2 in syn1.part_meronyms() or syn1 in syn2.part_meronyms():
                return True

            if syn1 in syn2.part_meronyms() or syn1 in syn2.part_meronyms():
                return True

    # Check for indirect relationships via lowest common hypernym
    for syn1 in synsets1:
        for syn2 in synsets2:
            common_hypernyms = syn1.lowest_common_hypernyms(syn2)
            if common_hypernyms:
                for hypernym in common_hypernyms:
                    # Exclude overly general concepts and ensure the concept has enough depth
                    if hypernym.name().split('.')[0] not in EXCLUDED_CONCEPTS and hypernym.min_depth() > 1:
                        return True

    # If no relationships are found, return False
    return False

temporal_regex = r"""
(?i)\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)s?\b|
\b(January|February|March|April|May|June|July|August|September|October|November|December)\b|
\b([0-9]{4})(s)?\b|
\b(lunch|bed|dinner)(time)\b|
\b(today|yesterday|tomorrow|tonight|last|next|ago|soon|early|late)\b|
\b(\d{1,2}(:\d{2})?\s*(AM|PM|am|pm|a.m.|p.m.)?)\b|
\b\d+\s*(BC|AD|B\.C\.|A\.D\.)\b|
\b(Time|Duration|Period|Interval|Span|Epoch|Era|Age|Eon|Moment|Instant|While)\b|
\b(Second|Minute|Hour|Day|Week|Month|Year|Decade|Centur|Millenni)(s|ies|um|a|y)?\b|
\b(Morning|Noon|Afternoon|Evening|Night|Midnight|Dawn|Dusk|Sunrise|Sunset)\b|
\b(Now|Then|Today|Tomorrow|Yesterday|Soon|Later|Before|After|Early|Late|Recently|Ago|Already|Still|Yet|Eventually|Frequently|Occasionally)\b|
\b(Clock|Watch|Calendar|Schedule|Timeline|Deadline|Appointment|Anniversary|Holiday|Season|Past|Present|Future|Ancient|Modern|Contemporary|Annual|Quarterly|Monthly|Weekly|Daily|Hourly)\b
\b(start|end|beginning|while|during|middle|conclusion|initiation|termination|commencement|cessation)\b|
\b(spring|summer|autumn|fall|winter)\b|
\b(during|while|throughout|since|until|till|before|after|when|whenever|once|upon|following|prior\sto|subsequent\sto|in\sthe\scourse\sof|pending|in\sthe\sevent\sof)\b|
\b(meanwhile|subsequently|thereafter|henceforth|formerly|previously|lately|recently|shortly|immediately|eventually|ultimately|at\slength)\b|
\b(in\sthe\smeantime|in\sthe\sinterim|for\sthe\stime\sbeing|from\stime\sto\stime|at\sthe\ssame\stime|at\sthis\spoint\sin\stime|in\sdue\stime|over\stime|as\stime\sgoes\sby|as\stime\spasses|from\snow\son|from\sthen\son|for\sthe\sduration\sof|in\sthe\slong\srun|in\sthe\sshort\sterm|in\sthe\snear\sfuture|in\sthe\sdistant\spast)\b
"""

location_pos_sequences = {
    ('IN', 'NNP'),        # "in Paris"
    ('IN', 'DT', 'NN'),   # "in the park"
    ('IN', 'DT', 'NNS'),  # "in the cities"
    ('IN', 'IN', 'DT', 'NN'),  # "across from the bank"
}
def has_location_context(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)

    for seq in location_pos_sequences:
        for i in range(len(tagged_tokens) - len(seq) + 1):
            if tuple(tagged_tokens[j][1] for j in range(i, i + len(seq))) == seq:
                return True  # Found a match for location-related POS sequence
    return False  # No match found


# List of auxiliary verbs
def feat_extract(question,answer):
    """
    Extract components of a sentence: subject, verbs (excluding auxiliary verbs),
    adjectives, objects, and nouns. Returns a dictionary with the extracted components.
    """

    features = {}

    # 1. Extract sentence components for question and answers
    q_components = extract_sentence_components(question.lower())
    a_components = extract_sentence_components(answer.lower())

    # Extract relevant components

    q_nouns = q_components["nouns"]
    a_nouns = a_components["nouns"]

    features["entity_match"] = bool(set(q_nouns).intersection(set(a_nouns)))

    # 5. Singular/Multiple Entity Matching
    features["has_singular_entity_and_match_found"] = q_components["singular_entity"] and a_components["singular_entity"]
    features["has_multiple_entity_and_match_found"] = q_components["multiple_entity"] and a_components["multiple_entity"]
    features["has_location_context"] = has_location_context(answer)
    matches = re.findall(temporal_regex,answer, re.VERBOSE)
    if matches:
        features["has_temporal_context"] = True
    else:
        features["has_temporal_context"] = False

    # Tokenize and POS tag the sentence
    qn = word_tokenize(question.lower())
    an = word_tokenize(answer.lower())
    qn = [word for word in qn if (word not in string.punctuation) and (word not in stop_words) and (word.isalpha() or word.isnumeric())]
    an = [word for word in an if (word not in string.punctuation) and (word not in stop_words) and (word.isalpha() or word.isnumeric())]

    # print("Q : ",question)
    # print("A : ",answer)
    # print(qn)
    # print(an)
    x = len(qn)
    common_words = set(qn).intersection(set(an))
    for word in common_words:
      qn.remove(word)
      an.remove(word)

    # print("qn : ",qn)
    qn = [word for word in qn if word not in common_words]
    an = [word for word in an if word not in common_words]

    related_words = []
    for i in qn:
      for j in an:
        if has_relationship(i,j):
          related_words.append((i,j))
          an.remove(j)
          break

    if x>0:
      # print(len(common_words),len(related_words),x)
      x = (len(common_words) + len(related_words))/x
    else:
      x = 0
    qn = [word for word in qn if word not in [i[0] for i in related_words]]
    an = [word for word in an if word not in [i[1] for i in related_words]]

    # print(x)
    # print("common words : ",common_words)
    # print("related words : ",related_words)
    y = (len(common_words)+len(related_words))
    if y:
      y = len(common_words)/y
    else:
      y = 0
    # print('y : ',y)
    features['sim_measure'] = x

    # print('xy : ',x*y)
    # print("has_multiple_entity_and found",features['has_multiple_entity_and_match_found'])
    # print("...................................")
    return features

def extract_boolean_features(question, answer):
    """
    Extracts boolean features for a question-answer pair.
    """
    features = {}

    # 1. Extract sentence components for question and answers
    q_components = extract_sentence_components(question.lower())
    a_components = extract_sentence_components(answer.lower())

    # Extract relevant components
    q_verbs = q_components["verbs"]
    q_adverbs = q_components["adverbs"]
    q_adjectives = q_components["adjectives"]
    q_nouns = q_components["nouns"]

    a_verbs = a_components["verbs"]
    a_adverbs = a_components["adverbs"]
    a_adjectives = a_components["adjectives"]
    a_nouns = a_components["nouns"]

    # 2. Check for verb and related word presence
    features["has_verb_and_related_word_found"] = False
    for q_verb in q_verbs:
        for a_verb in a_verbs:
            if has_relationship(q_verb, a_verb):
                features["has_verb_and_related_word_found"] = True
                break
        if features["has_verb_and_related_word_found"]:
            break

        for a_noun in a_nouns:
            if has_relationship(q_verb, a_noun):
                features["has_verb_and_related_word_found"] = True
                break
        if features["has_verb_and_related_word_found"]:
            break

    features["has_adverb_and_related_word_found"] = False
    for q_adverb in q_adverbs:
        for a_adverb in a_adverbs:
            if has_relationship(q_adverb, a_adverb):
                features["has_adverb_and_related_word_found"] = True
                break
        if features["has_adverb_and_related_word_found"]:
            break

        for a_noun in a_nouns:
            if has_relationship(q_adverb, a_noun):
                features["has_adverb_and_related_word_found"] = True
                break
        if features["has_adverb_and_related_word_found"]:
            break

    # 3. Check for adjective and related word presence
    features["has_adjective_and_related_word_found"] = False
    for q_adj in q_adjectives:
        for a_adj in a_adjectives:
            if has_relationship(q_adj, a_adj):
                features["has_adjective_and_related_word_found"] = True
                break
        if features["has_adjective_and_related_word_found"]:
            break

        for a_noun in a_nouns:
            if has_relationship(q_adj, a_noun):
                features["has_adjective_and_related_word_found"] = True
                break
        if features["has_adjective_and_related_word_found"]:
            break

    # 4. Entity Match (Simple noun overlap for demonstration)
    features["entity_match"] = bool(set(q_nouns).intersection(set(a_nouns)))

    # 5. Singular/Multiple Entity Matching
    features["has_singular_entity_and_match_found"] = q_components["singular_entity"] and a_components["singular_entity"]
    features["has_multiple_entity_and_match_found"] = q_components["multiple_entity"] and a_components["multiple_entity"]

    return features

def create_features_dataframe(question, answers):
    feature_list = []

    for answer in answers:
        features = feat_extract(question, answer)
        feature_list.append(features)

    df_features = pd.DataFrame(feature_list)

    # df_features['label'] = labels

    return df_features


################################################################################

def predict_when(df):
    with open('when_naive_bayes.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    
    predictions = loaded_model.predict_proba(df)
    return predictions

def predict_what(df):
    with open('what_naive_bayes.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    
    predictions = loaded_model.predict_proba(df)
    return predictions

def predict_who(df):
    with open('who_naive_bayes.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    predictions = loaded_model.predict_proba(df)
    return predictions

def predict_which(df):
    with open('which_naive_bayes.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    predictions = loaded_model.predict_proba(df)
    return predictions

def predict_how(df):
    with open('how_naive_bayes.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)  

    predictions = loaded_model.predict_proba(df)
    return predictions

def predict_yes_or_no(df):
    with open('yes_or_no_naive_bayes.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    
    predictions = loaded_model.predict_proba(df)
    return predictions

def predict_where(df):
    with open('where_naive_bayes.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)  

    predictions = loaded_model.predict_proba(df)
    return predictions

def predict_why(df):
    with open('why_naive_bayes.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file) 

    predictions = loaded_model.predict_proba(df)
    return predictions

def predict_other(df):
    with open('other_naive_bayes.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)  

    predictions = loaded_model.predict_proba(df)
    return predictions   
                 
# Define the location-related POS tag sequences
location_pos_sequences = {
    ('IN', 'NNP'),        # "in Paris"
    ('IN', 'DT', 'NN'),   # "in the park"
    ('IN', 'DT', 'NNS'),  # "in the cities"
    ('IN', 'IN', 'DT', 'NN'),  # "across from the bank"
}
def has_location_context(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    
    for seq in location_pos_sequences:
        for i in range(len(tagged_tokens) - len(seq) + 1):
            if tuple(tagged_tokens[j][1] for j in range(i, i + len(seq))) == seq:
                return True  # Found a match for location-related POS sequence
    return False  # No match found


#################################################################
### chatbot helper functions

# Function to categorize questions based on keywords
def categorize_questions(text):
    # story_id = str(uuid.uuid4())
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if (word not in string.punctuation) and (word.isalpha() or word.isnumeric())]
    # print(tokens)
    token_count = len(tokens)
    # print(token_count)
    if token_count<=2:
        return "incomplete123"

    if token_count > 20:
        return "story"
    

    text_lower = text.lower()
        # Check for the presence of keywords anywhere in the question
    if "what" in text_lower:
        return "what"
    elif "who" in text_lower:
        return "who"
    elif "how" in text_lower:
        return "how"
    elif "when" in text_lower:
        return "when"
    elif "where" in text_lower:
        return "where"
    elif "why" in text_lower:
        return "why"
    elif text_lower.startswith(("is","was","were", "are", "can", "do", "does", "did", "will", "would", "could", "should", "have", "has", "had", "might", "must")):
        return "yes_or_no"
    else:
        return "other"


def process_question(question,tag,story_id,db_path):

    print("tag : ..........", tag)
    conn = sqlite3.connect(db_path)
    sent = retrieve_story_sentences(story_id,conn)
    for i in sent:
      print(i)
    df = create_features_dataframe(question,sent)

    ans = []
    conn.close()

    if tag=='where':
        ans = predict_where(df)
    elif tag=='when':
        ans = predict_when(df)
    elif tag=='How':
        ans = predict_how(df)
    elif tag=='who':
        ans = predict_who(df)
    elif tag=='which':
        ans = predict_which(df)
    elif tag=='what':
        ans = predict_what(df)
    elif tag=='yes_or_no':
        ans = predict_yes_or_no(df)
    elif tag=='why':
        ans = predict_why(df)
    else:
        ans = predict_other(df)
    
    prob_sent = []
    
    for i in range(len(ans)):
      prob_sent.append((ans[i][1],sent[i]))
    prob_sent.sort(key=lambda x : x[0],reverse=True)
    return prob_sent[0][1]


##########################################################################
#  chatbot


custom_reflections = {
    "i am": "you are",
    "i was": "you were",
    "i": "you",
    "i'm": "you are",
    "i'd": "you would",
    "i've": "you have",
    "i'll": "you will",
    "my": "your",
    "you are": "I am",
    "you were": "I was",
    "you've": "I have",
    "you'll": "I will",
    "your": "my",
    "yours": "mine",
    "you": "me",
    "me": "you"
}

patterns = [
    (r"hi|hello|hey", ["Hello!", "Hi there!", "Hey! How can I assist you?"]),
    (r"what is your name?", ["I'm a chatbot created to help you.", "You can call me ChatBot!"]),
    (r"how are you?", ["I'm a program, so I don't have feelings, but thank you for asking!", "I'm here to assist you with any questions."]),
    (r"what can you do?", ["I can help you with general questions, chat with you, and assist you with simple tasks."]),
    (r"quit|exit|bye|goodbye", ["Goodbye! It was nice talking to you.", "Take care! See you soon."]),
    (r"incomplete123", [
    "It seems your input is incomplete. Could you provide more details?",
    "I think something might be missing in your input. Could you elaborate?",
    "Your input doesn’t seem complete. Can you clarify or provide more context?",
    "Hmm, it looks like you didn’t finish your input. Could you try again?",
    "I’m having trouble understanding your incomplete input. Can you add more details?",
    "It appears your input is missing some information. Could you provide the full details?",
    "I can’t respond properly to an incomplete input. Could you rephrase or complete it?",
    "It seems like there’s something missing in what you’re inputting. Could you try rephrasing?",
    "Your input isn’t quite complete. Can you clarify or provide more context?",
    "I’m not sure how to respond to an incomplete input. Could you provide more information?"
    ]),
    (r"unknown123",[
        "Sorry, I don’t have enough information to answer that. Could you provide more context?",
        "I’m not sure about that. Can you give me a bit more detail or rephrase the question?",
        "I don’t quite understand your question. Could you clarify or ask something else related to the story?",
        "Hmm, that’s a bit unclear. Could you explain further or ask something different?",
        "I don’t know the answer to that. Could you provide more details or ask about something else in the story?",
        "Unfortunately, I don’t have enough information to respond. Could you elaborate?",
        "That’s outside my knowledge for now. Can you give me more context or rephrase your question?",
        "I’m afraid I don’t have an answer to that. Would you like to ask something else?",
        "I couldn’t find any details on that. Maybe try rephrasing your question or adding more context?",
        "I don’t have the answer to that at the moment. Could you provide more information?"
        ]),
        (r".*",["unknown"]),
]


create_db(json_path)
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

word_story_dict, entity_story_mapping = load_stories_from_db(conn)


# Initialize chatbot with patterns and reflections
chatbot = Chat(patterns, custom_reflections)
cur_questions = []

# Start the chatbot conversation loop
print("Welcome to the chatbot! Type 'quit' or 'exit' to end the chat.")
while True:
    user_input = input("You: ")
    user_input = user_input.lower()
    
    response = chatbot.respond(user_input)
    if response == "unknown":
        tag = categorize_questions(user_input)
        if tag == "incomplete123":
            response = chatbot.respond(tag)
            print("\nChatbot:",response)
            continue

        if tag == "unknown123":
            response = chatbot.respond(tag)
        else:
            ques = user_input.lower().strip()
            # if len(cur_questions) == 10:
            #     cur_questions.pop(0)
            # cur_questions.append(ques)
            # relevent_stories = handle_questions(cur_questions, word_story_dict, entity_story_mapping)
            # print(relevent_stories)
            # print(relevent_stories[-1])
            # print("..................................................................")
            relevent_stories = retrieves_stories(ques,db_path)
            if not relevent_stories:
                response = chatbot.respond("unknown123")
                print("\nChatbot:",response)
                continue
            # print(relevent_stories)
            res = process_question(ques,tag,relevent_stories,db_path)
            if not res:
                response = chatbot.respond("unknown123")
                continue
            print("\nChatbot:",res)
            continue
    else:
        if user_input.lower().strip() in ["quit", "exit","bye"]:
            print("\nChatbot: Goodbye! Take care.")
            break
    # response = chatbot.respond(user_input)
        print("\nChatbot:", response)
        continue

conn.close()