# ChatBot_Project
# LONG-TERM MEMORY CHATBOT
# Author:
# Aman Sharma
# PROJECT OVERVIEW
 This project focuses on the development of a Long-Term Memory Chatbot capable of answering questions based on stories. Using a structured dataset and advanced preprocessing techniques, the chatbot performs reading comprehension tasks to provide relevant and contextually accurate answers.
# DATASET DESCRIPTION
I-Format: JSON containing 500 stories.

II-Contents: Stories, questions, answers, and span 
texts for answer extraction.

III-Questions per Story: 10–24 questions, totaling 7983 questions.

IV-Categories: What, Who, Which, How, Yes/No, When, Where, Why, Other.

#METHODOLOGY
1. Data Preprocessing:
   I-Removed irrelevant characters, converted text to lowercase, and applied tokenization. 

   II-Eliminated stopwords using NLTK and lemmatized words for consistency.
2- Story Identification:
   I-Extracted named entities using NLTK for enhanced entity-based analysis.
   II-Used synonyms and rare-word frequency analysis for better query matching.
3- Feature Engineering:
   I-Categorized question-answer pairs and derived specific features for each type (Where, When, etc.).
   II-Used word overlap and entity matching to improve context understanding.
4-Sentence Extraction:
   I-Retrieved relevant story sentences based on semantic similarity and contextual cues.
   II-Implemented fallback mechanisms for unclear answers.
5-Models Used:
   I-Naive Bayes Classifier:
    a. Achieved 62.59% accuracy.
    b. Efficient for predicting question-answer relevance.
   II-Random Forest Classifier:
    a. Achieved 60.57% accuracy.
    b. Used for validating user questions.
# RESULTS
  The chatbot demonstrated moderate accuracy with room for improvement in model tuning and feature engineering. Key achievements include efficient question validation and accurate sentence retrieval based on semantic relevance.
# REFERENCES
  I- https://doi.org/10.48550/arXiv.1810.03918
  II- https://doi.org/10.48550/arXiv.1808.07042
  III- https://doi.org/10.1145/3386723.3387897
