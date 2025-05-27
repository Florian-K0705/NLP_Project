import nltk
from nltk.tokenize import word_tokenize

def tokenize_text(text):
    tokens = word_tokenize(text.lower())

    return tokens