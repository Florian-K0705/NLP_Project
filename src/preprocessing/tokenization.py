import nltk
from nltk.tokenize import word_tokenize

def tokenize_text(text):
    try:
        tokens = word_tokenize(text.lower())
    except:
        return []

    return tokens