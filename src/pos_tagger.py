import nltk
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

def pos_tag(sentence):
    """
    Takes a sentence string and returns POS tags
    """
    tokens = word_tokenize(sentence)   # convert string → list of words
    return nltk.pos_tag(tokens)
