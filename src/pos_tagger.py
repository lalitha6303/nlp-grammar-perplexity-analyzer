import nltk
from nltk.tokenize import word_tokenize

# Download ALL required NLTK resources for Streamlit Cloud (Python 3.13)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

def pos_tag(sentence):
    tokens = word_tokenize(sentence)
    return nltk.pos_tag(tokens)
