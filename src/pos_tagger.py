import nltk

# Download correct NLTK resources for Streamlit Cloud (Python 3.13)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

def pos_tag(tokens):
    return nltk.pos_tag(tokens)
