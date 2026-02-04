import nltk

# Download required NLTK data (only once)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("punkt", quiet=True)

def pos_tag(tokens):
    return nltk.pos_tag(tokens)
