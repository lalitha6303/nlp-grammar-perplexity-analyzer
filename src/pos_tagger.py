import nltk

def pos_tag(sentence):
    nltk.download("averaged_perceptron_tagger", quiet=True)
    tokens = sentence.split()
    return nltk.pos_tag(tokens)
