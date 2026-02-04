import os
from tokenizer import tokenize
from spell_checker import load_dictionary, correct_spelling
from grammar_candidates import generate_candidates
from language_model import train_lm, perplexity
from pos_tagger import pos_tag
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(BASE_DIR, "data", "brown_corpus.txt")


sentence = input("Enter a sentence: ").strip().lower()
if not sentence:
    print("Invalid input")
    exit()

# Train LM
unigram, bigram = train_lm(CORPUS)

# Load dictionary
vocab = load_dictionary(CORPUS)

# Tokenize
tokens = tokenize(sentence)

# Spelling correction
tokens = correct_spelling(tokens, vocab)

# Grammar candidates
candidates = generate_candidates(tokens)

# Choose best sentence using perplexity
best_sentence = min(candidates, key=lambda s: perplexity(s, unigram, bigram))

print("\nOriginal Sentence:", sentence)
print("Corrected Sentence:", best_sentence)

print("\nPerplexity Scores:")
for s in candidates:
    print(s, "→", round(perplexity(s, unigram, bigram), 3))

print("\nPoS Tags:")
for w, t in pos_tag(best_sentence):
    print(w, "→", t)
