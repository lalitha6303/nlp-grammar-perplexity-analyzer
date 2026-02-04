import nltk
from nltk.corpus import brown

nltk.download("brown")

with open("data/brown_corpus.txt", "w", encoding="utf-8") as f:
    for sent in brown.sents():
        f.write(" ".join(sent).lower() + "\n")

print("✅ Brown corpus saved successfully!")
