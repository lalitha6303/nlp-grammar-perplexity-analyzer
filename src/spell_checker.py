from edit_distance import min_edit_distance

def load_dictionary(corpus_file):
    vocab = set()
    with open(corpus_file, encoding="utf-8") as f:
        for line in f:
            vocab.update(line.split())
    return vocab

def correct_spelling(tokens, vocab):
    corrected = []
    for w in tokens:
        if w in vocab:
            corrected.append(w)
        else:
            best = min(vocab, key=lambda x: min_edit_distance(w, x))
            corrected.append(best)
    return corrected
