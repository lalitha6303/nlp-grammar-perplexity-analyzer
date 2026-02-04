from collections import defaultdict
import math
from tokenizer import tokenize

def train_lm(corpus_file):
    unigram = defaultdict(int)
    bigram = defaultdict(int)

    with open(corpus_file, encoding="utf-8") as f:
        for line in f:
            tokens = tokenize(line)
            for i in range(len(tokens)-1):
                unigram[tokens[i]] += 1
                bigram[(tokens[i], tokens[i+1])] += 1

    return unigram, bigram


def perplexity(sentence, unigram, bigram):
    tokens = tokenize(sentence)
    if len(tokens) < 2:
        return float("inf")

    log_prob = 0
    for i in range(len(tokens)-1):
        prob = (bigram.get((tokens[i], tokens[i+1]), 1)) / (unigram.get(tokens[i], 1))
        log_prob += math.log(prob)

    return math.exp(-log_prob / len(tokens))
