import numpy as np
import hazm

stemmer = hazm.stemmer.Stemmer()


def tokenize(src):
    return hazm.word_tokenize(src)


def stem(word):
    return stemmer.stem(word)


def embedding(src, all_words):
    src = [stem(word) for word in src]

    one_hot = np.zeros(len(all_words), dtype='float32')

    for idx, word in enumerate(all_words):
        if word in src:
            one_hot[idx] = 1

    return one_hot
