import numpy as np
import json

import torch

from utils import tokenize, stem, embedding


def prepare_data(url):

    with open(url,  encoding='utf-8', errors='ignore') as f:
        data = json.load(f)

    all_words = []
    tags = []
    pattern_and_tag = []

    ignore_chars = ["؟", "!", ".", "،", ":", "؛"]

    for instance in data["data"]:
        tag = instance["tag"]
        tags.append(tag)

        for pattern in instance["patterns"]:
            tokenized_pattern = tokenize(pattern)
            stemmed_pattern = [stem(word) for word in tokenized_pattern if word not in ignore_chars]
            all_words.extend(stemmed_pattern)
            pattern_and_tag.append((stemmed_pattern, tag))

    all_words = sorted(set(all_words))

    X_train = []
    y_train = []

    for X, y in pattern_and_tag:
        X = embedding(X, all_words)
        X_train.append(X)

        y = tags.index(y)
        y_train.append(y)

    X_train = torch.from_numpy(np.array(X_train)).to(dtype=torch.float)
    y_train = torch.from_numpy(np.array(y_train)).to(dtype=torch.long)

    return X_train, y_train, all_words, tags

