# -*- coding: utf-8 -*-

"""
Language-related data loading helper functions and class wrappers.
"""

import re
import torch
import codecs
import spacy
import random

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
MASK_TOKEN = '<mask>'
# SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
SENTENCE_SPLIT_REGEX = re.compile(r'([\w+\']+)')


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, a):
        if isinstance(a, int):
            return self.idx2word[a]
        elif isinstance(a, list):
            return [self.idx2word[x] for x in a]
        elif isinstance(a, str):
            return self.word2idx[a]
        else:
            raise TypeError("Query word/index argument must be int or str")

    def __contains__(self, word):
        return word in self.word2idx


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def set_max_len(self, value):
        self.max_len = value

    def load_file(self, filename):
        with codecs.open(filename, 'r', 'utf-8') as f:
            for line in f:
                line = line.strip()
                self.add_to_corpus(line)
        self.dictionary.add_word(UNK_TOKEN)
        self.dictionary.add_word(PAD_TOKEN)
        self.dictionary.add_word(MASK_TOKEN)

    def add_to_corpus(self, line):
        """Tokenizes a text line."""
        # Add words to the dictionary
        words = line.split()
        # tokens = len(words)
        for word in words:
            word = word.lower()
            self.dictionary.add_word(word)

    def tokenize(self, line, max_len=20, in_training=False):
        # Tokenize line contents

        if in_training:
            p = 0.0
        else:
            p = 0.0

        rnd = random.random()
        if rnd < p:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(line)
            words = [token.text.lower() if token.dep_ != 'nsubj' else MASK_TOKEN for token in doc]
            # print(words)
        else:
            words = SENTENCE_SPLIT_REGEX.split(line.strip())
            words = [w.lower() for w in words if len(w) > 0 and w != ' ']

            if len(words) < 1:
                words.append(PAD_TOKEN)


        if words[-1] == '.':
            words = words[:-1]

        if max_len > 0:
            if len(words) > max_len:
                words = words[:max_len]
            elif len(words) < max_len:
                words = [PAD_TOKEN] * (max_len - len(words)) + words

        tokens = len(words)
        ids = torch.LongTensor(tokens)
        token = 0
        for word in words:
            if word not in self.dictionary:
                word = UNK_TOKEN
            ids[token] = self.dictionary[word]
            token += 1

        return ids

    def __len__(self):
        return len(self.dictionary)