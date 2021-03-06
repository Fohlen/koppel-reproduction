import json

import requests
import os
from typing import List, Tuple
from collections import Counter
from itertools import chain

import numpy as np
from datasets import load_dataset
from nltk import word_tokenize, pos_tag, ngrams

POS = Tuple[str, str]


def extract_trigrams(texts: List[str]) -> List[Tuple[str]]:
    trigrams = chain(*[ngrams(text, 3) for text in texts])
    trigram_frequency = Counter(trigrams)
    return [freq[0] for freq in trigram_frequency.most_common(100)]


def extract_bigrams(texts: List[str]) -> List[Tuple[str]]:
    bigrams = chain(*[ngrams(text, 2) for text in texts])
    bigram_frequency = Counter(bigrams)
    return [freq[0] for freq in bigram_frequency.most_common(100)]


def extract_rare_pos_tags(pos_per_text: List[Tuple[POS]]) -> List[Tuple[POS]]:
    pos_tags = chain(*pos_per_text)
    pos_frequency = Counter(pos_tags)
    return [bigram[0] for bigram in pos_frequency.most_common()[::-250]]


def get_matches(annotation: str) -> List[str]:
    data = json.loads(annotation)
    return [match["rule"]["id"] for match in data["matches"]]


class KoppelVectorizer:
    def __init__(self, features: List[str] = None, stopwords_path: str = None, language_tool: str = 'http://localhost:8081/v2/check'):
        if features is None:
            features = ['stopwords', 'trigrams', 'bigrams', 'rare_pos_tags']
        if stopwords_path is None:
            stopwords_path = os.path.join(os.getcwd(), 'onix.txt')
        self.language_tool = language_tool
        self.stopwords = load_dataset('text', data_files=[stopwords_path], split='train') if 'stopwords' in features else []
        self.annotations = load_dataset('text', data_files=[os.path.join(os.getcwd(), 'languagetool-short.txt')], split='train') if 'annotation' in features else []
        self.tokens_per_text = None
        self.pos_per_text = None
        self.trigrams_by_frequency = None
        self.bigrams_by_frequency = None
        self.rare_pos_tags_by_frequency = None
        self.features = features

    def fit(self, texts: List[str]):
        self.tokens_per_text = [word_tokenize(text) for text in texts]
        self.trigrams_by_frequency = extract_trigrams(texts) if 'trigrams' in self.features else []
        self.bigrams_by_frequency = extract_bigrams(texts) if 'bigrams' in self.features else []
        self.pos_per_text = [pos_tag(tokens) for tokens in self.tokens_per_text] if 'rare_pos_tags' in self.features else []
        self.rare_pos_tags_by_frequency = extract_rare_pos_tags(self.pos_per_text) if 'rare_pos_tags' in self.features else []

    def transform(self, texts: List[str], annotations: List[str] = []) -> np.ndarray:
        stopwords_enabled = 'stopwords' in self.features
        bigrams_enabled = 'bigrams' in self.features
        trigrams_enabled = 'trigrams' in self.features
        rare_pos_enabled = 'rare_pos_tags' in self.features
        annotation_enabled = 'annotation' in self.features
        vector_length = (stopwords_enabled * len(self.stopwords)) + \
                        (trigrams_enabled * len(self.trigrams_by_frequency)) + \
                        (bigrams_enabled * len(self.bigrams_by_frequency)) + \
                        (rare_pos_enabled * len(self.rare_pos_tags_by_frequency)) + \
                        (annotation_enabled * len(self.annotations))

        matrix = np.empty([len(texts), vector_length])
        for index, text in enumerate(texts):
            tokens = word_tokenize(text)
            vector = np.zeros(vector_length)

            if stopwords_enabled:
                for j, stopword in enumerate(self.stopwords['text']):
                    c = tokens.count(stopword)
                    vector[j] += c

            if bigrams_enabled:
                offset = (stopwords_enabled * len(self.stopwords))
                bigrams = list(ngrams(text, 2))
                for k, bigram in enumerate(self.bigrams_by_frequency):
                    c = bigrams.count(bigram)
                    vector[offset+k] += c

            if trigrams_enabled:
                offset = (stopwords_enabled * len(self.stopwords)) + (bigrams_enabled * len(self.bigrams_by_frequency))
                trigrams = list(ngrams(text, 3))
                for l, trigram in enumerate(self.trigrams_by_frequency):
                    c = trigrams.count(trigram)
                    vector[offset+l] += c

            if rare_pos_enabled:
                offset = (stopwords_enabled * len(self.stopwords)) \
                         + (bigrams_enabled * len(self.bigrams_by_frequency)) \
                         + (trigrams_enabled * len(self.trigrams_by_frequency))
                pos_bigrams_per_text = list(ngrams(pos_tag(tokens), 2))
                for m, pos in enumerate(self.rare_pos_tags_by_frequency):
                    c = pos_bigrams_per_text.count(pos)
                    vector[offset+m] += c

            if annotation_enabled:
                offset = (stopwords_enabled * len(self.stopwords)) \
                         + (bigrams_enabled * len(self.bigrams_by_frequency)) \
                         + (trigrams_enabled * len(self.trigrams_by_frequency)) \
                         + (rare_pos_enabled * len(self.rare_pos_tags_by_frequency))

                matches = get_matches(annotations[index])
                for n, annotation in enumerate(self.annotations['text']):
                    c = matches.count(annotation)
                    vector[offset+n] += c

            matrix[index] = vector
        return matrix

    def fit_transform(self, texts, annotations) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts, annotations)
