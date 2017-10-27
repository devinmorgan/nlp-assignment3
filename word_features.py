from string import ascii_uppercase, digits
import numpy as np


CAPITAL_LETTERS = set(ascii_uppercase)
SPECIAL_CHARACTERS = set("#$%&*()-_+=[]|:;\"\'<,>.?/")
DIGITS = set(digits)
GENE_1_LABEL = "GENE1"
GENE_2_LABEL = "GENE2"
TAG_LABEL = "TAG"


class FeatureExtractor:
    def __init__(self, train_corpus, ngram_size, pref_suff_uniqueness=30):
        self.ngram_size = ngram_size
        self.min_uniqueness = pref_suff_uniqueness
        self.char_ngrams, self.char_ngram_mapping = self.extract_char_ngrams(train_corpus)

    @staticmethod
    def get_capital_letters_features(word):
        count = 0
        for c in word:
            if c in CAPITAL_LETTERS:
                count += 1
        return np.array([[count]])

    @staticmethod
    def get_special_characters_feature(word):
        count = 0
        for c in word:
            if c in SPECIAL_CHARACTERS:
                count += 1
        return np.array([[count]])

    @staticmethod
    def get_digits_features(word):
        count = 0
        for c in word:
            if c in DIGITS:
                count += 1
        return np.array([[count]])

    def extract_char_ngrams(self, corpus_path):
        words = set()
        with open(corpus_path) as f:
            while True:
                f.readline()  # Skip ID lines
                text = f.readline().strip()
                if text:
                    for token in text.split(" "):
                        parts = token.split("_")
                        word = parts[0]
                        tag = parts[1]
                        if tag == GENE_1_LABEL or tag == GENE_2_LABEL:
                            words.add(word)
                else:
                    break
        char_ngrams = {}
        for word in words:
            if len(word) >= 3:
                for i in xrange(len(word) - 2):
                    char_ngram = word[i:i+3]
                    char_ngrams[char_ngram] = char_ngrams.get(char_ngram, 0) + 1
            else:
                char_ngrams[word] = char_ngrams.get(word, 0) + 1
        strong_ngrams = set([char_ngram for char_ngram in char_ngrams.keys() if char_ngrams[char_ngram] >= self.min_uniqueness])
        ngram_index_mapping = { char_ngram:i for i, char_ngram in enumerate(strong_ngrams) }
        return strong_ngrams, ngram_index_mapping

    def get_feature_vector_for_word(self, word, prev_labels):
        char_ngram_features = self.get_ngram_features(word)
        word_length_feature = np.array([[len(word)]])
        capital_letters_feature = FeatureExtractor.get_capital_letters_features(word)
        digits_feature = FeatureExtractor.get_digits_features(word)
        special_characters_feature = FeatureExtractor.get_special_characters_feature(word)
        previous_genes_features = np.array([prev_labels])
        return np.concatenate(
            (char_ngram_features,
             word_length_feature,
             capital_letters_feature,
             digits_feature,
             special_characters_feature,
             previous_genes_features), axis=1)

    def get_ngram_features(self, word):
        ngram_feature_vector = np.zeros((1, len(self.char_ngrams)))
        for char_ngram, index in self.char_ngram_mapping.iteritems():
            if char_ngram in word:
                ngram_feature_vector[:, index] = 1
        return ngram_feature_vector

    def feature_vector_size(self):
        char_ngrams_size = len(self.char_ngrams)
        word_length_size = 1
        capital_letters_size = 1
        digits_size = 1
        special_characters_size = 1
        preceding_genes_size = self.ngram_size - 1
        return char_ngrams_size \
               + word_length_size \
               + capital_letters_size \
               + digits_size \
               + special_characters_size \
               + preceding_genes_size
