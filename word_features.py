from string import ascii_uppercase
import numpy as np


CAPITAL_LETTERS = set(ascii_uppercase)
SPECIAL_CHARACTERS = set("#$%&*()-_+=[]|:;\"\'<,>.?/")


class FeatureExtractor:
    def __init__(self, train_corpus, ngram_size, pref_suff_uniqueness=30):
        self.ngram_size = ngram_size
        self.min_uniqueness = pref_suff_uniqueness
        self.prefixes, self.prefix_index_mapping, self.suffixes, self.suffix_index_mapping = self.extract_prefixes_and_suffixes(train_corpus)

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

    def extract_prefixes_and_suffixes(self, corpus_path):
        words = set()
        with open(corpus_path) as f:
            while True:
                f.readline()  # Skip ID lines
                text = f.readline().strip()
                if text:
                    words.update(text.split(" "))
                else:
                    break
        prefixes = {}
        suffixes = {}
        for word in words:
            for i in range(2, 5):
                pref = word[:i]
                prefixes[pref] = prefixes.get(pref, 0) + 1
                suff = word[-i:]
                suffixes[suff] = suffixes.get(suff, 0) + 1
        strong_prefixes = set([pref for pref in prefixes.keys() if prefixes[pref] > self.min_uniqueness])
        prefix_index_mapping = { pref:i for i, pref in enumerate(strong_prefixes) }
        strong_suffixes = set([suff for suff in suffixes.keys() if suffixes[suff] > self.min_uniqueness])
        suffix_index_mapping = { suff:i for i, suff in enumerate(strong_suffixes) }
        return strong_prefixes, prefix_index_mapping, strong_suffixes, suffix_index_mapping

    def get_feature_vector_for_word(self, word, prev_labels):
        prefix_features = self.get_prefix_features(word)
        suffix_features = self.get_suffix_features(word)
        word_length_feature = np.array([[len(word)]])
        capital_letters_feature = self.get_capital_letters_features(word)
        special_characters_feature = self.get_special_characters_feature(word)
        previous_genes_features = np.array([prev_labels])
        return np.concatenate(
            (prefix_features,
             suffix_features,
             word_length_feature,
             capital_letters_feature,
             special_characters_feature,
             previous_genes_features), axis=1)

    def get_prefix_features(self, word):
        prefix_feature_vector = np.zeros((1, len(self.prefixes)))
        for pref, index in self.prefix_index_mapping.iteritems():
            if word[:len(pref)] == pref:
                prefix_feature_vector[:, index] = 1
        return prefix_feature_vector

    def get_suffix_features(self, word):
        suffix_feature_vector = np.zeros((1, len(self.suffixes)))
        for suff, index in self.suffix_index_mapping.iteritems():
            if word[-len(suff):] == suff:
                suffix_feature_vector[:, index] = 1
        return suffix_feature_vector

    def feature_vector_size(self):
        prefixes_size = len(self.prefixes)
        suffixes_size = len(self.suffixes)
        word_length_size = 1
        capital_letters_size = 1
        special_characters_size = 1
        preceding_genes_size = self.ngram_size - 1
        return prefixes_size \
               + suffixes_size \
               + word_length_size \
               + capital_letters_size \
               + special_characters_size \
               + preceding_genes_size
