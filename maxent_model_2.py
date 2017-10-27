import numpy as np
from word_features import FeatureExtractor
from sklearn.linear_model import LogisticRegression


GENE_1_LABEL = "GENE1"
GENE_2_LABEL = "GENE2"
TAG_LABEL = "TAG"


class MaxEnt2:
    def __init__(self, training_corpus_path, ngram_size, pref_suff_uniqueness):
        self.logistic = LogisticRegression()
        self.ngram_size = ngram_size
        self.fe = FeatureExtractor(training_corpus_path, ngram_size, pref_suff_uniqueness)

    @staticmethod
    def get_label_for_tag(tag):
        if tag == TAG_LABEL:
            return 0
        if tag == GENE_1_LABEL:
            return 1
        if tag == GENE_2_LABEL:
            return 1

    @staticmethod
    def add_new_and_remove_oldest(array, new_val):
        array.pop()
        array[:0] = new_val
        return array

    @staticmethod
    def extract_words_and_labels(corpus_path):
        with open(corpus_path) as f:
            words_to_labels = {}
            while True:
                f.readline()  # Skip ID lines
                text = f.readline().strip()
                if text:
                    for token in text.split(" "):
                        parts = token.split("_")
                        word = parts[0]
                        tag = parts[1]
                        words_to_labels[word] = MaxEnt2.get_label_for_tag(tag)
                else:
                    break
            return words_to_labels

    def extract_data_and_labels(self, corpus_path):
        words_to_labels = MaxEnt2.extract_words_and_labels(corpus_path)
        feature_vectors = []
        labels = []
        prev_labels = [0]*self.ngram_size
        for word, label in words_to_labels.iteritems():
            feature_vector = self.fe.get_feature_vector_for_word(word, prev_labels)
            feature_vectors.append(feature_vector)
            label = words_to_labels[word]
            labels.append(label)
            prev_labels = MaxEnt2.add_new_and_remove_oldest(prev_labels, label)
        n = len(labels)
        d = self.fe.feature_vector_size()
        data_matrix = np.zeros((n, d), dtype=bool)
        labels_vector = np.array(labels, dtype=bool)
        for i in range(n):
            data_matrix[i:i + 1, :] = feature_vectors[i]
        return data_matrix, labels_vector

    def train(self, training_data_file):
        train_d, train_l = self.extract_data_and_labels(training_data_file)
        self.logistic.fit(train_d, train_l)

    def tag_text(self, text):
        tokens = text.strip().split(" ")
        new_tokens = []
        words_list = [token.split("_")[0] for token in tokens]
        prev_labels = [0] * self.ngram_size
        for word in words_list:
            feature_vector = self.fe.get_feature_vector_for_word(word, prev_labels)
            prediction = self.logistic.predict(feature_vector)
            tag = TAG_LABEL
            if prediction == 1:
                tag = GENE_1_LABEL
            new_token = word + "_" + tag
            new_tokens.append(new_token)
            prev_labels = MaxEnt2.add_new_and_remove_oldest(prev_labels, prediction)
        return " ".join(new_tokens)

