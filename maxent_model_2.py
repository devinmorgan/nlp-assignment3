import numpy as np
import word_features as wf
from sklearn.linear_model import LogisticRegression


class MaxEnt2:
    def __init__(self, context_word_size):
        self.logistic = LogisticRegression()
        self.cws = context_word_size

    def get_feature_vector_for_ngram(self, ngram):
        vector = np.zeros((1, wf.FEATURE_VECTOR_SIZE * self.cws), dtype=bool)
        for i in xrange(len(ngram)):
            word = ngram[i]
            word_index_offset = i * wf.FEATURE_VECTOR_SIZE
            for j, c in enumerate(word):
                char_index_offset = j * wf.NUM_ACCEPTED_CHARACTERS
                char_value_offset = wf.char_index_map[c]
                feature_index = word_index_offset + char_index_offset + char_value_offset
                vector[:, feature_index] = 1
        return vector

    def extract_ngrams_and_labels(self, corpus_path):
        with open(corpus_path) as f:
            ngrams_to_label = {}
            ngrams_list = []
            while True:
                f.readline()  # Skip ID lines
                text = f.readline().strip()
                if text:
                    tokens = ["_TAG"] * (self.cws - 1) + text.split()
                    for i in xrange(self.cws - 1, len(tokens)):
                        ngram = tuple([tokens[j].split("_")[0] for j in xrange(i - self.cws + 1, i + 1)])
                        ngrams_list.append(ngram)
                        tag = tokens[i].split("_")[1]
                        ngrams_to_label[ngram] = wf.get_label_for_tag(tag)
                else:
                    break
            return ngrams_to_label, ngrams_list

    def extract_data_and_labels(self, corpus_path):
        ngrams_to_labels, ngrams_list = self.extract_ngrams_and_labels(corpus_path)
        ngrams_list = [""] * (self.cws - 1) + ngrams_list
        feature_vectors = []
        labels = []
        for ngram in ngrams_list:
            feature_vector = self.get_feature_vector_for_ngram(ngram)
            feature_vectors.append(feature_vector)
            label = ngrams_to_labels[ngram]
            labels.append(label)

        n = len(labels)
        d = wf.NUM_ACCEPTED_CHARACTERS * wf.MAXIMUM_WORD_LENGTH * self.cws
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
        words_list = [""]*(self.cws - 1) + [token.split("_")[0] for token in tokens]
        for i in xrange(len(words_list)):
            context_words = tuple([words_list[j] for j in xrange(i - self.cws + 1, i + 1)])
            feature_vector = self.get_feature_vector_for_ngram(context_words)
            prediction = self.logistic.predict(feature_vector)
            tag = wf.NON_GENE_TAG
            if prediction == 1:
                tag = wf.GENE_TAG
            new_token = words_list[i] + "_" + tag
            new_tokens.append(new_token)
        return " ".join(new_tokens)
