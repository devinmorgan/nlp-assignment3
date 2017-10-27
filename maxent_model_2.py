import numpy as np
from word_features import FeatureExtractor
from sklearn.linear_model import LogisticRegression


class MaxEnt2:
    def __init__(self, training_corpus_path, ngram_size, pref_suff_uniqueness):
        self.logistic = LogisticRegression()
        self.n = ngram_size
        self.fe = FeatureExtractor(training_corpus_path, ngram_size, pref_suff_uniqueness)

    def get_feature_vector_for_ngram(self, ngram):
        prev_labels = pass
        return self.fe.get_feature_vector_for_word(ngram)

    def extract_ngrams_and_labels(self, corpus_path):
        with open(corpus_path) as f:
            ngrams_to_label = {}
            while True:
                f.readline()  # Skip ID lines
                text = f.readline().strip()
                if text:
                    tokens = ["_TAG"] * (self.n - 1) + text.split(" ")
                    for i in xrange(self.n - 1, len(tokens)):
                        ngram = tuple([tokens[j].split("_")[0] for j in xrange(i - self.n + 1, i + 1)])
                        tag = tokens[i].split("_")[1]
                        ngrams_to_label[ngram] = wf.get_label_for_tag(tag)
                else:
                    break
            return ngrams_to_label

    def extract_data_and_labels(self, corpus_path):
        ngrams_to_labels = self.extract_ngrams_and_labels(corpus_path)
        feature_vectors = []
        labels = []
        # for ngram in ngrams_list:
        for ngram, label in ngrams_to_labels.iteritems():
            feature_vector = self.get_feature_vector_for_ngram(ngram)
            feature_vectors.append(feature_vector)
            label = ngrams_to_labels[ngram]
            labels.append(label)

        n = len(labels)
        d = wf.NUM_ACCEPTED_CHARACTERS * wf.MAXIMUM_WORD_LENGTH * self.n
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
        words_list = [""]*(self.n - 1) + [token.split("_")[0] for token in tokens]
        for i in xrange(len(words_list)):
            context_words = tuple([words_list[j] for j in xrange(i - self.n + 1, i + 1)])
            feature_vector = self.get_feature_vector_for_ngram(context_words)
            prediction = self.logistic.predict(feature_vector)
            tag = wf.NON_GENE_TAG
            if prediction == 1:
                tag = wf.GENE_TAG
            new_token = words_list[i] + "_" + tag
            new_tokens.append(new_token)
        return " ".join(new_tokens)

