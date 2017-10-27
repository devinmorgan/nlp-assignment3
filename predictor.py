from maxent_model_1 import MaxEnt1
from maxent_model_2 import MaxEnt2

TRAIN_DATA_CORPUS = "data/train.tag"
DEV_DATA_CORPUS = "data/dev.tag"
TEST_DATA_CORPUS = "data/test.tag"

MODEL_1_OUTPUT_FILE = "output/output_test1.tag"
MODEL_2_OUTPUT_FILE = "output/output_test2.tag"


class Predictor:
    def __init__(self, model):
        model.train()
        self.model = model

    @staticmethod
    def write_output_to_file(output, file_name):
        with open(file_name, 'w') as f:
            for line in output:
                f.write(line + "\n")

    def tag_document(self, corpus_path, output_file):
        with open(corpus_path) as f:
            output = []
            while True:
                id = f.readline().strip()
                output.append(id)
                text = f.readline().strip()
                if text:
                    predicted_text = self.model.tag_text(text)
                    output.append(predicted_text)
                else:
                    break
            Predictor.write_output_to_file(output, output_file)


# p1 = Predictor(MaxEnt1(TRAIN_DATA_CORPUS))
# p1.tag_document(DEV_DATA_CORPUS, MODEL_1_OUTPUT_FILE)
p2 = Predictor(MaxEnt2(TRAIN_DATA_CORPUS, ngram_size=2))
p2.tag_document(DEV_DATA_CORPUS, MODEL_2_OUTPUT_FILE)

# import numpy as np
# m1 = MaxEnt1()
# m1_train_d, _ = m1.extract_data_and_labels(TRAIN_DATA_CORPUS)
# m2 = MaxEnt2(context_word_size=1)
# m2_train_d, _ = m2.extract_data_and_labels(TRAIN_DATA_CORPUS)
# print(np.array_equal(m1_train_d, m2_train_d))