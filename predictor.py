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


p2 = Predictor(MaxEnt2(TRAIN_DATA_CORPUS, ngram_size=2, pref_suff_uniqueness=1))
p2.tag_document(DEV_DATA_CORPUS, MODEL_2_OUTPUT_FILE)
