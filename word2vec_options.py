# Options for word2vec model. You could train new model by corpus of text or load existing model
WORD2VEC_MODE = "TRAIN_AND_SAVE" # or "LOAD"

# Options for word2vec train mode
CORPUS_PATH = "data/wordsim_corpus/"

CONTEXT_WIDTH = 5
LEARNING_RATE = 0.03
VECTOR_SIZE = 300
NEGATIVE_SAMPLES = 10
WORD2VEC_MODEL = "CBOW"  # "SKIPGRAM" or "CBOW"

MODEL_SAVE_PATH = "data/models/wordsim.model"


# Options for word2vec load existing model
WORD2VEC_MODEL_PATH = "data/models/wordsim.model"


#MODE = "WORD2VEC"  # "WORDSIM_ANALYSIS"

