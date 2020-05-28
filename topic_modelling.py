import sys
from os.path import join, isfile
from os import listdir

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

from word2vec_options import *
from text_preprocessing import *

# Build tokenized and lemmatized wordsim_corpus without stop words
corpus = []
document_name_list = [f for f in listdir(CORPUS_PATH) if isfile(join(CORPUS_PATH, f))]
for name in document_name_list:
    with open(join(CORPUS_PATH, name), 'r') as doc:
        text = doc.read()
        sentences = tokenize_text(text)
        for sentence in sentences:
            lemmas = lemmatize_text(sentence)
            preprocessed_sentence = remove_stop_words(lemmas)
            if len(preprocessed_sentence) < 2:
                continue
            corpus.append(preprocessed_sentence)

word2vec_model_mode = None
if WORD2VEC_MODEL == 'SKIPGRAM':
    word2vec_model_mode = 1
elif WORD2VEC_MODEL == 'CBOW':
    word2vec_model_mode = 0
else:
    print("Unrecognized WORD2VEC_MODEL: {0}. Choose one of 'CBOW' or 'SKIPGRAM'".format(WORD2VEC_MODEL))
    sys.exit(1)

word2vec_model = Word2Vec(size=VECTOR_SIZE, alpha=LEARNING_RATE, window=CONTEXT_WIDTH, sg=word2vec_model_mode, min_count=1)

word2vec_model.build_vocab(sentences=corpus)

word2vec_model.train(sentences=corpus, total_examples=len(corpus), epochs=5)

output = word2vec_model.predict_output_word(['football'])
print(output)
