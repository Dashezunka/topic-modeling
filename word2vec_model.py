import sys
import csv
from os.path import join, isfile
from os import listdir

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec,KeyedVectors

from word2vec_options import *
from text_preprocessing import *

# Build tokenized and lemmatized wordsim_corpus without stop words

def train_model():
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

    model = Word2Vec(size=VECTOR_SIZE, alpha=LEARNING_RATE, window=CONTEXT_WIDTH, sg=word2vec_model_mode,
                              min_count=1, negative=NEGATIVE_SAMPLES)

    model.build_vocab(sentences=corpus)
    model.train(sentences=corpus, total_examples=len(corpus), epochs=20)
    return model


# Compare model results for wordsim words with gold standard values
def compare_with_wordsim(word2vec_model):
    with open('data/wordsim_similarity_goldstandard.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        result = []
        for row in csv_reader:
            word1 = row[0].lower()
            word2 = row[1].lower()
            wordsim_similarity = row[2]
            # check if both words from wordsim exist in model dictionary
            model_similarity = None
            if word1 in word2vec_model.wv.vocab and word2 in word2vec_model.wv.vocab:
                model_similarity = word2vec_model.wv.similarity(word1, word2)
            result.append((word1, word2, wordsim_similarity, model_similarity))
        return result


word2vec_model = None

if WORD2VEC_MODE == "TRAIN_AND_SAVE":
    word2vec_model  = train_model()
    word2vec_model.save(MODEL_SAVE_PATH)
elif WORD2VEC_MODE == "LOAD":
    word2vec_model = Word2Vec.load(WORD2VEC_MODEL_PATH)

# Make a comparison of model results and gold standard values for words from wordsim353 corpus.
res = compare_with_wordsim(word2vec_model)

# Save results as csv
print("Save results as csv")
# for wordsim analysis
with open('data/results/wordsim_analysis_{0}_width{1}_vsize{2}_nsample{3}_max.csv'
                  .format('word2vec', CONTEXT_WIDTH, VECTOR_SIZE, NEGATIVE_SAMPLES), 'w') as out:
    analysis_out = csv.writer(out)
    analysis_out.writerow(['word1', 'word2', 'golden_value', 'model_value'])
    for word1, word2, golden_value, model_value in res:
        analysis_out.writerow([word1, word2, golden_value, model_value])

#print(word2vec_model.wv.similar_by_word('walk'))
