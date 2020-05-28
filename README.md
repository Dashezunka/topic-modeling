# Models for topic modeling and similarity analysis
There are two models for topic modeling and similarity analysis: word2vec model and pLSA ones.

## Word2vec model
### Description
The algorithm works in the following way:
- Builds tokenized and lemmatized corpus removing stop words. 
- Trains a model on this corpus.
- Makes a comparison of model results and *gold standard* values for words from `wordsim353` corpus.
- Saves results in csv-file.

**Note:** Model can work in two modes: *TRAIN_AND_SAVE* or *LOAD* using existing word2vec model.  
All word2vec params and model modes are described in `options.py` and can be adjusted there.

### Install and configure
You need to install dependencies from `requirements.txt` using
`pip3 install -r requirements.txt`   

Adjust all model params and *CORPUS_PATH* in `options.py` before starting the model.  

### Running command
Try `python3 word2vec_model.py` in the project directory.

## pLSA model
### Description
The algorithm works in the following way:
- Builds a tokenized and lemmatized document corpus without stop words, 
that consists of tuples (doc_name, doc_sentences_list).
- Builds frequency dicts for documents.
- 'Trains' a model on the corpus using pLSA algorithm.
- Saves *documents-topics distribution matrix* and *matrix of top words for every topic* as csv-files.
- Makes a comparison of model results and *gold standard* values for words from `wordsim353` corpus.
- Saves results of the comparison in csv-file.

**Note:** All pLSA params are described in `options.py` and can be adjusted there.

### Install and configure
You need to install dependencies from `requirements.txt` using
`pip3 install -r requirements.txt`   

Adjust all model params and *CORPUS_PATH* in `options.py` before starting the model.    

### Running command
Try `python3 plsa_model.py` in the project directory.