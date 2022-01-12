
# Data manipulation
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os



# Text Analytics
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.probability import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

nltk.download('wordnet')  # run once
nltk.download('stopwords')  # run once

import gensim
from gensim import corpora
from gensim import models
from gensim.models import KeyedVectors

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

import spacy
# fsou  rom spacy.lang.en import English 
from spacy import displacy


import itertools
from collections import Counter

# spacy.cli.download("en_core_web_sm") one tome dowload
# Instantiate the English model: nlp 
nlp = spacy.load('en_core_web_sm')
# ,tagger=False, parser=False, matcher=False) 


def entityRecog(text):
    """
    Description:
    function is used to identify the named “real-world” objects, like persons, companies or location.
    
    Inputs:
    text - list with text for entity recognition (output from readData function)
    
    Outputs:
    df - dataframe with all the entities and labels
    """

    docs = pd.DataFrame()

    # Create a new document: doc 
    docs = [ nlp(i) for i in text]

    # Entity Recognition starts here
    entity_df = pd.DataFrame()

    ent_text = []
    ent_label = []
    doc_index = []
    i = 0

    # Print all of the found entities and their labels 
    for doc in docs:
        for ent in doc.ents: 
            doc_index.append(i)
            ent_text.append(ent.text)
            ent_label.append(ent.label_)
        i=i+1

    entity_df = {'index':doc_index,'Label':ent_label,'Text':ent_text}

    entity_df = pd.DataFrame(entity_df)

    # Tokenization and Lemmatization
   # docs = [ nlp(i) for i in text]
    token_df = pd.DataFrame()

    doc_token = []
    doc_lemma = []
    doc_index = []
    i=0

    for doc in docs:
        for token in doc:
            doc_index.append(i)
            doc_token.append(token.text)
            doc_lemma.append(token.lemma_)
        i=i+1

    token_df = {'index':doc_index,'token':doc_token,'Lemma':doc_lemma}
    token_df = pd.DataFrame(token_df)

    return entity_df,token_df

def tokenize_df(df):
    """
    Description:
    function is tokens the tweets
    
    Inputs:
    inputs - dataframe you want to tokenize
    
    Outputs:
    word_tok_df - tokenize dataframe
    """
    documents = [sent_tokenize(text) for text in df['body']]
    word_tok  = []
    
    stop_words = set(stopwords.words('english'))

    for docs in documents:
        word_tokens  = [word_tokenize(doc) for doc in docs]
        word_tokens1 = [[txt for txt in i if txt.isalpha() and txt not in stop_words] for i in word_tokens]
        word_tokens2 = list(itertools.chain.from_iterable(word_tokens1))
        word_tokens2 = [word for word in word_tokens2]
        word_tok.append(word_tokens2)
        
    df['tokens'] = word_tok
    # df['tokens'] = df.word_tok.map(tokenReplace)
  #  word_tok_df      = {'word_tok':word_tok}
  #  word_tok_df      = pd.DataFrame(word_tok_df)

    return df


def googleModel(df,limit = 100000):
    """
    Description:
    function is to create word2vec using Google's pre-trained Word2Vec model
    
    Inputs:
    df,limit - tokenized dataframe you want to extract, limit in the google model
    
    Outputs:
    model - google model.
    word_tok_vec_avg - vectors after passed through google pretrained word2vec model
    word_tok_df - tokenize dataframe
    """


    if os.path.isfile('/Users/minu/Downloads/GoogleNews-vectors-negative300.bin.gz'):
        # get news as tokens
        print("split the tweet into tokens..")
        df = tokenize_df(df)
        print("tokenization is completed.")
        print("Google work to vector process started..")
        #word 2 vector using google pretrained model
        model     = gensim.models.KeyedVectors.load_word2vec_format('/Users/minu/Downloads/GoogleNews-vectors-negative300.bin.gz', binary=True, limit = limit) 
        #word_tok_vec     = [np.array([model.wv[word] for word in each if word in model.wv])  for each in df['tokens']] 
        word_tok_vec     = [np.array([model[word] for word in each if word in model])  for each in df['tokens']] 
        word_tok_vec_avg = np.hstack([np.mean(news, axis = 0)[:,np.newaxis] for news in word_tok_vec]).T 
        print("Vectorisation is completed..")
        return word_tok_vec_avg,model,df
    else:
        print('Error as File : GoogleNews-vectors-negative300.bin.gz does not exist. Please download it from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit to ../src/features folder')
        return None,None,None
    
    
