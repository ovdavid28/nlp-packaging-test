import spacy
import sys

nlp = spacy.load("en_core_web_sm")
from collections import Counter
import nltk, re, pprint
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from spacy.lang.en.stop_words import STOP_WORDS

counts = Counter()


def lemmat(doc):
  lemmas=[token.lemma_ for token in doc if not token.is_punct and not token.is_space and not token.lower_ in STOP_WORDS]
  return lemmas
#lemmat(nlp(t0))


def tf_fun(doc,word):
  list1=lemmat(doc)
  letter=nlp(word)
  sil=lemmat(letter)
  times=list1.count(' '.join(sil))
  return times

#tf_fun(nlp(t1),'China')



list_doc=[t0, t1, t2, t3, t4, t5, t6]

def idf_fun(docs,word):
  row=[]
  for i in docs:
    lemma=lemmat(nlp(i))
    times=lemma.count(''.join(lemmat(nlp(word))))
    if times > 0:
      row.append(1)
  if len(row) > 0:
    return (1/len(row))
  else:
    return 0

#idf_fun(list_doc,'China')


def tf_idf(word, doc, docs):
    tf = tf_fun(doc, word)
    idf = idf_fun(docs, word)
    return tf * idf


#tf_idf('China', nlp(t1), list_doc)


def all_lemmas(docs):
  l=[]
  for i in docs:
    lemma=lemmat(nlp(i))
    l.append(lemma)
  flat_list = [item for sublist in l for item in sublist]
  return set(flat_list)

#all_lemmas(list_doc)


def tf_idf_doc(doc, docs):
    lemmas = all_lemmas(docs)
    dict_lemmas = {i: tf_idf(i, doc, docs) for i in lemmas}
    return dict_lemmas


#tf_idf_doc(nlp(t3), list_doc)


import pandas as pd


for doc in list_doc:
  dict_df=(tf_idf_doc(nlp(doc),list_doc))
#  print(dict_df)


  def dataframe(docs):
      index = [0, 1, 2, 3, 4, 5, 6]
      dt = []
      for doc in docs:
          dt.append(tf_idf_doc(nlp(doc), docs))
      df = pd.DataFrame(dt, index=index)
      return df


#  dataframe(list_doc)