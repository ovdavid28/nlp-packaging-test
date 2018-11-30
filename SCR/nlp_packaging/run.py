import spacy
import sys

nlp = spacy.load("en_core_web_sm")
from collections import Counter
import nltk, re, pprint
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from spacy.lang.en.stop_words import STOP_WORDS

import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

list_doc=[t0, t1, t2, t3, t4, t5, t6]

for doc in list_doc:
  dict_df=(tf_idf_doc(nlp(doc),list_doc))

def dataframe(docs):
  index=[0, 1, 2, 3, 4, 5, 6]
  dt=[]
  for doc in docs:
    dt.append(tf_idf_doc(nlp(doc),docs))
  df = pd.DataFrame(dt,index=index)
  return df
df=dataframe(list_doc)



plt.figure(figsize = (20,5))
sns.heatmap(df, vmax=1,  vmin=0,  cmap="Blues_r")
plt.savefig('tf_idf.png')