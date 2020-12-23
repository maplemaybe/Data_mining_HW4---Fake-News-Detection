# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 15:10:39 2020

@author: user
"""
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
stopwords = stopwords.words('english')
from sklearn.metrics import average_precision_score, precision_score,f1_score,recall_score
import matplotlib.pyplot as plt
%matplotlib inline

#將資料讀取出來
df_sample_submission = pd.read_csv('sample_submission.csv')
df_train = pd.read_csv('train.csv',delimiter='\t')
df_test = pd.read_csv('test.csv',delimiter='\t')

df_test = pd.concat([df_test,df_sample_submission['label']], axis=1)
df_test = df_test.drop(['id'], axis=1)
#df_sample_submission.head()

df_content = pd.concat([df_train,df_test], axis=0)
df_content = df_content.reset_index(drop=True)
df_content

df_content.drop(df_content.loc[df_content['label']=='label'].index, inplace=True)

def listToString(s):  
    str1 = ""  
    for ele in s:  
        str1 += ele  +' ' 
    return str1

# segmentation
import wordninja
# First split words that combine together
# ex. protectyourselffrom -> protect yourself from
segmented_content=[]
for i in range(len(df_content)):
    segmented_content.append(listToString(wordninja.split((df_content.iloc[i]['text']).replace('=', ''))))
df_content['segmented_content']=segmented_content

import nltk

# Lemmatization
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')

# Stemming
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

punctuations="?:!.,;*<>(){}[]·@#$%^&"

def lemmatize(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    for word in sentence_words:
        if word in punctuations:
            sentence_words.remove(word)

    combined_words=''
    for word in sentence_words:
        combined_words+= ' '+wordnet_lemmatizer.lemmatize(word, pos="v")
    return combined_words

porter=PorterStemmer()
def stemSentence(sentence):
    sentence_words=word_tokenize(sentence)
    combined_words=''
    for word in sentence_words:
        combined_words+= ' '+(porter.stem(word))
    return combined_words

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
stopwords = set(STOPWORDS)

cv = CountVectorizer(stop_words=(list(stopwords)+stop_words))

def get_tfidf_transformer(input_content):
    # Compute the IDF values
    # input with all training data    
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True).fit(cv.fit_transform(input_content).toarray())
    return tfidf_transformer

def tfidf(input_content, tfidf_transformer):
    # Initialize CountVectorizer
    word_count_vector = cv.fit_transform(input_content).toarray()
    # tf-idf scores
    tf_idf_vector=tfidf_transformer.transform(cv.transform(input_content))
    df_tf_idf_vector=pd.DataFrame(tf_idf_vector.toarray())
    return (df_tf_idf_vector)

#nltk.download('punkt')
# content: segmented, lemmatized, stemmed
lemmatized_content=[]
stemmed_content=[]
for i in range(len(df_content)):
    if i%2000==0:
        print(i)
    lemmatized_content.append(lemmatize(df_content.iloc[i]['segmented_content']))
    stemmed_content.append(stemSentence(df_content.iloc[i]['segmented_content']))

df_content['lemmatized_content']=lemmatized_content
df_content['stemmed_content']=stemmed_content

# count string length
lens = df_content.stemmed_content.str.len()
df_content['stemmed_content_length']=lens

# count words
count_words=[]
for i in range(len(df_content)):
    count_words.append(len(df_content.iloc[i]['stemmed_content'].split()))
df_content['stemmed_count_words']=count_words

count_words=[]
for i in range(len(df_content)):
    count_words.append(len(df_content.iloc[i]['segmented_content'].split()))
df_content['segmented_count_words']=count_words


#df_content.drop_duplicates(keep = False, inplace = True) 
df_content=df_content.reset_index(drop=True)
print(len(df_content))
df_content.head()

# tf-idf transformer
content_ifidf_transformer=get_tfidf_transformer(df_content['text'])
segmented_tfidf_transformer=get_tfidf_transformer(df_content['segmented_content'])
stemmed_tfidf_transformer=get_tfidf_transformer(df_content['stemmed_content'])
lemmatized_tfidf_transformer=get_tfidf_transformer(df_content['lemmatized_content'])

data = df_content.drop(['label'], axis=1)
target = df_content['label']
'''
# train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data,target, test_size = 0.2, random_state = 0)

'''
x_train = data.iloc[:4987]
y_train = target.iloc[:4987]
x_test = data.iloc[4987:]
y_test = target.iloc[4987:]

X_train = x_train.iloc[:, 7:]
X_test = x_test.iloc[:, 7:]

test_y = y_test.to_numpy()
test_y = test_y.astype(float)
test_y

#XGBoost
from sklearn.metrics import average_precision_score, precision_score,f1_score,recall_score,accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
XGB = XGBClassifier(
 n_estimators=300,
 max_depth=3,
 subsample=0.8,
 colsample_bytree=0.8,
 nthread=4,
 scale_pos_weight=1
        )

XGB.fit(X_train,y_train.astype(float))
y_pred=XGB.predict(X_test)

print('XGBClassifier:')
print('acc= ',accuracy_score(test_y,y_pred))
print('precision= ',precision_score(test_y, y_pred, average='macro'))
print('recall= ',recall_score(test_y, y_pred, average='macro'))
print('f1_score= ',f1_score(test_y, y_pred, average='weighted')  )

from sklearn.ensemble import GradientBoostingClassifier
gbr = GradientBoostingClassifier(n_estimators=300, max_depth=2, min_samples_split=2, learning_rate=0.1)
gbr.fit(X_train, y_train.astype(float))
#joblib.dump(gbr, 'train_model_result4.m')   # 保存模型
y_pred=gbr.predict(X_test)

print('GBDTClassifier:')
print('acc= ',accuracy_score(test_y,y_pred))
print('precision= ',precision_score(test_y, y_pred, average='macro'))
print('recall= ',recall_score(test_y, y_pred, average='macro'))
print('f1_score= ',f1_score(test_y, y_pred, average='weighted')  )

# lightgbm for classification
from lightgbm import LGBMClassifier
from matplotlib import pyplot

# fit the model on the whole dataset
LGBM = LGBMClassifier(n_estimators=300, max_depth=2, learning_rate=0.1)

LGBM.fit(X_train,y_train.astype(float))
y_pred=LGBM.predict(X_test)

print('LGBMClassifier:')
print('acc= ',accuracy_score(test_y,y_pred))
print('precision= ',precision_score(test_y, y_pred, average='macro'))
print('recall= ',recall_score(test_y, y_pred, average='macro'))
print('f1_score= ',f1_score(test_y, y_pred, average='weighted')  )