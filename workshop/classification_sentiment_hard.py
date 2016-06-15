# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:19:29 2016

@author: franckbardol
"""

import numpy as np
import sklearn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# data :
#pd.read_csv('https://raw.githubusercontent.com/vineetdhanawat/twitter-sentiment-analysis/master/datasets/Sentiment%20Analysis%20Dataset.csv' , sep=','
data = pd.read_csv('sentiment_hard.csv' , sep=',' , index_col = 'ItemID')

NB_LINE_TRAIN = np.int(0.7 * data.shape[0])

X_train = data.ix[ :NB_LINE_TRAIN , 'SentimentText']
Y_train = data.ix[ :NB_LINE_TRAIN , 'Sentiment'].copy()

X_test = data.ix[NB_LINE_TRAIN + 1: , 'SentimentText']
Y_test = data.ix[NB_LINE_TRAIN + 1: , 'Sentiment'].copy()

# ================================= METHOD 1
# ____________________ EXTRACTING FEATURES
# turn text to vector to perform analysis
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train.values)
# TF-IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, Y_train.values.astype(str))
# PREDICT !
X_new_counts = count_vect.transform(X_test.values)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)
print "predicted method 1 =", predicted

# ================================= METHOD 2 (better)
# ______________ PIPELINE (vectorizer -> transformer -> classifier)
text_clf = Pipeline([('my_vect', CountVectorizer()),
    ('my_tfidf', TfidfTransformer()),
    ('my_clf', MultinomialNB()),])
text_clf = text_clf.fit(X_train.values, Y_train.values.astype(str))

# run trained classifier on these UNSEEN data
predicted = text_clf.predict(X_test.values)
print "predicted method 2 =" , predicted

# ================================ REPORT
print(metrics.classification_report(Y_test.values.astype(str), predicted))
print metrics.confusion_matrix(Y_test.values.astype(str), predicted)

