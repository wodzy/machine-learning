# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:40:03 2015

@author: franckbardol
"""

import numpy as np
import sklearn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

df_train = pd.read_csv('sentiment_train.txt' , sep = '\t')
X_train = df_train.ix[ : , 'text']
Y_train = df_train.ix[ : , 'sentiment'].copy()
# translate text to category
Y_train[Y_train == 'pos'] = 1
Y_train[Y_train == 'neg'] = 2

df_test = pd.read_csv('sentiment_test.txt' , sep = '\t')
X_test = df_test.ix[ : , 'text']
Y_test = df_test.ix[ : , 'sentiment'].copy()
Y_test[Y_test == 'pos'] = 1
Y_test[Y_test == 'neg'] = 2

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

#more data to test here:
#raw.githubusercontent.com/vineetdhanawat/twitter-sentiment-analysis/master/datasets/Sentiment%20Analysis%20Dataset.csv