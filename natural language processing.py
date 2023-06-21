# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:05:15 2023

@author: MRUTYUNJAY
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"D:\Datascience Classes\25th april\4.CUSTOMERS REVIEW DATASET\Restaurant_Reviews.tsv",delimiter = '\t', quoting = 3)
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
from sklearn.feature_extraction.text import TfidfVectorizer
cv=TfidfVectorizer()
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training the Naive Bayes model on the Training set
'''
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
'''
# Training the Decision tree model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier1=LogisticRegression()
classifier1.fit(X_train,y_train)
y_pred1=classifier1.predict(X_test)
# SVM(Support Vector Machine)
from sklearn.svm import SVC
classifier2= SVC(kernel='poly')
classifier2.fit(X_train, y_train)
y_pred2=classifier2.predict(X_test)
# KNN(K Nearest Neighbour)
from sklearn.neighbors import KNeighborsClassifier
classifier3=KNeighborsClassifier(algorithm=)
classifier3.fit(X_train,y_train)
y_pred3=classifier3.predict(X_test)
# Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier4=RandomForestClassifier()
classifier4.fit(X_train,y_train)
y_pred4=classifier4.predict(X_test)
# XGBOOST
from xgboost import XGBClassifier
classifier5= XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.000001, max_delta_step=0, max_depth=17,
              min_child_weight=1,monotone_constraints='()',
              n_estimators=225, n_jobs=0, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None) 
classifier5.fit(X_train, y_train)
y_pred5=classifier5.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred2)
print(cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred2)
print(ac)
  

bias = classifier2.score(X_train, y_train)
bias
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier2, X = X_train, y = y_train, cv = 5)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))