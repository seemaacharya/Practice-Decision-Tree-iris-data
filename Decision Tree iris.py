# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 20:05:21 2021

@author: Soumya PC
"""
#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing datset
data = pd.read_csv('iris.csv')
data.head()
data['Species'].unique()
data.Species.value_counts()
colnames = list(data.columns)
predictors = colnames[:4]
target = colnames[4]
            
#train test split
from sklearn.model_selection import train_test_split
train,test = train_test_split(data, test_size = 0.20, random_state=0)

#using Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy')
model1 = model.fit(train[predictors], train[target])
pred = model.predict(test[predictors])
type(pred)
pd.Series(pred).value_counts()

#evaluation
pd.crosstab(test[target],pred)

#Trained data
temp = pd.Series(model.predict(train[predictors])).reset_index(drop=True)
#Test data
np.mean(pd.Series(train.Species).reset_index(drop=True) == pd.Series(model.predict(train[predictors])))

#Accuracy test
np.mean(pred==test.Species)

#plot
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(model1,filled=True)
tree.plot_tree
