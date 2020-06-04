# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 19:39:09 2020

@author: USER
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

iris=load_iris()

print(dir(iris))
print(iris['data'])

iris1=iris.data
#iris=pd.DataFrame(iris.data,columns=iris.feature_names)

print(iris.target)
print(iris.feature_names)
#print(iris.feature_names)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(iris1,iris.target,test_size=0.1)



models=LogisticRegression()

models.fit(X_train,Y_train)

print(models.predict([[6.7,3.0,5.2,2.3]]))

print(models.score(X_train,Y_train))

import pickle
pickle.dump(models,open('model_iris.pkl','wb'))

mods=pickle.load(open('model_iris.pkl','rb'))


print(mods.predict([[5.1,3.5,1.4,0.2]]))
print(mods.predict([[5.9,3.0,5.1,1.8]]))
print(mods.predict([[6. , 2.7, 5.1 ,1.6]]))
print(mods.predict([[5.6, 3. , 4.1 ,1.3]]))



