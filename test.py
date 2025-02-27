import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random
import math
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Dataset/Corrupted1_20_large.csv',nrows=1000)
dataset.fillna(0, inplace = True)
mean=dataset['Mystery_Data_Y'].mean(axis=0)
range=mean/4
print(range)
dataset['Adamic_Adar_index']=0

y_test = dataset['Modified']

corrupted = []

for i in dataset['Mystery_Data_Y']:
  first=1
  second=1
  third=1
  for j in dataset['Mystery_Data_Y']:
    if(abs(j-i)<=range):
      first=first+1
    if(abs(j-i)<=2*range):
      second=second+1
    if(abs(j-i)<=3*range):
      third=third+1
  index=0
  if(first!=1):
    index=(1/math.log(first))
  if(second!=1):
    index=index+(1/math.log(second))
  if(third!=1):
    index=index+(1/math.log(third))
  corrupted.append(index)

print(type(corrupted))

pred = []
for value in corrupted:
    if value > 0.70:
        pred.append(True)
    else:
        pred.append(False)

        
acc = accuracy_score(y_test, pred)
print(acc)


X = dataset['Mystery_Data_Y'].ravel().astype(np.float64)
X = X.reshape(-1, 1)
print(X.shape)


X_train, X_test, y_train, y_tests = train_test_split(X, pred, test_size=0.2) #split data into train & test
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
acc = accuracy_score(y_tests, pred)
print("rf "+str(acc))


lof = LocalOutlierFactor(n_neighbors=5, novelty=True)
lof.fit(X)

# Predict the outlier scores
y_pred = lof.predict(X)

pred = []
for value in y_pred:
    if value == 1:
        pred.append(False)
    else:
        pred.append(True)

acc = accuracy_score(y_test, pred)
print(acc)        


iso = IsolationForest(n_estimators=1)
iso.fit(X)

# Predict the outlier scores
y_pred = iso.predict(X)

pred = []
for value in y_pred:
    if value == 1:
        pred.append(False)
    else:
        pred.append(True)

acc = accuracy_score(y_test, pred)
print(acc)


ocs = OneClassSVM()
ocs.fit(X)

# Predict the outlier scores
y_pred = ocs.predict(X)

pred = []
for value in y_pred:
    if value == 1:
        pred.append(False)
    else:
        pred.append(True)

acc = accuracy_score(y_test, pred)
print(acc)









