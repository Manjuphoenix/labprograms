import csv
import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB

from sklearn import metrics
from sklearn.model_selection import train_test_split

data = pd.read_csv('diabetes_csv.csv')
print(data.head)

x_train = np.array(data.iloc[:-100,0:-1])
y_train = np.array(data.iloc[:-100,-1])
x_test = np.array(data.iloc[-100:,0:-1])
y_test = np.array(data.iloc[-100:,-1])

#print(data.head())

model = GaussianNB()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Accuracy of classifier is:",metrics.accuracy_score(y_test, y_pred)*100,"%")
