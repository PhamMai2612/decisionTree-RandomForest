
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#funtion import dataset

data = pd.read_csv("TitanicPreprocessed.csv")
Y = data.values[:, -1]
X = data.values[:, 0:-1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X=X_train, y=Y_train)
print(clf.score(X=X_test, y=Y_test))
