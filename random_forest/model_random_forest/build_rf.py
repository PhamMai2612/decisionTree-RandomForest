
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr

data = pd.read_csv("TitanicPreprocessed.csv")
Y = data.values[:, -1]
X = data.values[:, 0:-1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)

print(rf.score(X=X_test, y=y_test))


