# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:43:22 2021

@author: laure
"""

# Example from: https://pythonprogramming.net/machine-learning-python3-pandas-data-analysis/

#preprocess data
import pandas as pd

df = pd.read_csv("Datasets/diamonds.csv", index_col=0)
df.head()

cut_class_dict = {"Fair":1, "Good":2, "Very Good":3, "Premium": 4, "Ideal":5}
clarity_dict = {"I3": 1, "I2": 2, "I1": 3, "SI2": 4, "SI1": 5, "VS2": 6, "VS1": 7, "VVS2": 8, "VVS1": 9, "IF": 10, "FL": 11}
color_dict = {"J": 1,"I": 2,"H": 3,"G": 4,"F": 5,"E": 6,"D": 7}

df['cut'] = df['cut'].map(cut_class_dict)
df['clarity'] = df['clarity'].map(clarity_dict)
df['color'] = df['color'].map(color_dict)
df.head()

#lin reg
import sklearn
from sklearn.linear_model import SGDRegressor

df = sklearn.utils.shuffle(df)
# always shuffle your data to avoid any bias

# use .values to convert to numpy array
X = df.drop("price", axis=1).values
Y = df["price"].values

# save some for testing
test_size = 200

X_train = X[:-test_size]
Y_train = Y[:-test_size]

X_test = X[-test_size:]
Y_test = Y[-test_size:]


# train and test classifier!
#clf = SGDRegressor(max_iter=1000)
#clf.fit(X_train, Y_train)
#
#print(clf.score(X_test, Y_test))


# Now try support vector regression instead
from sklearn import svm
clf2 = svm.SVR()
clf2.fit(X_train, Y_train)
print(clf2.score(X_test,Y_test))

for X, Y in list(zip(X_test, Y_test))[:10]:
    print(clf.predict([X])[0], y)
    




