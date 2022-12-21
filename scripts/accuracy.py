import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation

X_train = pd.read_csv("X_train.csv", index_col=0)
X_test = pd.read_csv("X_test.csv", index_col=0)
y_train = pd.read_csv("y_train.csv", index_col=0)
y_test = pd.read_csv("y_test.csv", index_col=0)


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)
acc = np.zeros(1)
acc[0] = metrics.accuracy_score(y_test, y_pred)
# print(y_pred)
# print("Accuracy:", acc[0])
np.savetxt("predict.txt", y_pred, delimiter=",", fmt='%s')
np.savetxt("Accuracy.txt", acc, delimiter=",")