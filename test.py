import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.externals import joblib
import numpy as np

## Get the data
music_data = pd.read_csv('music.csv')
# print(music_data)
## Prepare input set X
X = music_data.drop(columns=['genre'])
# print(X)
## Prepare the output set Y
Y = music_data['genre']
# print(Y)
## Create the machine learning model
model = DecisionTreeClassifier()
## Provide the learning data to the model
# model.fit(X, Y)
## Make a prediction on new inputs
# predictions = model.predict([[21, 1], [20, 0], [32, 0]])
# print(predictions)

## Measure accuracy, precision, recall and f1 score of the model - Split dataset into training and testing
## Remember average = micro for multiclass problem
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
# model.fit(X_train, Y_train)
# test_predictions = model.predict(X_test)
# accuracy = accuracy_score(Y_test, test_predictions)
# recall = recall_score(Y_test, test_predictions, average='micro')
# precision = precision_score(Y_test, test_predictions, average='micro')
# f_measure = f1_score(Y_test, test_predictions, average='micro')
# print(accuracy)
# print(recall)
# print(precision)
# print(f_measure)

## Measure accuracy, precision, recall and f1 score of the model - Split dataset into 10 folds
kfold = KFold(n_splits=10, shuffle=True)
model = DecisionTreeClassifier()

accuracy = cross_val_score(model, X, Y, cv=kfold)
print('Accuracy', np.mean(accuracy), accuracy)
recall = cross_val_score(model, X, Y, cv=kfold, scoring='recall_micro')
print('Recall', np.mean(recall), recall)
precision = cross_val_score(model, X, Y, cv=kfold, scoring='precision_micro')
print('Precision', np.mean(precision), precision)
f1 = cross_val_score(model, X, Y, cv=kfold, scoring='f1_micro')
print('F1', np.mean(f1), f1)


## Saving the learned machine learning model to make predictions later
# save model
joblib.dump(model, 'learned-model.joblib')

# load model
trained_decision_tree_model = joblib.load('learned-model.joblib')
