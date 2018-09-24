import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import sklearn as sklearn

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('./iris.csv')

print(df.describe())
print(df.dtypes)
# df['petal_width'].plot.hist()
# plt.show()

# Divide into 2 sets -- training and testing
all_inputs = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values 
all_classes = df['species'].values
(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=1)

# Classifier into different categories
classifier = DecisionTreeClassifier()
classifier.fit(train_inputs, train_classes)

# get accuracy of decision tree
print('Accuracy:', classifier.score(test_inputs, test_classes))