import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import sklearn as sklearn

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('./iris.csv')

print(df.describe())
print(df.dtypes)