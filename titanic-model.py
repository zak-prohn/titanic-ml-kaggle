import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#import data into training, test from csv files
training = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

training['train_test'] = 1
test['train_test'] = 0
test['Survived'] = np.NaN
all_data = pd.concat([training, test])

#print(all_data.columns)
# Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'train_test'],
#      dtype='object')

# survival = Survived, 0 = no, 1 = yes
# pclass = Ticket Class 1, 2, 3
# sex = Sex
# age = Age in Years
# sibsp	= # of siblings / spouses aboard the Titanic	
# parch	= # of parents / children aboard the Titanic	
# ticket = Ticket Number
# fare = Passenger fare
# cabin = Cabin Number
# embarked = Port of Embarkation, C = Cherbourg, Q = Queenstown, S = Southhampton

#Explore if age, wealth, location have any correlation with survival in training
#training.info()
#print(training.describe())

