import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

import seaborn as sns

import warnings 
warnings.filterwarnings("ignore")

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
test_PassengerId = test_df['PassengerId']

# train_df.columns
# train_df.head()
# train_df.describe()

# %% Variable Description

# Columns
# PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'

# PassengerId : each passenger has unique id
# Survived : passenger survive = 1 or died = 0
# Pclass : passenger class
# Name : passenger's name
# Sex : Gender Info
# Age : Age of passenger
# SibSp : number of siblings/spouses
# Parch : number of parents/childs
# Ticket : Ticket number
# Fare : Amount of monet on spent ticket
# Cabin : cabin category
# Embarked : port where passenger embarked (C = cherbourg,Q = queenstown, S = southempton)

train_df.info()