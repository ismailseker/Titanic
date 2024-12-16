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

# dtypes: float64(2), int64(5), object(5)

# float: Fare and Age
# int: pclass,sibsp,parch,passngerId and survived
# object: cabin, embarked,ticket, name and sex


# %% Univariate Variable Anaylsis

# Categorical: Survived,Sex,Embarked,Pclass,Name,Ticket,Cabin,Sibsp and parch
# Numerical: Fare,Age,passngerId

def bar_plot(variable):
    
    var = train_df[variable]
    varValue = var.value_counts()
        
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()    
    print("{}: \n {}".format(variable,varValue))

category1 = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]

for c in category1:
    bar_plot(c)


category2 = ["Name","Ticket","Cabin"]

for c in category2:
    print("{} \n".format(train_df[c].value_counts()))
    
def hist_plot(variable):
    plt.figure(figsize=(9,3))
    plt.hist(train_df[variable])
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distrubution histogram".format(variable))
    plt.show()
    
numericVar = ["Fare","Age","PassengerId"]

for n in numericVar:
    hist_plot(n)

# %% Basic Data Analysis
     
# Pclass - survived
# Sex - survived
# SibSp - survived
# Parch - survived

# PRINT AND CHECK RESULT

# Pclass VS Survived Analysis
train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index = False).mean().sort_values(by="Survived",ascending = False)

# Sex VS Survived Analysis
train_df[["Sex","Survived"]].groupby(["Sex"],as_index = False).mean().sort_values(by="Survived",ascending = False)

# SibSp VS Survived Analysis
train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index = False).mean().sort_values(by="Survived",ascending = False)

# Parch VS Survived Analysis
train_df[["Parch","Survived"]].groupby(["Parch"],as_index = False).mean().sort_values(by="Survived",ascending = False)

    



