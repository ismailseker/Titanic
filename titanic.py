import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


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


# %% Outlier Detection

def detectOutlier(df,features):
    outlier_indices = []
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c], 25)
        # 3rd quartile
        Q3 = np.percentile(df[c], 75)
        # IQR
        IQR = Q3 - Q1
        # Outlier Step
        outlier_step = IQR * 1.5
        # Detect outlier and their indeces
        outlier_list_cols = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # Store indices
        outlier_indices.extend(outlier_list_cols)
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list (i for i,v in outlier_indices.items() if v > 2)
    print(Counter(outlier_indices))
    
    return multiple_outliers

train_df.loc[detectOutlier(train_df,["Age","SibSp","Parch","Fare"])]

# Drop Outlier
train_df = train_df.drop(detectOutlier(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)

# %% Missing Value

train_df_len = len(train_df)
train_df = pd.concat([train_df,test_df],axis=0).reset_index(drop=True)
# Find Missing Value
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()

# Embark has 2 missing value, Fare has 1

# Fill Missing Value

train_df[train_df["Embarked"].isnull()]
train_df.boxplot(column="Fare", by = "Embarked")
plt.show()

train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]

train_df[train_df["Fare"].isnull()]
np.mean(train_df[train_df["Pclass"] == 3]["Fare"])
train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))
    
# %% Visulazation

# Correlation between SibSp -- Parch -- Age -- Fare -- Survived

list1=["SibSp","Parch","Age","Fare","Survived"]
sns.heatmap(train_df[list1].corr(), annot=True, fmt =".2f")
plt.show()

# Result : Fare features seems to have correlation with survived feature (0.26).

# SipSp -- Survived

g = sns.catplot(x = "SibSp",y = "Survived",data = train_df,kind = "bar",height = 6)
g.set_ylabels("Survived Probability")
plt.show()

# Having a lotf of SibSp causes less chance to survive.
# If SibSp = 0 or 1 or 2, passenger has more chance to survive.
# We can consider a new feature describing categories

