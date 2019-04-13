import pandas
import numpy as np

# for Learning
from sklearn.linear_model import LogisticRegression

# for Feature Engineering
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# for visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


train_data_X = pandas.read_csv('train.csv', delimiter=',')
test_data_X  = pandas.read_csv('test.csv', delimiter=',')

test_data_X  = test_data_X.drop(columns=['Cabin', 'Ticket', 'PassengerId', 'Name'])
test_data_Y  = 0
train_data_Y = train_data_X['Survived']
train_data_X = train_data_X.drop(columns=['Cabin', 'Ticket', 'PassengerId', 'Name', 'Survived'])



important_features = ['Embarked', 'Sex', 'Pclass']

'''
    fillMissing: fill columns containing missing.
    Base on: Embarked, Sex, Pclass
    1. Filter rows whose features have the same value with the dealing with one   
'''


def quantization(data):

    for index, row in data.iterrows():
        # SibSp, Parch
        if row['SibSp'] > 1:
            data.at[row.name, 'SibSp'] = 2
        if row['Parch'] > 1:
            data.at[row.name, 'Parch'] = 2

        # Embarked
        if row['Embarked'] == 'S':
            data.at[row.name, 'Embarked'] = 0
        elif row['Embarked'] == 'C':
            data.at[row.name, 'Embarked'] = 1
        else:
            data.at[row.name, 'Embarked'] = 2

        # Sex
        if row['Sex'] == 'male':
            data.at[row.name, 'Sex'] = 1
        else:
            data.at[row.name, 'Sex'] = 2


def fill_missing(data, feature):
    null_data = data[data[feature].isnull()]

    for index, row in null_data.iterrows():
        nulldata_Embarked = row['Embarked']
        nulldata_Sex      = row['Sex']
        nulldata_Pclass   = row['Pclass']

        series_null = 1
        for im_feature, nulldata_value in zip(important_features, [nulldata_Embarked, nulldata_Sex, nulldata_Pclass]):
            if im_feature != feature:
                series_null = series_null & (data[im_feature] == nulldata_value)

        filtered_column = (data.loc[series_null])[feature]
        data.at[row.name, feature] = filtered_column.mean()


quantization(data=train_data_X)
fill_missing(train_data_X, 'Fare')
fill_missing(train_data_X, 'Age')

quantization(test_data_X)
fill_missing(test_data_X, 'Fare')
fill_missing(test_data_X, 'Age')

scaler = MinMaxScaler()
scaler.fit(train_data_X)

train_data_Xtransformed = scaler.transform(train_data_X)
test_data_Xtransformed = scaler.transform(test_data_X)

kfold = KFold(n_splits=5, shuffle=True, random_state=0)

sns.catplot(y="Age", x="Pclass", data=train_data_Xtransformed, hue="Sex", kind="violin")
plt.show()
