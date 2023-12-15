import pandas as pd

salaries = pd.read_csv('Salary Data.csv')

df = salaries.copy().dropna()
target = 'Salary'
encode = ['Education Level']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix = col)
    df = pd.concat([df,dummy], axis = 1)
    del df[col]

X = df.drop(columns = ["Gender", "Job Title", "Salary"])
y = df['Salary']

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)

import pickle
pickle.dump(lr, open('salaries_lr.pkl', 'wb'))
