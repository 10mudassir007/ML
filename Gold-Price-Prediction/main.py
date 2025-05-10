import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

df = pd.read_csv("Gold Price (2013-2023).csv")

df['Date'] = pd.to_datetime(df['Date'])

df = df.drop([df.columns[-2],df.columns[-1]],axis=1)

print(df.isnull().sum())

df.sort_values('Date',ascending=True,inplace=True)
df.reset_index(drop=True,inplace=True)

print(df.info())
print(df.describe())

df['Date'] = df['Date'].replace({",":""},regex=True)
df

features = df.drop(df.columns[0],axis=1)
def clean_and_convert(column):
    return pd.to_numeric(column.str.replace(',', '', regex=True), errors='coerce')
features = features.apply(clean_and_convert)


plt.plot(df['Date'], df['Price'])



X = features.drop(features[['Price']],axis=1).to_numpy()
y = np.array(features[['Price']])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15)

model = LinearRegression()
model.fit(X_train,y_train)

preds = model.predict(X_test)

print(model.score(X_test,y_test))
print(r2_score(y_test,preds))
print(mean_absolute_error(y_test,preds))
print(mean_squared_error(y_test,preds))