#Loading Data
import pandas as pd
df = pd.read_csv('/content/exoplaent_dataset.csv')

#Preparing Data

#Separating data as input label(x) and output label(y)

y = df['Habitability']
x = df.drop('Habitability', axis = 1)

#Splitting data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state= 0)

#Building Model

**Training Model(Linear Regression)**

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

#Predicting using Model

y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

#Evaluating Model

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

result = pd.DataFrame({
    'Train MSE': [train_mse],
    'Train R2': [train_r2],
    'Test MSE': [test_mse],
    'Test R2': [test_r2]
})
result
