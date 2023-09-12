# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and read the dataframe.

2.Assign hours to X and scores to Y.

3.Implement training set and test set of the dataframe

4.Plot the required graph both for test data and training data.

5.Find the values of MSE , MAE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: AUGUSTINE J
RegisterNumber:  212222240015

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)


#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```
## Output:

df.head()

![ML21](https://github.com/Augustine0306/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404460/986bc3dc-933b-44de-b395-6b6fdabc1d85)

df.tail()

![ML22](https://github.com/Augustine0306/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404460/50d78701-bb39-4192-8f5e-62030b4b311d)

Array value of X

![ML23](https://github.com/Augustine0306/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404460/28e8da05-2fa7-48e9-a5be-1cbd76c65c0e)

Array value of Y

![ML24](https://github.com/Augustine0306/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404460/85a72245-69dd-4cec-8318-6e9b9dbcad68)

Values of Y prediction

![ML25](https://github.com/Augustine0306/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404460/5e9d4d92-0e3c-45f0-ba7d-cdbd34522dcc)

Array values of Y test

![ML26](https://github.com/Augustine0306/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404460/b97748d7-0d30-4dc3-906c-73fb1543f474)

Training Set Graph

![ML27](https://github.com/Augustine0306/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404460/b0902126-84c1-4a44-a3ba-a1058327ae0a)

Test Set Graph

![ML28](https://github.com/Augustine0306/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404460/e8fb8366-048e-4537-bc82-73589187b9a0)

Values of MSE, MAE and RMSE

![ML29](https://github.com/Augustine0306/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404460/66d3990d-da8d-4f93-88f5-49f9f735a6c7)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
