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
![image](https://github.com/Augustine0306/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404460/9ac2175d-9770-41a6-ab7d-b1b1d90ebfa1)
![image](https://github.com/Augustine0306/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404460/e94d11d5-c9e8-4286-895a-d41067146253)
![image](https://github.com/Augustine0306/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404460/d3350966-a5a6-433c-9b57-a125ebe93e43)
![image](https://github.com/Augustine0306/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404460/8ce9820d-de28-4066-b469-a53cc1aa0200)
![image](https://github.com/Augustine0306/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404460/b98cfbed-41db-451a-b418-0f0b13208ae2)
![image](https://github.com/Augustine0306/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119404460/799b0a64-f77d-4919-91ae-201873c685ad)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
