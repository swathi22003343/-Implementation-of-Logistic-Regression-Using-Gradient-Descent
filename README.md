# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1: Start

step 2: Import Libraries: Import the necessary libraries - pandas, numpy, and matplotlib.pyplot.

step 3: Load Dataset: Load the dataset using pd.read_csv.

step 4: Remove irrelevant columns (sl_no, salary).

step 5: Convert categorical variables to numerical using cat.codes.

step 6: Separate features (X) and target variable (Y).

step 7: Define Sigmoid Function

step 8: Define the loss function for logistic regression.

step 9: Define Gradient Descent Function: Implement the gradient descent algorithm to optimize the parameters.

step 10: Training Model: Initialize theta with random values, then perform gradient descent to minimize the loss and obtain the optimal parameters.

step 11: Define Prediction Function: Implement a function to predict the output based on the learned parameters.

step 12: Evaluate Accuracy: Calculate the accuracy of the model on the training data.

step 13: Predict placement status for a new student with given feature values (xnew).

step 14: Print Results: Print the predictions and the actual values (Y) for comparison.

step 15: Stop. 

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SWATHI D
RegisterNumber: 212222230154

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("/content/Placement_Data.csv")
data
data= data.drop('sl_no',axis=1)
data= data.drop('salary',axis=1)

data["gender"]=data["gender"].astype('category')
data["ssc_b"]=data["ssc_b"].astype('category')
data["hsc_b"]=data["hsc_b"].astype('category')
data["degree_t"]=data["degree_t"].astype('category')
data["workex"]=data["workex"].astype('category')
data["specialisation"]=data["specialisation"].astype('category')
data["status"]=data["status"].astype('category')
data["hsc_s"]=data["hsc_s"].astype('category')
data.dtypes
# labelling the columns
data["gender"]=data["gender"].cat.codes
data["ssc_b"]=data["ssc_b"].cat.codes
data["hsc_b"]=data["hsc_b"].cat.codes
data["degree_t"]=data["degree_t"].cat.codes
data["workex"]=data["workex"].cat.codes
data["specialisation"]=data["specialisation"].cat.codes
data["status"]=data["status"].cat.codes
data["hsc_s"]=data["hsc_s"].cat.codes
data
X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values
Y
#initialize the mdel parameters.

theta= np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
  return 1/(1+ np.exp(-z))
def loss(theta, X,y):
  h=sigmoid(X.dot(theta))
  return -np.sum(y*np.log(h) + (1-y)* np.log(1-h))
def gradient_descent (theta, X,y,alpha, num_iterations):
  m=len(y)
  for i in range(num_iterations):
    h=sigmoid(X.dot(theta))
    gradient=X.T.dot(h-y)/m
    theta -= alpha*gradient
  return theta
theta= gradient_descent(theta, X,y ,alpha=0.01, num_iterations=1000)
def predict(theta, X):
  h= sigmoid(X.dot(theta))
  y_pred = np.where(h>=0.5,1,0)
  return y_pred
y_pred = predict(theta, X)

accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)

print(y_pred)
print(Y)
xnew = np.array([[ 0, 87, 0, 95, 0, 2, 78, 2, 0, 0 ,1, 0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
xnew = np.array([[ 0, 0, 0, 0, 0, 2, 8, 2, 0, 0 ,1, 0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```

## Output:
### dataset:
![ml ex 5-1](https://github.com/Gopika-9266/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122762773/c463a180-386a-4ca9-bcc4-c2b33c062a8d)

### dataset.types:
![ml exp 5-2](https://github.com/Gopika-9266/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122762773/2fcd8d44-53be-46fa-a81b-9f09b0dc6481)

### dataset:
![ml exp 5-3](https://github.com/Gopika-9266/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122762773/172d3bd8-1ca1-4333-b073-089190e93f93)

### Y:
![Screenshot 2024-05-07 151208](https://github.com/Gopika-9266/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122762773/c2e515aa-1a34-44c7-97bf-0d06dc055d0b)

### Accuracy: 
![ml exp 5-4](https://github.com/Gopika-9266/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122762773/34f5b41b-7afe-4b2e-9e4e-cf0bc20a590e)

### y_pred:
![ml exp 5-6](https://github.com/Gopika-9266/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122762773/1f305081-21ff-41ba-8ca1-a65ea9e91028)

### Y:
![ml exp 5-5](https://github.com/Gopika-9266/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122762773/654de1de-37fe-4551-aa51-1606c0de00b5)

### y_prednew:
![ml exp 5-7](https://github.com/Gopika-9266/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122762773/d8e28e9b-c0ed-4742-9339-8c0a08496ade)
![ml exp 5-8](https://github.com/Gopika-9266/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122762773/e07c4fca-a130-421d-8cc1-d9f0178e816b)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

