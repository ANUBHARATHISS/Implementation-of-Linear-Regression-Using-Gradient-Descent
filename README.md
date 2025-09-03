# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: ANUBHARATHI SS
RegisterNumber: 212223040017
*/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
  X=np.c_[np.ones(len(X1)),X1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)
    theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta


data=pd.read_csv("/content/50_Startups.csv")
data.head()


X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)


theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:
![linear regression using gradient descent](sam.png)

DATA INFORMATION:

<img width="986" height="366" alt="Screenshot 2025-09-03 112526" src="https://github.com/user-attachments/assets/badcea58-42df-4fc8-ae2c-819b9f7fadf7" />


VALUE OF X:

<img width="330" height="765" alt="Screenshot 2025-09-03 112607" src="https://github.com/user-attachments/assets/cd0f1a81-caf3-4efe-a0b8-e085accac74f" />

<img width="323" height="308" alt="Screenshot 2025-09-03 112635" src="https://github.com/user-attachments/assets/14db877b-a772-4515-a832-197025fb1bb7" />

VALUE OF X1_SCALED:

<img width="501" height="700" alt="Screenshot 2025-09-03 112653" src="https://github.com/user-attachments/assets/d50907f3-10dd-431f-a115-5d539bd12fa9" />

<img width="507" height="461" alt="Screenshot 2025-09-03 112701" src="https://github.com/user-attachments/assets/d803e4d0-a171-4f0b-ab8e-d3e0130b05ec" />

PREDICTED VALUE:

<img width="400" height="53" alt="Screenshot 2025-09-03 112713" src="https://github.com/user-attachments/assets/7ca7ca1a-171d-4a05-a7df-2b39552fe4e8" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
