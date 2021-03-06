import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as mc #ERRORS	
from sklearn.linear_model import LinearRegression
data=pd.read_csv("Tokyo 2021 dataset.csv")
print(data.info)
print(data)
print(data.isnull().sum())
x_test=data.loc[21:,["Rank"]]
y_des=data.loc[21:,["GoldMedal"]]
x=data.loc[:22,["Rank"]]
y=data.loc[:22,["GoldMedal"]]
print(y)
model=LinearRegression()
model.fit(x,y)
y_predicted=model.predict(x_test)
print(y_predicted)
print(model.intercept_) # best val (a) from y=a+bx linear
print(model.coef_) # slope (b)  from y=a+bx linear
print("absloute error is ",mc.mean_absolute_error(y_des,y_predicted)) #the difference and add them and mean/nof
print("squared error is ",mc.mean_squared_error(y_des,y_predicted)) # the difference and square them / nof
plt.scatter(x_test,y_des)
plt.show()