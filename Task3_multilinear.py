#Task 3
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sklearn
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
#Tokyo 2021 dataset.csv
data=pd.read_csv("Tokyo 2021 dataset.csv")
print(data)
print(data.info())
print(data)
print(data.isnull().sum())
#data.interpolate(inplace=True)
x=data[["BronzeMedal","SilverMedal","GoldMedal"]] # y=a+b1x1+b2x2 multilinear
y=data.Rank
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.4)
#print(y_test)
sns.pairplot(data,x_vars=["BronzeMedal","SilverMedal","GoldMedal"],y_vars=["Rank"],diag_kind=None,kind="scatter")
lr=sm.ols(formula='SilverMedal ~ BronzeMedal + Rank',data=data).fit() #["BronzeMedal","SilverMedal","GoldMedal"],y_vars=["Rank"],
y_sm_predict=lr.predict({"BronzeMedal":5.22,"Rank":3.22})
print(y_sm_predict)
print(lr.params)
print(lr.summary())
plt.show()
