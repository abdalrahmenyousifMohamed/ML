import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("Tokyo 2021 dataset.csv")
print(data)
data.isna().sum()
#Extracting Independent and dependent Variable  

x=data.iloc[:,data.columns!='Rank_by_Total']
y=data.iloc[:,data.columns=='Rank_by_Total']
print(x)
print(y)
x_train, x_test,y_train ,y_test=train_test_split(x,y,test_size=.2)
#2
sc=StandardScaler()
x_train_scale=sc.fit_transform(x_train)
x_train_scale
x_test_scale=sc.transform(x_test)
# # plt.plot(x_test_scale,color="red");

# # plt.plot(x_train_scale,color="blue")
# # plt.show()
z=sc.inverse_transform(x_test_scale)
#3
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train_scale,y_train.values.ravel()) #DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().return self._fit(X, y)
KNeighborsClassifier()
y_predict=model.predict(x_test_scale)
print(y_predict)
import sklearn.metrics as mc
dir(mc)
#####2
from sklearn.preprocessing import MinMaxScaler
MNS=MinMaxScaler()
x_train_MN=MNS.fit_transform(x_train)
print(x_train_MN)
from sklearn.preprocessing import RobustScaler
rs=RobustScaler()
x_rs=rs.fit_transform(x_train)
print(x_rs)
print(mc.accuracy_score(y_test,y_predict))
