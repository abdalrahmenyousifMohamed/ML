import pandas as pd
import numpy as np                          
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# check if data is empty
# data = sns.load_dataset('iris')
# if data.empty:data.interpolate()
iris_data = datasets.load_iris()
print(iris_data)
# use petal length and width only for analysis
X = iris_data.data[:,[2,3]]
y = iris_data.target
# Input the iris data into the pandas dataframe
iris_dataframe = pd.DataFrame(iris_data.data[:, [2, 3]], # values
                  columns=iris_data.feature_names[2:]) # ['petal length (cm)', 'petal width (cm)']
# View the first 5 rows of the data
print(iris_dataframe.head())
# Print the unique labels of the dataset
print('\n' + 'Unique Labels contained in this data are '
     + str(np.unique(y)))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#print(y_test)
print('The training set contains {} samples and the test set contains {} samples'.format(X_train.shape[0], X_test.shape[0]))
# visualize our data.
markers = ['x', '*' , 'o']
colors = ('red', 'blue', 'green')
print("unique y_test is" , (np.unique(y_test)))
color_map = ListedColormap(colors[:len(np.unique(y_test))])
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
           c=color_map(idx), marker=markers[idx], label=cl)
    #print("inedx ${idx} is " , X[y == cl, 0] , "l2\t", X[y == cl, 1])
plt.show()
standard_scaler = StandardScaler()
#then we will perform scaling on our data. 
standard_scaler.fit(X_train)
X_train_standard = standard_scaler.fit_transform(X_train)
X_test_standard = standard_scaler.transform(X_test)
print('The first five rows after standardisation look like this:\n')
print(pd.DataFrame(X_train_standard, columns=iris_dataframe.columns).head())
SVM = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0) #C ? = it's $1/\lambda$, where $\lambda$ is the regularization strength.
SVM.fit(X_train_standard, y_train)
y_svm_pred = SVM.predict(X_test)
print(y_svm_pred)
cn = confusion_matrix(y_test,y_svm_pred)
print("confusion_matrix for ( y_test , y_svm_pred) is\n {}".format(cn))
print('Accuracy of our SVM model on the training data is {:.2f} out of 1'.format(SVM.score(X_train_standard, y_train)))
print('Accuracy of our SVM model on the test data is {:.2f} out of 1'.format(SVM.score(X_test_standard, y_test)))
import warnings
def versiontuple(version):
   return tuple(map(int, (version.split("."))))
def decision_plot(X, y, classifier, test_idx=None, resolution=0.02):
   # setup marker generator and color map
   markers = ('s', 'x', 'o', '^', 'v')
   colors = ('red', 'blue', 'green', 'gray', 'cyan')
   cmap = ListedColormap(colors[:len(np.unique(y))])
   # plot the decision surface
   # create a mesh to plot in
   x1min, x1max = X[:, 0].min() - 1, X[:, 0].max() + 1
   x2min, x2max = X[:, 1].min() - 1, X[:, 1].max() + 1
   xx1, xx2 = np.meshgrid(np.arange(x1min, x1max, resolution),
                 np.arange(x2min, x2max, resolution))
   Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
   Z = Z.reshape(xx1.shape)
   plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
   plt.xlim(xx1.min(), xx1.max())
   plt.ylim(xx2.min(), xx2.max())
   for idx, cl in enumerate(np.unique(y)):
      plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
              alpha=0.8, c=cmap(idx),
              marker=markers[idx], label=cl)

decision_plot(X_test_standard, y_test, SVM)
plt.show()
