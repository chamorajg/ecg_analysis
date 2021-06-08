import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = csv.DictReader(open("all-samples-4types.csv"))
names = ['Amplitude','RR','Arritmia']
data = pd.read_csv('all-samples-4types.csv',names=names)
#print(data)
X_train = data.iloc[:,0:2]
y_train = data.iloc[:,2]
for j in range(len(y_train)) :
    if y_train[j] == '(N' :
        data.iloc[j,2] = 0
    elif y_train[j] == '(B':
        data.iloc[j,2] = 1
    elif y_train[j] == '(T':
        data.iloc[j,2] = 2
    elif y_train[j] == '(VT':
        data.iloc[j,2] = 3
y_train = data.iloc[:,2]
#print(y_train)
X_test = [[1.10,0.673]]
print('The input values of patient heart rate')
print('Amplitue', X_test[0][0])
print('RR interval', X_test[0][1])
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
if y_pred == 0 :
    print('The patient has a Normal Sinus Heart Rate')
elif y_pred == 1 :
    print('The condition of the patient heart is a BradyCardia')
elif y_pred == 2 :
    print('The condition of the patient heart is a Tachycardia')
elif y_pred == 3 :
    print('The condition of the patient heart is Ventricular Tachycardia')

from matplotlib.colors import ListedColormap
X, y = X_train, y_train
y = np.array(y)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max - x_min)/1000
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
Z = classifier.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Amplitude')
plt.ylabel('RR-Interval')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()