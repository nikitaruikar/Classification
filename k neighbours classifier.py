import numpy as np
import pandas as pd
import seaborn as sns
dataset=pd.read_csv('D:\\cognitior\\Basics of data science\\dataset\\Social_Network_Ads.csv')

dataset1= pd.get_dummies(dataset, columns=['Gender'], drop_first=True)
x = dataset1.iloc[:,:-1].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)

y = dataset.iloc[:,-1:].values
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

#for k neighbours classifier
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors=5, 
                                      metric='minkowski', p=2)
classifier_knn.fit(x_train, y_train)
y_pred = classifier_knn.predict(x_test)
y_pred
confusion_matrix(y_test, y_pred)

