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

#ForRandom forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators=3, criterion='entropy')
classifier_rf.fit(x_train,y_train)
y_pred = classifier_rf.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
