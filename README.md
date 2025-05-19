# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..

Developed by:SURUTHI S

RegisterNumber: 212224220114
*/
```
```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```


## Output:

## Encoding:

![image](https://github.com/user-attachments/assets/559685dc-b1a3-48fb-aab9-8c4bb01a0526)

## Head():

![image](https://github.com/user-attachments/assets/ddecf221-9175-4c61-82f7-e78994ac20c0)

## Info():

![image](https://github.com/user-attachments/assets/1b2aab57-4d47-4780-878d-765ad4d1939b)

## isnull().sum():

![image](https://github.com/user-attachments/assets/3a289b67-0f20-466f-b26a-c421b981818e)

 ## Prediction of y:

![image](https://github.com/user-attachments/assets/af3fca54-a031-49b8-b7b7-a7b0da8e3ac8)

 ## Accuracy:

 ![image](https://github.com/user-attachments/assets/0d2f189f-5055-4491-b954-17839513482b)

 ## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
