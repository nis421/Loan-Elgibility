# Loan-Elgibility
The Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers. It’s a classification problem , given information about the application we have to predict whether the they’ll be to pay the loan or not. We’ll start by exploratory data analysis , then preprocessing , and finally we’ll be testing different models such as Logistic regression and decision trees.
# 
The total code is in the python language ...
#Code...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
dataset=pd.read_csv("train.csv.xls")
dataset.head()
dataset.shape
dataset.info()
dataset.describe()
pd.crosstab(dataset['Credit_History'],dataset['Loan_Status'],margins=True)
pd.crosstab(dataset['Dependents'],dataset['Loan_Status'],margins=True)
dataset.boxplot(column='ApplicantIncome')
dataset['ApplicantIncome'].hist(bins=20)
dataset['CoapplicantIncome'].hist(bins=10)
dataset['CoapplicantIncome'].hist(bins=20)
dataset.boxplot(column='ApplicantIncome',by='Education')
dataset['LoanAmount'].hist(bins=20)
dataset['LoanAmount_log']=np.log(dataset['LoanAmount'])
dataset['LoanAmount_log'].hist(bins=20)
dataset.isnull().sum()
dataset['Gender'].fillna(dataset['Gender'].mode()[0],inplace=True)
dataset['Married'].fillna(dataset['Married'].mode()[0],inplace=True)
ataset['Dependents'].fillna(dataset['Dependents'].mode()[0],inplace=True)
dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0],inplace=True)
dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean(),inplace=True)
dataset['LoanAmount_log'].fillna(dataset['LoanAmount_log'].mean(),inplace=True)
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0],inplace=True)
dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0],inplace=True)
dataset.isnull().sum()
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0],inplace=True)
dataset['TotalIncome']=dataset['ApplicantIncome']+dataset['CoapplicantIncome']
dataset['TotalIncome_log']=np.log(dataset['TotalIncome'])
x=dataset.iloc[:,np.r_[1:5,9:11,13:15]].values
y=dataset.iloc[:,12].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()
for i in range(0,5):
    x_train[:,i]=labelencoder_x.fit_transform(x_train[:,i])
x_train[:,7]=labelencoder_x.fit_transform(x_train[:,7])
labelencoder_y=LabelEncoder()
y_train=labelencoder_y.fit_transform(y_train)
for i in range(0,5):
    x_test[:,i]=labelencoder_x.fit_transform(x_test[:,i])
x_test[:,7]=labelencoder_x.fit_transform(x_test[:,7])
labelencoder_y=LabelEncoder()
y_test=labelencoder_y.fit_transform(y_test)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)
from sklearn.tree import DecisionTreeClassifier
DTClassifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
DTClassifier.fit(x_train,y_train)
y_pred=DTClassifier.predict(x_test)
y_pred
from sklearn import metrics
print('the accuracy of decision tree is :',metrics.accuracy_score(y_pred,y_test))
from sklearn.naive_bayes import GaussianNB
NBClassifier=GaussianNB()
NBClassifier.fit(x_train,y_train)
y_pred=NBClassifier.predict(x_test)
y_pred
from sklearn import metrics
print('The Accuracy of Naive bayes is :',metrics.accuracy_score(y_pred,y_test))
dataset=pd.read_csv("test.csv.xls")
dataset['Gender'].fillna(dataset['Gender'].mode()[0],inplace=True)
dataset['Dependents'].fillna(dataset['Dependents'].mode()[0],inplace=True)
dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0],inplace=True)
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0],inplace=True)
dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0],inplace=True)
dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean(),inplace=True)
dataset['LoanAmount_log']=np.log(dataset['LoanAmount'])
dataset['TotalIncome']=dataset['ApplicantIncome']+dataset['CoapplicantIncome']
dataset['TotalIncome_log']=np.log(dataset['TotalIncome'])
test=dataset.iloc[:,np.r_[1:5,9:11,13:15]].values
for i in range(0,5):
    test[:,i]=labelencoder_x.fit_transform(test[:,i])
test[:,7]=labelencoder_x.fit_transform(test[:,7])
test=ss.fit_transform(test)
pred=NBClassifier.predict(test)
pred
