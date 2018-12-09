import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  
from sklearn import tree
data = pd.read_csv("to1.csv")
#data = np.array(data)
y=data.iloc[:,17:18]

def label_enc(X_train, X_test) :
    le=LabelEncoder()
    # Iterating over all the common columns in train and test
    for col in X_test.columns.values:
       # Encoding only categorical variables
       if X_test[col].dtypes=='object':
       # Using whole data to form an exhaustive list of levels
           data=X_train[col].append(X_test[col])
           le.fit(data.values)
           X_train[col]=le.transform(X_train[col])
           X_test[col]=le.transform(X_test[col])
    X_train_scale=scale(X_train)
    X_test_scale=scale(X_test)
    return X_train,X_test

def S_V_M(X_train, X_test, y_train, y_test):
    svclassifier = SVC(kernel='linear')  
    svclassifier.fit(X_train, y_train) 
    
    y_pred = svclassifier.predict(X_test)  
     
    print(confusion_matrix(y_test,y_pred))  
    print(classification_report(y_test,y_pred)) 
    
def Logistics(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
     
    print(confusion_matrix(y_test,y_pred))  
    print(classification_report(y_test,y_pred))  
    
def Decision(X_train, X_test, y_train, y_test):
    model = tree.DecisionTreeClassifier(criterion='gini')
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
     
    print(confusion_matrix(y_test,y_pred))  
    print(classification_report(y_test,y_pred))    

for i in range(12,17):
    X=data.iloc[:,0:i+1]      
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
    X_train, X_test = label_enc(X_train, X_test)
    print('!-----SVM-----!')
    S_V_M(X_train, X_test, y_train, y_test)
    print('!-----REGRESSION-----!')
    Logistics(X_train, X_test, y_train, y_test)
    print('!-----DECISION TREE-----!')
    Decision(X_train, X_test, y_train, y_test)
    print('!--------------------------------END--------------------------------------!')
    
