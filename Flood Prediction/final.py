#Flood prediction in rivers using Explainable AI(XAI)

import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objs as go
import chart_studio.plotly as py
import plotly
import shap
import shap.plots
from datetime import datetime
import joblib
from sklearn.preprocessing import Normalizer,MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
import shap
import matplotlib.pyplot as plt

pd.options.mode.copy_on_write = True

filenames=['Godavari']#,'Cauvery','Krishna','Mahanadi','Son']

Total_pred=pd.DataFrame()

for filename in filenames:
    
    print("Loading the data for "+filename+" river...")

    data1=pd.read_excel("Dataset/"+filename+".xlsx")
    print(data1.head())
    
    for i in range(1,len(data1.columns)):
	    data1[data1.columns[i]] = data1[data1.columns[i]].fillna(data1[data1.columns[i]].mean())
    
    y=data1['Flood']
    
    for i in range(len(y)):
        if(y[i] >= 0.1):
            y[i]=1
            
    d1=pd.DataFrame()
    d1["Day"]=data1['Date']
    d1['Months']=data1['Date']
    d1['Year']=data1['Date']
    data1['Date']=pd.to_datetime(data1['Date'])
    d1["Year"]=data1.Date.dt.year
    d1["Months"]=data1.Date.dt.month
    d1["Day"]=data1.Date.dt.day
    print(d1.head())
    data1.drop('Flood',axis=1,inplace=True)
    
    data1.drop('Date',inplace=True,axis=1)
    data1=pd.concat([d1,data1],axis=1)
    locate=0
    for i in range(len(data1["Day"])):
        if(data1["Day"][i]==31 and data1["Months"][i]==12 and data1["Year"][i]==2015):
            locate=i
            break
    i=locate+1
    
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(data1, y, train_size=0.8, random_state=42)

    x_train.drop(labels=['Day','Months','Year'],inplace=True,axis=1)
    x_test.drop(labels=['Day','Months','Year'],inplace=True,axis=1)


    #Model Training

    from sklearn.linear_model import  LogisticRegression
    #Rf =LogisticRegression(max_iter=10000)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV

    # Define the parameter grid
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # Create a Random Forest Classifier
    Rf = RandomForestRegressor(n_estimators=100,max_depth=20,warm_start=True)

    # Perform Grid Search
    grid_search = GridSearchCV(estimator=Rf, param_grid=param_grid, cv=5)
    grid_search.fit(x_train, y_train)

    #Train test split
    from sklearn.model_selection import train_test_split 
    x_train, x_test, y_train, y_test = train_test_split(data1, y, train_size=0.8, random_state=42)

    #RF classifier
    from sklearn.ensemble import RandomForestClassifier
    
    Rf = RandomForestClassifier(n_estimators=100,max_depth=20,warm_start=True)
    Rf.fit(x_train, y_train)
    y_predict=Rf.predict(x_test)

    #Model evaluation

    
    from sklearn.metrics import confusion_matrix,mean_absolute_error
    from sklearn.metrics import classification_report

    print("\n"+filename+".xlsx dataset evaluation")
    print("1.Train data accuracy =",Rf.score(x_train,y_train))
    print("2.Test data accuracy =",Rf.score(x_test,y_test))
    
    print("3.Classification report:\n",classification_report(y_test, y_predict))
    
    mae=mean_absolute_error(y_test, y_predict)
    print("4.Mean Absolute Error =",mae)
    
    from sklearn.metrics import confusion_matrix
    print("5.Confusion Matrix:\n",confusion_matrix(y_test,y_predict))
    
    Total_pred[filename]=y_predict
    # load JS visualization code to notebook
    shap.initjs()

    # Create the explainer 
    explainer = shap.TreeExplainer(Rf)
    shap_values = explainer.shap_values(x_test)

    shap.summary_plot(shap_values,show=True, feature_names=x_test.columns,class_names=['Low','High'])
    plt.gcf().set_size_inches(6,12)
    
    plt.tight_layout()
    print(f'New size: {plt.gcf().get_size_inches()}')
    print("Variable Importance Plot - Global Interpretation")


print(Total_pred)
Total_pred.to_csv("Output.csv")
