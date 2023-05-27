#!/usr/bin/env python
# coding: utf-8

# <center><h1> Predicting Customer Ad Clicks via Machine Learning </h1> </center>

# <center><h3> Sanoop Kammapata </h3> </center>

# ## Table of contents
# 1. [Introduction](#1.-Introduction)
# 2. [Data](#2.-Data)
# 3. [Develop Predictive Model](#3.-Develop-Predictive-Model)
# 4. [Conclusions](#4.-Conclusions)

# ## 1. Introduction <a name="1.-Introduction"></a>

# Objective of this project is to build a machine learning model that can detect potential users to convert or be interested in an ad.

# ### 1.1 Load required libraries 

# In[13]:


# libraries for loading the data from mySQL database
import pandas as pd
from sqlalchemy import create_engine
import pymysql
import configparser

# libraries for data cleaning and visualisation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
get_ipython().run_line_magic('matplotlib', 'inline')

# libraries for ML part
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix,accuracy_score


# ## 2. Data <a name="2.-Data"></a>

# The initial data sets (csv files) were loaded into mySQL database. After data cleaning and filtering (for one day) using mySQL the final data was inserted into a table called mldata for further processing with python.  

# ### 2.1 Connecting mySQL database and python using PyMySQL

# In[8]:


# Create engine

# uri based on mySQL database credentials
uri_direct = 'mysql+pymysql://root:password@22@localhost:3306/loyalty'

# use sqlalchemy to create a connection engine
engine_direct = create_engine(uri_direct)

# this connects to the sql engine
con = engine_direct.connect()


# ### 2.2 Load data from mySQL database

# In[9]:


# Load data
mldata = pd.read_sql('''
    SELECT * 
    FROM ctr.mldata;
''',con=con)
mldata.head()


# In[11]:


# close the connection between python and muSQL
con.close()


# The mldata from mySQL was saved (locally) as csv file so that it can be accessed further without any connection to the mySQL database.

# In[14]:


# Save data as csv file
mldata.to_csv('mldata.csv', index=False)


# In[6]:


# Load data from csv file
mldata = pd.read_csv('mldata.csv').drop(['Unnamed: 0'],axis=1)
mldata


# ### 2.3 Data cleaning and exploration

# In[20]:


mldata.info()


# In[21]:


mldata.describe()


# In[22]:


mldata.isnull().sum()


# In[109]:


mldata['clicked'].value_counts(normalize=True)


# In[137]:


mldata.groupby(['network', 'clicked'])['clicked'].count().unstack()


# In[113]:


mldata.groupby(['gender', 'clicked'])['clicked'].count().unstack()


# In[139]:


mldata.groupby(['ad_id_views', 'view_hour'])['view_hour'].count()


# ### 2.4 Exploratory Data Analysis

# In[106]:


plt.figure(figsize = (10, 5))
mldata['network'].value_counts().plot(kind='bar', color = ['green', 'blue', 'red','orange', 'brown', 'black'])
plt.title('Different network used', fontsize = 16)
plt.xlabel('Network', fontsize=10);
plt.ylabel('Frequency Count', fontsize=10);


# In[124]:


sns.countplot(data = mldata, x='network', hue='clicked')
plt.title('Clicked or unclicked based on networks', fontsize=15);

# 0 = No click
# 1 = click


# In[129]:


sns.countplot(data = mldata, x='kind_pay', hue='clicked');
plt.title('Kind pay and clicks', fontsize=15);


# In[130]:


sns.countplot(data = mldata, x='kind_card', hue='clicked')
plt.title('Kind card and clicks', fontsize=15);


# In[131]:


sns.countplot(data = mldata, x='industry', hue='clicked')
plt.xticks(rotation=90)
plt.title('Industry and clicks', fontsize=15);


# In[132]:


sns.countplot(data = mldata, x='gender', hue='clicked')
plt.title('Gender and clicks', fontsize=15);


# In[133]:


sns.countplot(data = mldata, x='view_hour', hue='clicked')
plt.title('Hourley views by click and unclick', fontsize=15);


# In[136]:


sns.countplot(data = mldata, x='click_hour', hue='clicked')
plt.title('Clicks by hour', fontsize=15)
plt.ylabel('Number of clicks')
plt.xlabel('hour')
plt.xticks(rotation=45);


# In[237]:


profile = ProfileReport(mldata)

profile.to_notebook_iframe()


# ### 2.5 Feature engineering

# In[8]:


# Number of clicks based on click_time (hour)

# Changing the click_time column to datetime format
mldata['click_time'] = pd.to_datetime(mldata.click_time, format='%Y-%m-%d %H:%M:%S')

# Adding two new columns click day and click hour from click_time
mldata['click_day'] = mldata['click_time'].dt.dayofweek
mldata['click_hour'] = mldata['click_time'].dt.hour


mldata.groupby('click_hour').agg({'clicked': 'sum'}).plot();
plt.ylabel('Number of clicks')
plt.xlabel('hour')
plt.title('Number of clicks by hour');


# In[9]:


# Number of clicks based on view_time (hour)

# Changing the view_time column to datetime format
mldata['view_time'] = pd.to_datetime(mldata.view_time, format='%Y-%m-%d %H:%M:%S')

# Adding two new columns view day and view hour from view_time
mldata['view_day'] = mldata['view_time'].dt.dayofweek
mldata['view_hour'] = mldata['view_time'].dt.hour


mldata.groupby('view_hour').agg({'clicked': 'sum'}).plot();
plt.ylabel('Number of clicks')
plt.xlabel('hour')
plt.title('Number of clicks by hour');


# ## 3. Develop Predictive Model <a name="3.-Develop-Predictive-Model"></a>

# In[217]:


mldata['clicked'].value_counts(normalize=True)


# Value count on clicked vs. unclicked show that the dataset is imbalanced.

# ### 3.1 Baseline model

# In[15]:


features = ['money', 'kind_card', 'kind_pay', 'network', 'gender', 'ad_id_views', 'view_hour']

X = pd.get_dummies(mldata[features].fillna(0))
y = mldata['clicked']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

dt = DecisionTreeClassifier(random_state=42)

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

print('Decision Tree', dt.score(X_test, y_test))


# #### Not able to add any more features, showing memmory error

# In[19]:


pd.Series(y_test).value_counts(normalize=True)


# In[20]:


pd.Series(y_resampled).value_counts(normalize=True)


# ### 3.2 Holdout cross validation 

# In[235]:


X.shape


# In[236]:


y.shape


# In[219]:


X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

rus=RandomOverSampler(random_state=42)

X_resampled,y_resample=rus.fit_resample(X_train,y_train)

model = DecisionTreeClassifier(max_depth=4)

model.fit(X_resampled,y_resample)

y_pred=model.predict(X_val)
print(classification_report(y_val,y_pred))


# In[175]:


model.feature_importances_


# In[183]:


model.fit(X_train_full,y_train_full)
y_pred=model.predict(X_test)
print(classification_report(y_test,y_pred))


# ### 3.3 K-fold cross validation for DecisionTreeClassifier

# In[189]:


kf=KFold(n_splits=5,shuffle=True,random_state=42)

scores = []
for train_index, val_index in kf.split(X_train_full):
    X_train, X_val = X_train_full.iloc[train_index], X_train_full.iloc[val_index]
    y_train, y_val = y_train_full.iloc[train_index], y_train_full.iloc[val_index]
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    model = DecisionTreeClassifier(max_depth=4,random_state=42)
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_val)
    score = f1_score(y_val, y_pred)
    scores.append(score)
np.mean(scores)


# ### 3.4 Cross validation using pipeline for DecisionTreeClassifier

# In[221]:


rus=RandomOverSampler(random_state=42)
dt=DecisionTreeClassifier(max_depth=4,random_state=42)
steps=[('rus',rus),('dt',dt)]
pipe=Pipeline(steps=steps)

scores=cross_val_score(pipe,X_train_full,y_train_full,cv=5,scoring='f1')

pipe.fit(X_train_full,y_train_full)
y_pred=pipe.predict(X_test)
print(classification_report(y_test,y_pred))


# ## 4. Conclusions <a name="4.-Conclusions"></a>
# 
# Click through rate project data has been analysed and DecisionTreeClassifier algorithm was applied to predict who is likely going to click on the Ad. Various features were analysed and it was found that most of the clicks were happened aroud 12 pm and 7 pm. Since the dataset was imbalanced, accuracy score cannot be used and f1-score was used to evaluate the DecisionTreeClassifier. The predicted f1-score is very low and this could be due to the issues associated with  dataset.

# In[ ]:




