#!/usr/bin/env python
# coding: utf-8

# In[74]:


import pandas as pd
#col_names = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity',  'speechiness', 'tempo', 'valence', 'year']
# load dataset
pima = pd.read_csv("C:/Users/DELL/Desktop/data.csv", header=None)


# In[75]:


pima.head()


# In[76]:


pima=pima.iloc[1:, :]


# In[85]:


import numpy as np
feature_cols = [0,2,3,4,5,7,8,9,10,13,15,16,17,18]
X = pima[feature_cols] # Features
Y = pima[11]
Y=Y.astype('int')


# In[86]:


X.head()


# In[87]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[88]:


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16)

# fit the model with data

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)


# In[89]:


from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[91]:


# import required modules
import seaborn as sns
sns.heatmap(cnf_matrix, annot=True)


# In[92]:


sns.heatmap(cnf_matrix/np.sum(cnf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')


# In[94]:


from sklearn.metrics import classification_report
target_names = ['without mode', 'with mode']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[ ]:




