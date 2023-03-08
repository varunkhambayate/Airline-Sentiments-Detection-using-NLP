#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv('Documents/Python for Data Science and Machine Learning/DATA/airline_tweets.csv')


# In[4]:


df.head()


# In[5]:


sns.countplot(data=df,x='airline_sentiment')
plt.xticks(rotation=90)


# In[6]:


sns.countplot(data=df,x='negativereason')
plt.xticks(rotation=90)


# In[7]:


data = df[['airline_sentiment','text']]


# In[8]:


data


# In[9]:


X = data['text']


# In[10]:


y = data['airline_sentiment']


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[14]:


tfidf = TfidfVectorizer(stop_words='english')


# In[15]:


tfidf.fit(X_train)


# In[16]:


X_train_tfidf = tfidf.transform(X_train)


# In[17]:


X_test_tfidf = tfidf.transform(X_test)


# In[18]:


from sklearn.metrics import plot_confusion_matrix, classification_report


# In[19]:


def report(model):
    preds = model.predict(X_test_tfidf)
    print(classification_report(y_test,preds))
    plot_confusion_matrix(model,X_test_tfidf,y_test)


# In[20]:


from sklearn.linear_model import LogisticRegression


# In[21]:


log_model = LogisticRegression()
log_model.fit(X_train_tfidf,y_train)


# In[22]:


report(log_model)


# In[23]:


from sklearn.pipeline import Pipeline 


# In[24]:


from sklearn.svm import SVC, LinearSVC
rbf_svc = SVC()
rbf_svc.fit(X_train_tfidf,y_train)
linear_svc = LinearSVC()
linear_svc.fit(X_train_tfidf,y_train)


# In[ ]:





# In[26]:


pipe = Pipeline([('tfidf',TfidfVectorizer()),('svc',LinearSVC())])
pipe.fit(X,y)
pipe.predict([' Flight'])


# In[ ]:




