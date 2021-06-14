#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


url="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"


# In[4]:


df = pd.read_csv(url)


# In[5]:


df.head()


# In[6]:


df.plot(x='Hours', y='Scores', style='o')
plt.title('Hours Vs Scores')
plt.xlabel('Time in Hours')
plt.ylabel('Scores in %')
plt.show()


# In[7]:


X = df.iloc[:, :-1].values  
y = df.iloc[:, 1].values


# In[8]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)


# In[9]:


from sklearn.linear_model import LinearRegression


# In[10]:


lr = LinearRegression()


# In[11]:


lr.fit(X_train, y_train)


# In[12]:


lr.score(X_train, y_train)


# In[13]:


lr.score(X_test, y_test)


# In[14]:


pred = lr.predict(X_test)


# In[15]:


from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[16]:


print(mean_squared_error(pred, y_test))


# In[17]:


print(np.sqrt(mean_squared_error(pred, y_test)))


# In[19]:


line = lr.coef_*X+lr.intercept_
plt.scatter(X,y)
plt.plot(X,line)
plt.show()


# In[20]:


df1 = pd.DataFrame(y_test)
df1


# In[21]:


df1['Prediction'] = pred


# In[22]:


df1


# In[23]:


hours = [[3]]
pred1 = lr.predict(hours)
pred1


# In[24]:


hours = [[5]]
pred2 = lr.predict(hours)
pred2


# In[25]:


hours = [[7]]
pred3 = lr.predict(hours)
pred3


# In[26]:


hours = [[9.25]]
pred2 = lr.predict(hours)
pred2

