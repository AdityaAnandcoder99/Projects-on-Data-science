#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = sns.load_dataset('iris')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df['species'].unique()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


sns.scatterplot(x='sepal_length', y='petal_length', data=df)


# In[10]:


sns.scatterplot(x='sepal_length', y='petal_length',hue ='species', data=df)


# In[11]:


sns.scatterplot(x='sepal_width', y='petal_width',hue ='species', data=df)


# In[12]:


sns.scatterplot(x='sepal_length', y='petal_width',hue ='species', data=df)


# In[13]:


sns.scatterplot(x='sepal_width', y='petal_length',hue ='species', data=df)


# In[15]:


sns.displot(df['sepal_length'])


# In[16]:


sns.displot(df['sepal_width'])


# In[18]:


sns.displot(df['petal_length'])


# In[19]:


sns.displot(df['petal_width'])


# In[20]:


iris = pd.DataFrame(df)
iris_df = df.drop(columns= ['species'] )
iris_df.head()


# In[21]:


from sklearn.cluster import KMeans


# In[22]:


wcss = []

clusters_range = range(1,15)
for k in clusters_range:
    km = KMeans(n_clusters=k)
    km = km.fit(iris_df)
    wcss.append(km.inertia_)


# In[23]:


plt.plot(clusters_range, wcss, 'go--', color='b')
plt.title('Elbowl Graph')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster sum of square')
plt.grid()
plt.show()


# In[24]:


model = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
predictions = model.fit_predict(iris_df)


# In[25]:


print(predictions)


# In[28]:


x = iris_df.iloc[:, [0, 1, 2, 3]].values
plt.scatter(x[predictions == 0, 0], x[predictions == 0, 1], s = 25, c = 'red', label = 'Iris-setosa')
plt.scatter(x[predictions == 1, 0], x[predictions == 1, 1], s = 25, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[predictions == 2, 0], x[predictions == 2, 1], s = 25, c = 'green', label = 'Iris-virginica')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
plt.legend()
plt.grid()
plt.show()

