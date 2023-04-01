#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation Using K-means Clustering

# # Importing The Dependencies

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
warnings.simplefilter("ignore")


# # Data Collection and Analysis

# In[2]:


#Loading the data from csv file to pandas dataframe
dataset = pd.read_csv('Mall_Customers.csv')


# In[5]:


#first 10 rows of the dataset
dataset.head(10)


# In[7]:


#shape of the dataset (rows, columns)
dataset.shape


# In[8]:


#getting some basic information about the dataset
dataset.info()


# In[10]:


#checking the null values if present in the dataset or not
dataset.isnull().sum()


# # Choosing the Annual Income Column and Spending Score Column

# In[11]:


x = dataset.iloc[:,[3,4]].values
x


# # Choosing the Number of Clusters

# ## WCSS -> Within Clusters Sum of Squares

# In[14]:


#finding WCSS values for different number of clusters

wcss =[]
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    #n_clusters -> each cluster will occur one by one till 10
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    


# In[15]:


#plot an elbow graph
sns.set()
plt.plot(range(1,11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# # Training the K-means Clustering Model

# In[16]:


kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)


# In[17]:


y = kmeans.fit_predict(x)


# In[18]:


y


# 5 clusters -> 0,1,2,3,4

# # Visualizing all the Clusters

# In[30]:


#plotting all the clusters and their centroids 
plt.figure(figsize=(8,8))
plt.scatter(x[y==0,0], x[y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(x[y==1,0], x[y==1,1], s=50, c='maroon', label='Cluster 2')
plt.scatter(x[y==2,0], x[y==2,1], s=50, c='orange', label='Cluster 3')
plt.scatter(x[y==3,0], x[y==3,1], s=50, c='violet', label='Cluster 4')
plt.scatter(x[y==4,0], x[y==4,1], s=50, c='blue', label='Cluster 5')

#plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='cyan', label='Centroids')
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()

