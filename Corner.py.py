#!/usr/bin/env python
# coding: utf-8

# In[222]:


#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# In[223]:


#Load datasets
order_products = pd.read_csv('C:/Users/leona/OneDrive/Documents/DS Project - Corner/datascience-test-master-2/data/order_products.csv', sep=',') #, encoding='latin-1')
orders = pd.read_csv('C:/Users/leona/OneDrive/Documents/DS Project - Corner/datascience-test-master-2/data/orders.csv', sep=',')
shoppers = pd.read_csv('C:/Users/leona/OneDrive/Documents/DS Project - Corner/datascience-test-master-2/data/shoppers.csv', sep=',')
storebranch = pd.read_csv('C:/Users/leona/OneDrive/Documents/DS Project - Corner/datascience-test-master-2/data/storebranch.csv', sep=',')


# In[224]:


#Analysing some metrics thru descriptive statistics
shoppers.describe()
orders.describe()


# In[225]:


#Checking null values
order_products.isnull().sum()
orders.isnull().sum()
storebranch.isnull().sum()

#Null values found on the follow columns: found_rate, accepted_rate, rating
shoppers.isnull().sum()

#Identifying how these amounts represent relatively according to the whole dataset (percentual)
shoppers.isnull().sum() / shoppers["shopper_id"].count() * 100

#Checking some descriptive statistics to find an interesting way to replace the missing values
shoppers["found_rate"].describe()
shoppers["accepted_rate"].describe()
shoppers["rating"].describe()

#Checking mean and median groupped by Seniority to see if there any relation between, but discovered that it's not relevant
#Median was used to discart outliers if exist
shoppers.groupby("seniority").agg({'found_rate': ['mean','median'], 
                                   'accepted_rate': ['mean','median'], 
                                   'rating': ['mean','median']}).sort_values('seniority', ascending=True)

shoppers.agg({'found_rate': ['mean','median'], 
                                   'accepted_rate': ['mean','median'], 
                                   'rating': ['mean','median']})
#shoppers.describe()


# In[226]:


#I've decided to impute the median of each metric instead of missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer = imputer.fit(shoppers[['found_rate', 'accepted_rate', 'rating']])
shoppers[['found_rate', 'accepted_rate', 'rating']] = imputer.transform(
    shoppers[['found_rate', 'accepted_rate', 'rating']].values)


# In[227]:


#Testing a previous null case and checking that now it is input as the median values
shoppers[shoppers["shopper_id"] == "db39866e62b95bb04ebb1e470f2d1347"]


# In[228]:


orders["on_demand"].value_counts()
orders.count()
order_products.count()


# In[229]:


#Creating a procedure for splitting the "buy_unit" field into the two new fields: "quantity_UN" and "quantity_KG"
order_products.loc[order_products['buy_unit']=="UN", 'quantity_UN'] = order_products['quantity']
order_products.loc[order_products['buy_unit']!="UN", 'quantity_UN'] = 0
order_products.loc[order_products['buy_unit']=="KG", 'quantity_KG'] = order_products['quantity']
order_products.loc[order_products['buy_unit']!="KG", 'quantity_KG'] = 0
order_products


# In[230]:


#It's necessary to aggregate the 'order_products' dataset by 'order_id'
order_products_agg = order_products.groupby('order_id').sum()[['quantity_UN', 'quantity_KG']]
order_products_agg = order_products_agg.reset_index(level=0)
order_products_agg


# In[231]:


#Merging the datasets together, applying inner (using the 'order_products_agg' dataset)
corner = pd.merge(orders, order_products_agg, how="inner", on=["order_id", "order_id"])
corner = pd.merge(corner, shoppers, how="inner", on=["shopper_id", "shopper_id"])
corner = pd.merge(corner, storebranch, how="inner", on=["store_branch_id", "store_branch_id"])


# In[232]:


#Analysing the merge results
print(orders.shape)
print(order_products.shape)
print(shoppers.shape)
print(storebranch.shape)
print(corner.shape)
#print(corner.isnull().sum())
#print(corner.count())


# In[233]:


#Analysing the distribution of the label field
sns.distplot(corner['total_minutes'])

#Analysing the correlation among total_minutes and the other metrics available
corner.corr()
sns.heatmap(corner.corr(), annot=True)


# In[234]:


#Renaming the columns name
corner = corner.rename(columns={'lat_x': 'lat_order', 'lng_x': 'lng_order'})
corner = corner.rename(columns={'lat_y': 'lat_store', 'lng_y': 'lng_store'})


# In[235]:


#Converting to integer values: "True" to 1 / "False" to 0
corner['on_demand'] = corner["on_demand"].apply(lambda x : 1 if (x==True) else 0)


# In[236]:


#Converting categorical field into numeric by LabelEncoder library
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
corner['seniority'] = le.fit_transform(corner['seniority'])
corner['promised_time_'] = le.fit_transform(corner['promised_time'])

corner


# In[237]:


#Moving the rows that we want to predict (the goal of this activity) to a new dataset
predict = corner[corner['total_minutes'].isnull()]
predict = predict[['lat_order','lng_order','on_demand','quantity_UN','quantity_KG','seniority','found_rate',
      'picking_speed','accepted_rate','rating','lat_store','lng_store','promised_time_']]


# In[238]:


#Removing predict dataset
X = corner.dropna(subset=['total_minutes'], inplace=True)
Y = corner.dropna(subset=['total_minutes'], inplace=True)

#Picking up the useful fields for the feature dataset
X = corner[['lat_order','lng_order','on_demand','quantity_UN','quantity_KG','seniority','found_rate',
      'picking_speed','accepted_rate','rating','lat_store','lng_store','promised_time_']]

#Label dataset
Y = corner[['total_minutes']]


# In[245]:


print(X.shape)
print(Y.shape)
print(predict.shape)
print(corner.shape)


# In[244]:


#Transforming the features to a Standard scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
predict = scaler.fit_transform(predict)


# In[246]:


#Splitting my dataset into train/test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# In[248]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, Y_train)

