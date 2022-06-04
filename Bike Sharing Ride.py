#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


data = pd.read_csv('hour.csv')
data.head()


# In[17]:


#Check for Null

data.isnull().sum()


# In[62]:


#Shape
data.shape


# In[72]:


#Duplicate Rows
duplicate = data[data.duplicated()]
duplicate


# In[24]:


#Check if registered + casual = cnt for all the records. If not, the row is junk and should be dropped.

np.sum((data.casual+data.registered) != data.cnt)


# In[75]:


#Write the Code to drop the rows where this is true

data.drop(data[data["registered"]+data["casual"]!=data["cnt"]].index, inplace= True)


# In[26]:


#Month values should be 1-12 only
np.unique(data.mnth)


# In[ ]:





# In[27]:


#Hour values should be 0-23
np.unique(data.hr)


# In[32]:


#The variables ‘casual’ and ‘registered’ are redundant and need to be dropped. 
#Instant’ is the index and needs to be dropped too. 
#The date column dteday will not be used in the model building, and therefore needs to be dropped.
#Create a new dataframe named inp1.

col_drop = ['casual','registered','instant','dteday']

inp1 = data.drop(col_drop, axis = 1).copy()
inp1.head()


# # Univariate Analysis

# In[33]:


inp1.describe()


# In[34]:


inp1.temp.plot.density()


# In[76]:


#Denisty Plot By Seaborn

sns.kdeplot(data['temp'])


# In[77]:




sns.boxplot(inp1.atemp)


# In[38]:


inp1.hum.plot.hist()


# In[78]:


sns.histplot(data = inp1.hum)


# In[ ]:





# In[ ]:





# In[39]:


inp1.windspeed.plot.density()


# In[40]:


inp1.cnt.plot.density()


# In[41]:


sns.boxplot(inp1.cnt)


# In[42]:


inp1.cnt.quantile([0.1, 0.25, 0.5, 0.70, 0.9, 0.95, 0.99])


# In[43]:


#563 is the 95th percentile – only 5% records have a value higher than this. Taking this as the cutoff.

inp2 = inp1[inp1.cnt < 563].copy()


# In[44]:


sns.boxplot(inp2.cnt)


# # Bivariate Analysis

# In[81]:


plt.figure(figsize=[12,6])
sns.boxplot("hr", "cnt", data=inp2)


# In[47]:


plt.figure(figsize=[12,6])
sns.boxplot("weekday", "hr", data=inp2)


# In[83]:




plt.figure(figsize=[10,6])

sns.boxplot("mnth", "cnt", data=inp2)


# In[53]:


plt.figure(figsize=[10,6])
sns.boxplot("season", "cnt", data=inp2)


# In[88]:


#Make a bar plot with the median value of cnt for each hr
#Bar Plot Stick tell about the error

sns.barplot(x='hr', y='cnt', data =inp2)


# In[89]:


num_vars = ['temp', 'atemp', 'hum', 'windspeed']
corrs = inp2[num_vars].corr()
corrs


# In[61]:


sns.heatmap(corrs, annot=True, cmap="Reds")


# # Data preprocessing
# 
# A few key considerations for the preprocessing: 
# 
# There are plenty of categorical features. Since these categorical features can’t be used in the predictive model, you need to convert to a suitable numerical representation. Instead of creating dozens of new dummy variables, try to club levels of categorical features wherever possible. For a feature with high number of categorical levels, you can club the values that are very similar in value for the target variable. 

# 1. Treating mnth column
# 
# For values 5,6,7,8,9,10, replace with a single value 5. This is because these have very similar values for cnt.

# In[116]:


inp3 = inp2.copy()


# In[117]:


#Replace Function

inp3.mnth[inp3.mnth.isin([5,6,7,8,9])] = 5


# In[118]:


np.unique(inp3.mnth)


# Treating hr column
# 
# Create new mapping: 0-5: 0, 11-15: 11; other values are untouched. Again, the bucketing is done in a way that hr values with similar levels of cnt are treated the same.

# In[119]:


inp3.hr[inp3.hr.isin([0,1,2,3,4])] = 0
inp3.hr[inp3.hr.isin([11,12,13,14,15])] = 11


# In[120]:


np.unique(inp3.hr)


# #Get dummy columns for season, weathersit, weekday, mnth, and hr.

# In[121]:


list = ['season', 'weathersit', 'weekday', 'mnth','hr']


# In[123]:


inp3 = pd.get_dummies(inp3, columns=list)


# In[124]:


inp3.head()


#  Train test split: Apply 70-30 split.
# 
# - call the new dataframes df_train and df_test
# 

# In[154]:


from sklearn.model_selection import train_test_split
df_train, df_test =  train_test_split(inp3, test_size=0.30, random_state=32)


# In[155]:


df_train.shape


# In[156]:


df_test.shape


#  Separate X and Y for df_train and df_test. For example, you should have X_train, y_train from df_train. y_train should be the cnt column from inp3 and X_train should be all other columns.

# In[157]:


y_train = df_train.pop("cnt") #Poped cnt stored in y_train
X_train = df_train


# In[158]:


y_test = df_test.pop("cnt")
X_test = df_test


# In[159]:


X_train


# In[160]:


y_train


# In[161]:


X_test


# In[162]:


y_test


# # Model building
# 
# Use linear regression as the technique

# In[163]:


from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()


# In[164]:


#Fit() training

linear_reg.fit(X_train, y_train)


# In[166]:


y_pred = linear_reg.predict(X_test)
y_pred


# Report the R2 on the train set

# In[170]:


#Calculate r2 score
from sklearn.metrics import r2_score

print(r2_score(y_pred,y_test))


# In[172]:


#Cross validation

from sklearn.metrics import r2_score

print(r2_score(linear_reg.predict(X_train),y_train))


# In[173]:


#Overfitting - Training accuracy is more but test accuracy is less


# In[ ]:




