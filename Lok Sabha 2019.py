#!/usr/bin/env python
# coding: utf-8

# In[407]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[280]:


# Load the data file using pandas. 
data = pd.read_csv('LS_2.0_(2).csv')
data


# In[281]:


#Replacing Nan as Not Available
data.replace({'Not Available': np.nan}, inplace=True)


# In[282]:


data.info()


# In[284]:


#Checking Null Valye
data.isnull().sum()


# In[612]:


# Checking if  null values present. We need to exclude NOTA votes while checking it.

data = data[data['PARTY']!= 'NOTA']
data = data.dropna()
data.isnull().sum()

df = data.copy()


# In[613]:


#Correcting the Names of Columns
df.rename(columns={'CRIMINAL\nCASES':'Criminal','GENERAL\nVOTES':'General Votes',
                   'POSTAL\nVOTES':'Postal Votes','TOTAL\nVOTES':'Total Votes',
                   'OVER TOTAL ELECTORS \nIN CONSTITUENCY':'OVER TOTAL ELECTORS IN CONSTITUENCY',
                   'OVER TOTAL VOTES POLLED \nIN CONSTITUENCY':'OVER TOTAL VOTES POLLED IN CONSTITUENCY'},inplace=True)


# In[614]:


#Converting the data type of the Criminal Case
df['Criminal']= df['Criminal'].astype(int)


# In[615]:


#Cleaning LIABILITIES Numeric
df['LIABILITIES'] =  df['LIABILITIES'].str.replace('[Rs ,]','')
df[['LIABILITIES','LIABILITIES2']] = df['LIABILITIES'].str.split('\n', expand=True)
df = df.drop(columns=['LIABILITIES2'])


# In[455]:


#Cleaning ASSETS Numeric
df['ASSETS'] =  df['ASSETS'].str.replace('[Rs ,]','')
df[['ASSETS','ASSETS2']] = df['ASSETS'].str.split('\n', expand=True)
df = df.drop(columns=['ASSETS2'])


# In[616]:


df.head()


# Checking the values of the Columns

# In[617]:


df['EDUCATION'].value_counts()


# In[618]:


# Removing the \n from 'Post Graduate\n'
df['EDUCATION'].replace(to_replace='Post Graduate\n', value='Post Graduate', inplace=True)

# 'Graduate Professional' are Graduates, so replacing 'Graduate Professional' with 'Graduate'
df['EDUCATION'].replace(to_replace='Graduate Professional', value='Graduate', inplace=True)

# 'Literate' = 8th Pass in our society
df['EDUCATION'].replace(to_replace='Literate', value='8th Pass', inplace=True)

# Any education level below 8th pass is illiterate
df['EDUCATION'].replace(to_replace='5th Pass', value='Illiterate', inplace=True)


# In[622]:


df['EDUCATION'].value_counts()


# In[623]:


df['GENDER'].value_counts()


# In[624]:


df['CATEGORY'].value_counts()


# In[494]:


df.info()


# In[625]:


df.describe()


# # Educational Qualification Count Graph

# In[626]:


plt.figure(figsize=(15,5))
sns.countplot(x='EDUCATION',data=df);


# # Criminal Candidate in Each State

# In[627]:


plt.figure(figsize=(10,15))
sns.barplot(x='Criminal',y='STATE',data=df,)


# # Top 15 States - Candidate involved in Criminal case

# In[708]:


state_criminal_case = df.groupby('STATE')[['Criminal']].sum().sort_values(by='Criminal',ascending = False).head(15)


# In[878]:


plt.subplots(1, figsize=(20, 8))
sns.barplot(x = state_criminal_case.index , y = state_criminal_case['Criminal'] );


# # Top 15 States - Winner Candidate Involved in Criminal Case

# In[707]:


state_criminal_winner = df[df['WINNER']>0].groupby('STATE')[['Criminal']].sum().sort_values(by='Criminal',ascending = False).head(15)


# In[884]:


plt.subplots(1, figsize=(20, 8))
sns.barplot(x = state_criminal_winner.index , y = state_criminal_winner['Criminal']  , palette='flare');


# # Education vs Criminal Barplot

# In[889]:


plt.figure(figsize=(20,6))
ax = sns.barplot(x="EDUCATION", y="Criminal", data=df)


# # GENDER CALCULATION

# In[633]:


df['GENDER'].value_counts()


# In[706]:


M=df['GENDER'].value_counts().MALE
F=df['GENDER'].value_counts().FEMALEGENDER = [M, F]
GENER_LABLE =['MALE', 'FEMALE']
plt.pie(GENDER, labels = GENER_LABLE, autopct='%1.1f%%')
plt.show() 


# 
# 

# # Barplot of category Growth:

# In[891]:


plt.figure(figsize=(5,5))
sns.countplot(x='CATEGORY',data=df,palette='YlGnBu');


# # TOP 5 Party

# In[698]:


TopParty = df['PARTY'].value_counts().head(5)
plt.subplots(1, figsize=(20, 8))
sns.barplot(x = TopParty.index , y = TopParty );


# # Top 15 Party - Candidate involved in Criminal case
# 

# In[705]:


Part_criminal_case = df.groupby('PARTY')[['Criminal']].sum().sort_values(by='Criminal',ascending = False).head(15)
plt.subplots(1, figsize=(20, 8))
sns.barplot(x = Part_criminal_case.index , y = Part_criminal_case['Criminal'] );


# # Top 15 States - Winner Candidate Involved in Criminal Case

# In[900]:


party_criminal_winner = df[df['WINNER']>0].groupby('PARTY')[['Criminal']].sum().sort_values(by='Criminal',ascending = False).head(15)
plt.subplots(1, figsize=(20, 8))
sns.barplot(x = party_criminal_winner.index , y = party_criminal_winner['Criminal'], palette='Set2' );


# # Criminal Age Group

# In[724]:


Age_criminal_winner = df[df['Criminal']>0].groupby('AGE')[['Criminal']].sum().sort_values(by='Criminal',ascending = False).head(15)


# In[901]:


plt.subplots(1, figsize=(20, 8))
sns.barplot(x = Age_criminal_winner.index , y = Age_criminal_winner['Criminal'], palette='Set1' );


# # Top 15 States with Total Votes

# In[735]:


Total_Voter = df[df['Total Votes']>0].groupby('STATE')[['Total Votes']].sum().sort_values(by=
                        ['Total Votes'], ascending= False).head(15)


# In[902]:


plt.subplots(1, figsize=(20, 8))
sns.barplot(x = Total_Voter.index , y = Total_Voter['Total Votes'],palette='Paired' );


# # Correlation

# In[737]:


correlation = df.corr()


# In[744]:


plt.subplots(1, figsize=(15, 15))
sns.heatmap(correlation, annot=True)


# # State -  Party with maximum seats

# In[775]:


State_Winner = df[df['WINNER']>0].groupby('STATE')[['PARTY']].max()


# In[784]:


plt.subplots(1, figsize=(20, 20))
sns.scatterplot(x=State_Winner['PARTY'] ,y=State_Winner.index)


# # Top 5 State with maximum Female Winner

# In[876]:


Female_winners = df[(df['WINNER']==1) & (df['GENDER']=='FEMALE')]
Female_max = Female_winners.groupby('STATE').count()["WINNER"].sort_values(ascending=False).head(10)


# In[893]:


plt.subplots( figsize=(15, 5))
sns.barplot(x = Female_max.index , y = Female_max, palette='PuBu' );


# In[ ]:





# In[ ]:




