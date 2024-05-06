#!/usr/bin/env python
# coding: utf-8

# 

# In[145]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[146]:


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# In[147]:


df = pd.read_csv('./titanic.csv')
df.shape


# In[148]:


df.sample(10)


# In[149]:


df.info()


# In[150]:


df.isnull().sum()


# In[151]:


df.describe()


# In[152]:


for x in df.columns:
    print([x,df[x].nunique()])


# In[153]:


cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
def search_substring(big_string, substring_list):
    for substring in substring_list:
        if substring in big_string:
            return substring
    return substring_list[-1]


# In[154]:


def get_title(string):
    import re
    regex = re.compile(r'Mr|Don|Major|Capt|Jonkheer|Rev|Col|Dr|Mrs|Countess|Dona|Mme|Ms|Miss|Mlle|Master', re.IGNORECASE)
    results = regex.search(string)
    if results != None:
        return(results.group().lower())
    else:
        return(str(np.nan))


# In[155]:


title_dictionary = {
    "capt":"Officer", 
    "col":"Officer", 
    "major":"Officer", 
    "dr":"Officer",
    "jonkheer":"Royalty",
    "rev":"Officer",
    "countess":"Royalty",
    "dona":"Royalty",
    "lady":"Royalty",
    "don":"Royalty",
    "mr":"Mr",
    "mme":"Mrs",
    "ms":"Mrs",
    "mrs":"Mrs",
    "miss":"Miss",
    "mlle":"Miss",
    "master":"Master",
    "nan":"Mr"
}


# In[156]:


df['Deck'] = df['Cabin'].map(lambda x: search_substring(str(x), cabin_list))
df['Title'] = df['Name'].apply(get_title)


# In[157]:


df = df.drop(columns=['PassengerId','Name','Ticket','Cabin'], axis=1)


# In[158]:


df_obj = df.select_dtypes(include=['object'])
df_num = df.select_dtypes(exclude=['object'])


# In[159]:


num_imputer = SimpleImputer(strategy="median")
df["Age"] = num_imputer.fit_transform(df[["Age"]])
df["Fare"] = num_imputer.fit_transform(df[["Fare"]])


# In[160]:


cat_imputer = SimpleImputer(strategy="most_frequent")
imputed_embarked = cat_imputer.fit_transform(df[["Embarked"]])
df["Embarked"] = pd.Series(imputed_embarked.flatten(), name="Embarked")


# In[161]:


label_encoder = LabelEncoder()
for c in df_obj.columns:
    df[c] = label_encoder.fit_transform(df[c])


# In[162]:


scaler = StandardScaler()
df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])


# In[163]:


X = df.drop("Survived", axis=1)
y = df["Survived"]

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[164]:


df.sample(5)


# In[ ]:





# In[ ]:




