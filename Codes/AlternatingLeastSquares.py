#!/usr/bin/env python
# coding: utf-8

# In[2]:


from google.colab import drive
import os

drive.mount('/content/drive/')  


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


get_ipython().system('pip install --upgrade implicit')


# In[5]:


import implicit
from scipy.sparse import coo_matrix
from implicit.evaluation import mean_average_precision_at_k


# In[6]:


path = '/content/drive/MyDrive/Colab Notebooks/data/'
csv_train = f'{path}transactions_train.csv'

csv_users = f'{path}customers.csv'
csv_items = f'{path}articles.csv'

df = pd.read_csv(csv_train, dtype={'article_id': str}, parse_dates=['t_dat'])


# In[18]:


dfu = pd.read_csv(csv_users)
dfi = pd.read_csv(csv_items, dtype={'article_id': str})


# In[8]:


df.head()


# In[19]:


dfu.head()


# In[10]:


dfi.head()


# In[13]:


start = '2020-01-21' 
end =  str(df['t_dat'].max()).split(' ')[0]


# In[14]:


df = df[(df['t_dat'] >= start) & (df['t_dat'] <= end)]
df.shape


# In[20]:


Users= dfu['customer_id'].unique().tolist()

Articles = dfi['article_id'].unique().tolist()

user_ids = dict(list(enumerate(Users)))
item_ids = dict(list(enumerate(Articles)))


# In[21]:


map={u: id for id, u in user_ids.items()}
df['user_id']=df['customer_id'].map(map)
map={i: id for id, i in item_ids.items()}
df['item_id']=df['article_id'].map(map)


# In[25]:


r=df['user_id'].values
c=df['item_id'].values
data=np.ones(df.shape[0])
sparseMatrix=coo_matrix((data, (r, c)), shape=(len(Users), len(Articles)))


# In[26]:


sparseMatrix.shape


# In[31]:


def getCooMatrix(df,Users,Articles):
    #get the customers and articles sparse matrix
    r=df['user_id'].values
    c=df['item_id'].values
    data=np.ones(df.shape[0])
    coo=coo_matrix((data, (r, c)), shape=(len(Users), len(Articles)))
    return coo


# In[34]:


def get_Train_GT(df, validation_days=7):
    #get the ground truth
    validation_cut = df['t_dat'].max() - pd.Timedelta(validation_days)

    df_train = df[df['t_dat'] < validation_cut]
    df_GT = df[df['t_dat'] >= validation_cut]
   


    coo_train=getCooMatrix(df_train,Users,Articles)
    coo_GT=getCooMatrix(df_GT,Users,Articles)

    train=coo_train.tocsr()
    GT=coo_GT.tocsr()
    
    return {'coo_train': coo_train,'train': train,'GT': GT}


# In[35]:


def validate(matrices, factors=200, iterations=20):
    #run a expriment, factors: model para, 
    # get train and test data
    coo_train, train, GT = matrices['coo_train'], matrices['train'], matrices['GT']
    
    model = implicit.als.AlternatingLeastSquares(factors=factors,iterations=iterations,regularization=0.01)
    model.fit(coo_train, show_progress=True)
    
    

    # the evaluation function from implicit
    map12 = mean_average_precision_at_k(model, train, GT, K=12, show_progress=True, num_threads=2)
    print('Factors:',factors,' Iters:',iterations,' mean_average_precision', map12)


    return map12


# In[33]:


matrices = get_Train_GT(df)


# Main progress:

# In[ ]:


best_score = 0
best_iter=-1
best_fac=-1
for iterations in [10,100,200]:
  for factors in [50,100,200,300]:
    map12 = validate(matrices, factors, iterations)
    if map12 > best_score:
      best_score = map12
      best_iter=iterations
      best_fac=factors

print('The best result: ',best_score,' best factors:',best_fac,' best_iter: ',best_iter)


# In[ ]:





# In[ ]:




