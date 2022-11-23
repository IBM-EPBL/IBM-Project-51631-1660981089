#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import requests
import io
url = r"https://github.com/IBM-EPBL/IBM-Project-51631-1660981089/blob/main/Data%20Collection/DataSet1986-2018.xlsx?raw=true"
download = requests.get(url).content
data = pd.read_excel(url,index_col=0,parse_dates=[0])
print(data.head())


# In[3]:


data.isnull().any()
data.isnull().sum()
data.dropna(axis = 0, inplace = True)
data.isnull().sum()
data_oil = data.reset_index()['Closing Value']
data_oil


# In[4]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler (feature_range=(0,1))
data_oil=scaler.fit_transform(np.array(data_oil).reshape(-1,1))


# In[5]:


plt.plot(data_oil)


# In[6]:


training_size=int(len(data_oil)*0.65)
test_size=len(data_oil)-training_size
train_data, test_data=data_oil[0:training_size,:],data_oil[training_size:len(data_oil),:1]
training_size,test_size
train_data.shape


# In[7]:


def create_dataset(dataset, time_step=1): 
  dataX, dataY = [], [] 
  for i in range(len(dataset)-time_step-1):
    a = dataset[i:(i+time_step), 0]
    dataX.append(a)
    dataY.append(dataset[i + time_step, 0])
  return np.array(dataX), np.array(dataY)
  
time_step = 10
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)
print(X_train.shape), print(y_train.shape)
print(X_test.shape), print(ytest.shape)
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test= X_test.reshape(X_test.shape[0],X_test.shape[1],1)


# In[8]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[9]:


model = Sequential()


# In[10]:


model.add(LSTM(50,return_sequences=True,input_shape=(10,1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))


# In[11]:


model.add(Dense (1))
model.summary()


# In[ ]:




