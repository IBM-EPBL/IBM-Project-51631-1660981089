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


# In[12]:


model.compile(loss= 'mean_squared_error', optimizer='adam')


# In[13]:


model.fit(X_train,y_train,validation_data=(X_test,ytest), epochs=50,batch_size=64, verbose = 1)


# In[14]:


train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

import math 
from sklearn.metrics import mean_squared_error 
math.sqrt(mean_squared_error(y_train,train_predict))


# In[15]:


from tensorflow.keras.models import load_model
model.save("crude_oil.h5")


# In[16]:


look_back=10
trainpredictPlot = np.empty_like(data_oil)
trainpredictPlot[:, :] = np.nan
trainpredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

testPredictPlot = np.empty_like(data_oil)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(data_oil)-1, :] = test_predict

plt.plot(scaler.inverse_transform(data_oil))
plt.plot(trainpredictPlot)
plt.plot(testPredictPlot)
plt.show()

len(test_data)
x_input=test_data[2866:].reshape(1,-1)
x_input.shape
temp_input=list(x_input)
temp_input=temp_input[0].tolist()
temp_input

lst_output=[]
n_steps=10
i=0
while(i<10):
  if(len(temp_input)>10):

    x_input=np.array(temp_input[1:]) 
    print("{} day input {}".format(i,x_input))
    x_input=x_input.reshape(1,-1)
    x_input = x_input.reshape((1, n_steps, 1))

    yhat = model.predict(x_input, verbose=0)
    print("{} day output {}".format(i,yhat))
    temp_input.extend(yhat[0].tolist())
    temp_input=temp_input[1:] #print(temp_input)
    lst_output.extend(yhat.tolist())
    i=i+1
  else:
    x_input = x_input.reshape((1, n_steps,1))
    yhat = model.predict(x_input, verbose=0) 
    print(yhat[0]) 
    temp_input.extend(yhat[0].tolist()) 
    print(len(temp_input)) 
    lst_output.extend(yhat.tolist()) 
    i=i+1

day_new=np.arange(1,11)
day_pred=np.arange(11,21)
len(data_oil)

plt.plot(day_new, scaler.inverse_transform(data_oil[8206:]))
plt.plot(day_pred, scaler.inverse_transform(lst_output))

df3=data_oil.tolist()
df3.extend(lst_output)
plt.plot(df3[8100:])

df3=scaler.inverse_transform(df3).tolist()


# In[ ]:




