#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install numpy


# In[3]:


pip install pandas


# In[4]:


pip install matplotlib


# In[5]:


pip install pandas_datareader


# In[6]:


pip install yfinance


# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pandas_datareader as data
import yfinance as yf
import pickle


# In[8]:


# stock_symbol =  ('TSLA' , 'AAPL' , 'MSFT' )
tesla = pd.read_csv("C:/Users/Umer/Downloads/tesla.csv")
# tesla = yf.download(tickers = stock_symbol , period = '20y' , interval = '1d')                    
tesla.head()
# start = '2010-01-01'
# end = '2019-12-31'
# tesla =data.DataReader('AAPL','yahoo',start ,end)


# In[9]:


tesla.info()


# In[10]:


tesla['Date'] = pd.to_datetime(tesla['Date'])
print(f'Dataframe contains stock prices between {tesla.Date.min()} {tesla.Date.max()}') 
print(f'Total days = {(tesla.Date.max()  - tesla.Date.min()).days} days')


# In[11]:


tesla.info()


# In[12]:


tesla.describe()


# In[13]:


tesla = tesla.reset_index()
tesla.head()


# In[14]:


tesla = tesla.drop(['Date','Adj Close'], axis =1)
tesla.head()


# In[15]:


plt.figure(figsize=(15,5))
plt.plot(tesla['Close'])
plt.title('Tesla Close price.', fontsize=10)
plt.ylabel('Price in dollars.')


# In[16]:


tesla.isnull().sum()


# In[17]:


tesla


# In[18]:


features = ['Open', 'High', 'Low', 'Close', 'Volume']
 
plt.subplots(figsize=(20,10))
 
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.distplot(tesla[col])
plt.show()


# In[19]:


#Moving Average
ma100 = tesla.Close.rolling(100).mean()
ma100


# In[20]:


ma200 = tesla.Close.rolling(200).mean()
ma200


# In[21]:


plt.figure(figsize = (15,5))
plt.plot(tesla.Close,'y')
plt.plot(ma100, 'r')


# In[22]:


plt.figure(figsize = (15,5))
plt.plot(tesla.Close,'y')
plt.plot(ma200, 'g')


# In[23]:


plt.figure(figsize = (15,5))
plt.plot(tesla.Close,'y')
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')


# In[24]:


tesla.shape


# In[25]:


#Splitting data into Trainning and Test

data_training = pd.DataFrame(tesla['Close'][0:int(len(tesla)*0.70)])
data_testing =  pd.DataFrame(tesla['Close'][int(len(tesla)*0.70) : int(len(tesla))])

print(data_training.shape)
print(data_testing.shape)


# In[26]:


data_training.head()


# In[27]:


data_testing.head()


# In[28]:


#For preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)
data_training_array


# In[29]:


x_train = []
y_train = []

for i in range (100,data_training_array.shape[0]):
  x_train.append(data_training_array[i-100:i])
  y_train.append(data_training_array[i,0])

x_train , y_train = np.array(x_train) , np.array(y_train)


# In[30]:


x_train.shape


# In[31]:


get_ipython().system('pip install keras')


# In[32]:


get_ipython().system('pip install tensorflow')


# In[33]:


#ML Model
from keras.layers import Dense , Dropout , LSTM
from keras.models import Sequential


# In[34]:


model = Sequential()
model.add(LSTM(units = 50 , activation = 'relu' ,return_sequences = True,
               input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60 , activation = 'relu' ,return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80 , activation = 'relu' ,return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120 , activation = 'relu'))
model.add(Dropout(0.5))
    

model.add(Dense(units = 1))


# In[35]:


model.summary()


# In[36]:


model.compile(optimizer = 'adam' , loss= 'mean_squared_error') #for time series analysis ,we use mean squared error
hist=model.fit(x_train , y_train , epochs = 50)


# In[37]:


plt.figure(figsize = (15,6))
plt.plot(hist.history['loss'])
plt.title('Training model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


# In[38]:


model.save('keras_model.h5')


# In[39]:


data_testing.head() #to predict 1535 day ,I need previous 100 days to evaluate test data


# In[40]:


data_testing.tail(100)


# In[41]:


past_100_days = data_training.tail(100)


# In[42]:


final_df = past_100_days.append(data_testing ,ignore_index = True) 
#appending data training 100 days with data testing


# In[43]:


final_df.head()


# In[44]:


input_data = scaler.fit_transform(final_df)
input_data


# In[45]:


input_data.shape


# In[46]:


x_test =[]
y_test = []

for i in range (100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test ,y_test = np.array (x_test) , np.array(y_test)


# In[47]:


x_test.shape


# In[48]:


y_test.shape


# In[49]:


#making prediction 
y_predict = model.predict (x_test)


# In[50]:


y_predict.shape


# In[51]:


y_test


# In[52]:


y_predict


# In[53]:


scaler.scale_


# In[54]:


scale_factor = 1/0.0049128
y_predict = y_predict*scale_factor
y_test = y_test*scale_factor


# In[55]:


plt.figure(figsize = (12,6))
plt.plot(y_test,'b' ,label = 'Original Price')
plt.plot(y_predict,'r' ,label = 'Predicted Price')

plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




