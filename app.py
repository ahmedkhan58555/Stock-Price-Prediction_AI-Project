import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import tensorflow
import yfinance as yf
import seaborn as sb

st.title(" Stock Prediction ")
user_input =  st.text_input('Enter Stock Tickle','AAPL')
# model = load_model('keras_model.h5')
start = '2010-01-01'
end = '2019-12-31'
tesla = yf.download(user_input ,start ,end) 

st.subheader('Dataset')
st.write(tesla)

#describing the data
st.subheader('Dataset Description')
st.write(tesla.describe())

#visualization
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(13,5))
plt.plot(tesla.Close)
plt.title('Tesla Close price.', fontsize=10)
plt.ylabel('Price in dollars.')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100ma & 200ma\nma : Moving Average')
ma100=tesla.Close.rolling(100).mean()
ma200=tesla.Close.rolling(200).mean()
fig = plt.figure(figsize=(13,5))
plt.plot(tesla.Close,'y')
plt.plot(ma100,'r')
plt.plot(ma200, 'g')
st.pyplot(fig)

# st.subheader('Graph of Every Features')
# features = ['Open', 'High', 'Low', 'Close', 'Volume']
# fig = plt.subplots(figsize=(20,10))
# for i, col in enumerate(features):
#   plt.subplot(2,3,i+1)
#   sb.distplot(tesla[col])
# st.pyplot(fig)


# st.subheader('Graph of Every Features')
# fig = plt.subplots(figsize=(20,10))
# for i, col in enumerate(features):
#   plt.subplot(2,3,i+1)
#   sb.boxplot(df[col])
# st.pyplot(fig)


#Splitting data into Trainning and Test
data_training = pd.DataFrame(tesla['Close'][0:int(len(tesla)*0.70)])
data_testing =  pd.DataFrame(tesla['Close'][int(len(tesla)*0.70) : int(len(tesla))])

#For preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


#load my model
model = load_model('keras_model.h5')

#testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing ,ignore_index = True)
input_data = scaler.fit_transform(final_df)



x_test =[]
y_test = []

for i in range (100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test ,y_test = np.array (x_test) , np.array(y_test)

y_predict = model.predict (x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predict = y_predict*scale_factor
y_test = y_test*scale_factor


#final graph
st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize = (13,5))
plt.plot(y_test,'b' ,label = 'Original Price')
plt.plot(y_predict,'r' ,label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
