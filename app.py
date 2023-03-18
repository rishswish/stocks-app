import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pandas_datareader import data as pdr
import yfinance as yf
import tensorflow as tf
from tensorflow import keras

yf.pdr_override()

from datetime import datetime
startdate = datetime(2013,1,1)
enddate = datetime(2022,12,31)

st.title('Stock Trend Predictions')

user_input=st.text_input('Enter Stock Ticker','AAPL')

df = pdr.get_data_yahoo(user_input, start=startdate, end=enddate)

#Describing Data
st.subheader('Data from 2010-2019')
st.write(df.describe())

st.subheader('Closing Price vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(df.Close,'g')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)
from sklearn.preprocessing import MinMaxScaler
model = keras.models.load_model('keras_model.h5')
data_train=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
past_100_days=data_train.tail(100)
final_df=past_100_days.append(data_test,ignore_index=True)
scaler=MinMaxScaler(feature_range=(0,1))
test_data=scaler.fit_transform(final_df)
x_test=[]
y_test=[]

for i in range(100,test_data.shape[0]):
    x_test.append(test_data[i-100:i])
    y_test.append(test_data[i,0])
    
x_test,y_test=np.array(x_test),np.array(y_test)
y_pred=model.predict(x_test)
scale_factor=1/0.00380952
y_pred=y_pred*scale_factor
y_test=y_test*scale_factor

st.subheader('Predictions vs Original')
fig=plt.figure(figsize=(12,6))
plt.plot(y_test,label='orginal_price')
plt.plot(y_pred,label='predicted_price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)
