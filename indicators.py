import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

#SMA
def SMA(data,window_size=20):
    res=data.rolling(window_size).mean()
    res.dropna(inplace=True)
    return res

#Bollinger band
def bollinger_bands(data,window_size=20,multiplier=2):
    sma=SMA(data,window_size)
    sma.dropna(inplace=True)

    std=data.rolling(window_size).std()
    std.dropna(inplace=True)
    upper=sma+std*multiplier
    lower=sma-std*multiplier

    return upper,lower,sma

def B_percent(data,upper,lower,window_size=20):
    if len(data)!=len(upper):
        data=data.iloc[20:]
    res=(data-lower)/(upper-lower)
    return res

#RSI
def rsi(data,window_size=14):
    data=pd.DataFrame(data)
    daily_returns=data-data.shift(1)
    daily_returns.iloc[0]=0.0

    gain=pd.DataFrame(index=daily_returns.index)
    gain=gain.join(daily_returns.loc[(daily_returns>0).any(axis=1)])
    gain=gain.fillna(0)
    gain=gain.rolling(window_size).sum()
    gain.dropna(inplace=True)
    gain=gain/window_size
    
    loss=pd.DataFrame(index=daily_returns.index)
    loss=loss.join(daily_returns.loc[(daily_returns<0).any(axis=1)])
    loss=loss.fillna(0)
    loss=loss.rolling(window_size).sum()
    loss.dropna(inplace=True)
    loss=(-1*loss)/window_size

    res=100-(100/(1+gain/loss))
    
    return res

#EMA
def EMA(data,window_size=10):
    res=data.ewm(span=window_size)
    return res.mean()

#MACD
def macd(data,p1=12,p2=26,p3=9):
    EMA12=EMA(data,p1)
    EMA26=EMA(data,p2)
    macd_line=EMA12-EMA26
    signal_line=EMA(macd_line,window_size=p3)
    macd_hist=macd_line-signal_line

    return macd_line,signal_line,macd_hist

#stochastic indicator
def stochastic_oscillator(data_close,data_high,data_low,k_period=14,d_period=3):
    highest_high=data_high.rolling(k_period).max()
    lowest_low=data_low.rolling(k_period).min()

    K_percent=(data_close-lowest_low)/(highest_high-lowest_low)*100
    K_percent.dropna(inplace=True)
    D_percent=K_percent.rolling(d_period).mean()
    D_percent.dropna(inplace=True)
    return K_percent,D_percent

#momentum
def momentum(data,period=20):
    res=(data-data.shift(period))/data.shift(period)*100
    return res