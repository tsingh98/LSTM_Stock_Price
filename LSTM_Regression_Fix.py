import pandas as pd
#import modin.pandas as pd
import quandl
quandl.ApiConfig.api_key = "47d6qS5DtPwi1miuHQHh"
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from sklearn.model_selection import GridSearchCV 
from time import time
#from report import report
import warnings


#import glob

from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.svm import SVC

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout



#Using Bollinger_percent, 20-day STD, MACD, exp3, RSI, Stochastic Oscillator 
def Indicators(df,period):
    #df['pct_change'] = df.open.pct_change()
    #df['log_return'] = np.log(1 + df['pct_change'])
    
    df['Middle Band'] = df['open'].rolling(window=20).mean()
    df['20 Day STD'] = df['open'].rolling(window=20).std()
    df['Upper Band']= df['Middle Band']+df['20 Day STD']*2
    df['Lower Band']= df['Middle Band']-df['20 Day STD']*2
    df['Band Width']= ((df['Upper Band'] - df['Lower Band']) / df['Middle Band']) * 100
    df['Bollinger_percent'] = ((df.close-df['Lower Band'])/(df['Upper Band']-df['Lower Band']))*100
    
    
    #df['exp1'] = df.apply(lambda x: x['close'].ewm(span=12,min_periods=12).mean())
    #df['exp2'] = df.apply(lambda x: x['close'].ewm(span=26,min_periods=26).mean())
    df['exp1'] = df.close.ewm(span=12,min_periods=12, adjust=False).mean()
    df['exp2'] = df.close.ewm(span=26,min_periods=26, adjust=False).mean()
    
    df['MACD'] = df['exp1']-df['exp2']
    
    df['exp3'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    RSI_computer = pd.DataFrame(df.close.shift().diff())
    RSI_computer.rename(columns={'close': 'delta'},inplace = True)
    
    RSI_computer['u'] = RSI_computer.delta * 0
    RSI_computer['d'] = RSI_computer.u.copy()
    
    RSI_computer['u'].loc[(RSI_computer.delta > 0)] = RSI_computer.delta.loc[(RSI_computer.delta >0)]
    RSI_computer['d'].loc[(RSI_computer.delta < 0)] = -RSI_computer.delta.loc[(RSI_computer.delta <0)]
    
    RSI_computer['u'].loc[RSI_computer.u.index[period-1]]= np.mean(RSI_computer.u.loc[:period])
    RSI_computer.u = RSI_computer.u.drop(RSI_computer.u.index[:(period-1)])
    
    RSI_computer['d'].loc[RSI_computer.d.index[period-1]]= np.mean(RSI_computer.d.loc[:period])
    RSI_computer.d = RSI_computer.d.drop(RSI_computer.d.index[:(period-1)])
    
    RSI_computer['RS'] = RSI_computer.u.ewm(span=period-1, adjust=False).mean()/ RSI_computer.d.ewm(span=period-1, adjust=False).mean()
    
    df['RSI_final']= pd.Series(100-100/(1+RSI_computer.RS))
    #df.join(RSI_final) 
    #print('Fine')
    
    df['Stochastic_Oscillator'] = (df.close-df.low.rolling(14).min()/df.high.rolling(14).max()-df.low.rolling(14).min())*100 
    #print('Fine2')
    
    dm = ((df['high'].shift() + df['low']/2) - ((df['high'].shift() + df['low'].shift())/2))
    br = (df['volume'] / 100000000) / ((df['high'] - df['low']))
    
    EVM = dm / br 
    df['EVM_MA'] = pd.Series(EVM.rolling(14).mean()) 
    
    
    df_prelim = df[['date','Bollinger_percent','MACD','exp3','RSI_final','Stochastic_Oscillator','close','EVM_MA','Band Width']]
    
    
    
    df_final = DealWithMissing(df_prelim)
    #print('Fine3')
    #df_final2 = Normalise(df_final)
    #print('Fine4')
    df_final['Difference']= df_final['close'].diff(periods=-1)
    #print('Fine5')
    df_final['UP_DOWN'] = np.where(df_final['Difference']>0,1,0) 
    #print('Fine6')
    df_final3 = DealWithMissing(df_final)

        
    
    df_submit = df_final3[['date','Bollinger_percent','MACD','exp3','RSI_final','Stochastic_Oscillator','close','EVM_MA','Band Width','UP_DOWN']]
    
            
    
    return(df_submit,df_final3)
    
    
def Normalise(df):
    
    normalise_df = (df-df.mean())/df.std()
    
    return(normalise_df)
   
def DealWithMissing(df):
    fixed_df = df.dropna(how='any')
    return(fixed_df)
    

def dict_maker(df,list_tickers):
    dict1 = {}
    for i in list_tickers:
        dict1[i]=df[df['ticker']==i]
    return(dict1)    




def Panda_sorter(df):
    df1 = df.sort_values(by=['AUC'],ascending=False)
    return(df1)
        
    
def LSTM_Price(df,key):
    
    #df = Indicative_dicts['AAPL']
    
    colnum=df.shape[0]
    colnum-1000

    
    sc = MinMaxScaler(feature_range = (0, 1))
    sc1 = MinMaxScaler(feature_range = (0, 1))
    
    #sc1 = MinMaxScaler(feature_range = (0, 1))

    #Real stock values for testing
    Close_prices = df[['date','close']]
    Close_prices= Close_prices.iloc[colnum-1000:,:]
    #Close_prices = Close_prices.reshape(-1,1)
    #Close_prices = sc1.fit_transform(Values_price)
    
    #f1=Close_prices[['close']]
    #f1.reshape(-1,1)
    
    #scaled features
    Features = df.iloc[:,1:9].values
    
    f1 = Features[:,5]
    f1=f1.reshape(-1,1)
    Placeheld = sc1.fit_transform(f1)
    Features_Scaled = sc.fit_transform(Features)
    #Values_c = ADI_df.iloc[:,9].values
    
        
    X_train = []
    y_train = []
    
    for i in range(60,colnum-1000):
        X_train.append(Features_Scaled[i-60:i,:])
        y_train.append(Features_Scaled[i,5])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    X_test = []
    y_test = []
    
    for i in range(colnum-1000,colnum):
        X_test.append(Features_Scaled[i-60:i,:])
        y_test.append(Features_Scaled[i,5])
        
    X_test, y_test = np.array(X_test), np.array(y_test)    
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 8))



        
    regressor = Sequential()    
    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 8)))
    regressor.add(Dropout(0.2))
    
    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    
    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    
    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))
    
    # Adding the output layer
    regressor.add(Dense(units = 1))
    
    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs = 15, batch_size = 32,validation_split=0.2)


        
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc1.inverse_transform(predicted_stock_price)
    predicted_stock_price = pd.DataFrame(predicted_stock_price)
    
   # together = pd.concat([Close_prices, pd.DataFrame(predicted_stock_price)], axis=1, ignore_index=True)
    together = pd.concat([Close_prices.reset_index(drop=True), predicted_stock_price], axis=1)    
    
    
    fig,ax  = plt.subplots()
    fig.autofmt_xdate()
    ax.plot(Close_prices.iloc[:,0],Close_prices.iloc[:,1], color = 'blue', label = 'Predicted {} Stock Price'.format(key))
    ax.plot(together.iloc[:,0] ,together.iloc[:,2] ,color = 'red', label = 'Real {} Stock Price'.format(key))
    plt.title('{} Stock Price Prediction'.format(key))
    plt.xlabel('Time')
    plt.ylabel('{} Stock Price'.format(key))
    plt.legend()
    plt.savefig('./Plots/{}.pdf'.format(key))


    


    #"Shepherd {} is {} years old.".format(shepherd, age)

# Creating a data structure with 60 timesteps and 1 output
"""
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
"""
    
    
    
def main():
    
    tickers_10_data=pd.read_csv('./tickers.csv')
    list_tickers=tickers_10_data['Ticker'].tolist()
    
    _10_tickers_data = quandl.get_table('WIKI/PRICES',ticker=list_tickers, date={"gte": '2000-01-01', 'lte': '2019-01-10'},
                         paginate=True) 
    
    
    _10_tickers_data = _10_tickers_data.iloc[::-1]
    
    if(_10_tickers_data.isnull().values.any()):
        Tickers_ready = DealWithMissing(_10_tickers_data)
    else:
        Tickers_ready=_10_tickers_data 
        
    Ticks_dict = dict_maker(Tickers_ready,list_tickers)
    
    
    Indicative_dicts = {}
    Feature_eng_dicts = {}
    
    for key in Ticks_dict:
        Indicative_dicts[key],Feature_eng_dicts[key]=Indicators(Ticks_dict[key],14)
        
    for key in Indicative_dicts:
        LSTM_Price(Indicative_dicts[key],key)
 



if __name__== "__main__":
    main()
    

