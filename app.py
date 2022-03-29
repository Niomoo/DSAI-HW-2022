import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.tsa.api as smt 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore')

def process_data(data):
    df = pd.DataFrame(data)
    date_time = pd.to_datetime(df.pop('Date'), format='%Y%m%d')
    df['Operating Reserve(MW)'] = pd.to_numeric(df['Operating Reserve(MW)'], downcast='float', errors='coerce')
    arima_data = df['Operating Reserve(MW)']
    arima_data.index = date_time
    return arima_data

def arima_AIC(data, p=4, d=3, q=4):
    best_AIC =["pdq",10000000]
    L =len(data)
    AIC = []
    name = []
    for i in range(p):
        for j in range(1,d):
            for k in range(q):            
                model = ARIMA(data, order=(i,j,k))
                fitted = model.fit(disp=-1)
                AIC.append(fitted.aic)
                name.append(f"ARIMA({i},{j},{k})")
                print(f"ARIMA({i},{j},{k})：AIC={fitted.aic}")
                if fitted.aic < best_AIC[1]:
                    best_AIC[0] = f"ARIMA({i},{j},{k})"
                    best_AIC[1] = fitted.aic

def arima_mse(data, p=4, d=3, q=4):
    period = 3
    best_pdq =["pdq",10000000]
    L =len(data)
    train = data[:(L-period)]
    test = data[-period:]
    mse_r = []
    name = []
    for i in range(p):
        for j in range(1,d):
            for k in range(q):            
                model = ARIMA(train, order=(i,j,k))
                fitted = model.fit(disp=-1)
                fc, se, conf = fitted.forecast(period, alpha=0.05)  
                mse = mean_squared_error(test,fc)
                mse_r.append(mse)
                name.append(f"ARIMA({i},{j},{k})")
                print(f"ARIMA({i},{j},{k})：MSE={mse}")
                if mse < best_pdq[1]:
                    best_pdq[0] = f"ARIMA({i},{j},{k})"
                    best_pdq[1] = mse

def training(arima_data, fc_data):
    #data spilt
    period = 15
    L = len(arima_data)
    x_train = arima_data[:L]
    #Build Model 
    model = ARIMA(x_train, order=(0, 1, 0)) 
    fitted = model.fit(disp=-1)
    #Forecast
    fc, se, conf = fitted.forecast(period, alpha=0.05) # 95% conf
    #Make as pandas series
    fc_series = pd.Series(fc, index=fc_data.index)
    return fc_series

# You can write code above the if-main block.
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    # import pandas as pd
    df_training = pd.read_csv(args.training)
    arima_data = process_data(df_training)
    test_data = [{"Date": '20220330'},
             {"Date": '20220331'},
             {"Date": '20220401'},
             {"Date": '20220402'},
             {"Date": '20220403'},   
             {"Date": '20220404'},
             {"Date": '20220405'},
             {"Date": '20220406'},
             {"Date": '20220407'},
             {"Date": '20220408'},
             {"Date": '20220409'},
             {"Date": '20220410'},
             {"Date": '20220411'},
             {"Date": '20220412'},
             {"Date": '20220413'}]
    fc_data = pd.DataFrame(test_data)
    date_time = pd.to_datetime(fc_data.pop('Date'), format='%Y%m%d')
    fc_data.index = date_time
    # model = Model()
    # model.train(df_training)
    # df_result = model.predict(n_step=7)

    df_result = training(arima_data, fc_data)
    df_result = pd.DataFrame(df_result)
    df_result.columns = ['operating_reserve(MW)']
    df_result.to_csv(args.output)