# DSAI-HW-2022
## Execution
```
python3 app.py --training training_data.csv --output submission.csv
```
## Datasets
### Resource
1. [過去電力供需資訊](https://data.gov.tw/dataset/19995)
2. [每日尖峰備轉容量率](https://data.gov.tw/dataset/25850)
### Pre-processing
- 將 2020~2022 年的電力供需資訊整合成一份存入 training_data.csv 中
- 留下需要的欄位 `[日期]`、`[淨尖峰供電能力(MW)]`、`[尖峰負載(MW)]`、`[備轉容量(MW)]`、`[備轉容量率(%)]`做後續分析
- 為方便繪圖，將資料欄位轉成英文顯示`[Date]`、`[Net Peak Capability(MW)]`、`[Peak Load(MW)]`、`[Operating Reserve(MW)]`、`[Operating Reserve Percent(%)]`

![](https://i.imgur.com/It7Jg7E.png)
### Visualization
1. 三年內的資料趨勢
![](https://i.imgur.com/Ihhz4qa.png)
    - 可看出每年大略具有週期性的規律
    - 2021 夏季的備轉容量不穩定
2. 單年內的資料趨勢
![](https://i.imgur.com/kT1LIdZ.png)
    - 春季及冬季的變化相對較小
    - 每周亦有週期性變化

## Training
### ARIMA
本次作業使用ARIMA模型，ARIMA（Autoregressive Integrated Moving Average model）為時間序列預測分析方法之一。
![](https://i.imgur.com/ZEPjTCj.png)
步驟如下：
1. 首先將時間序列繪出趨勢、季節性、殘差三種表，了解備轉容量的特性。
![](https://i.imgur.com/uW6I8io.png)
    - 可以發現備轉容量具有季節性的特性。
2. 使用 Dickey Fuller test 進行 stationary test
    - 原始資料
        ```
        Results of Dickey-Fuller Test
        ================================================
        Test Statistic                  -2.160106
        p-value                          0.221042
        #Lags Used                      21.000000
        Number of Observations Used    768.000000
        Criterical Value (1%)           -3.438893
        Criterical Value (5%)           -2.865311
        Criterical Value (10%)          -2.568778
        dtype: float64
        ================================================
        The data is non-stationary, so do differencing!
        ```
        Test Statistic > Criterical Value (10%) ，無法拒絕原假設，因此時間序列是 non-stationary 的。
    - 一階微分
        ```
        Results of Dickey-Fuller Test
        ================================================
        Test Statistic                -1.108158e+01
        p-value                        4.297974e-20
        #Lags Used                     2.000000e+01
        Number of Observations Used    7.680000e+02
        Criterical Value (1%)         -3.438893e+00
        Criterical Value (5%)         -2.865311e+00
        Criterical Value (10%)        -2.568778e+00
        dtype: float64
        ================================================
        The data is stationary. (Criterical Value 1%)
        ```
        經過一階微分後，時間序列 stationary。
3. 畫 ACF 圖和 PACF 圖
    ![](https://i.imgur.com/8WUS3cj.png)
4. 預測模型建立
    - 尋找最小 AIC 的方式來選擇 p,d,q 的值
    ```
    ARIMA(0,1,0)：AIC=12545.556131043306
    ARIMA(0,1,1)：AIC=12344.148386365752
    ARIMA(0,1,2)：AIC=12304.607086685884
    ARIMA(1,1,0)：AIC=12444.892273927475
    ARIMA(1,1,1)：AIC=12294.503189176083
    ARIMA(1,1,2)：AIC=12294.695352995679
    ARIMA(2,1,0)：AIC=12389.524714224977
    ARIMA(2,1,1)：AIC=12295.255488542653
    ARIMA(2,1,2)：AIC=12295.934908084022
    ARIMA(3,1,0)：AIC=12369.54401935606
    ARIMA(3,1,1)：AIC=12294.899914587973
    ARIMA(3,1,2)：AIC=12295.51094928546
    ===========================================================
    This best model is ARIMA(1,1,1) based on argmin AIC.
    ```
    
    ![](https://i.imgur.com/sRktYRp.png)

    - 尋找最小 MSE 的方式來選擇 p,d,q 的值
    ```
    ARIMA(0,1,0)：MSE=207110.65426559388
    ARIMA(0,1,1)：MSE=908594.3087258801
    ARIMA(0,1,2)：MSE=687545.1844399123
    ARIMA(1,1,0)：MSE=440421.90261780703
    ARIMA(1,1,1)：MSE=591627.7934018732
    ARIMA(1,1,2)：MSE=576641.1937216343
    ARIMA(2,1,0)：MSE=617362.7177798965
    ARIMA(2,1,1)：MSE=577938.1392229861
    ARIMA(2,1,2)：MSE=598028.4534226378
    ===========================================================
    This best model is ARIMA(0,1,0) based on argmin MSE.
    ```

    ![](https://i.imgur.com/QOCq5Ce.png)

5. 預測結果

    最後由於是預測「未來」15天的資料，因此選用 MSE 的方式預測
    
    ![](https://i.imgur.com/O8tTw9g.png)
