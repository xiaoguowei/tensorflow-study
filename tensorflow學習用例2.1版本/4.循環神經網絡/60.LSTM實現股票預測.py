#傳統循環網絡RNN,可以通過記憶體實現短期記憶進行連續數據的預測,但是當連續數據的序列變長時,會使展開時間步過長,在反向傳播更新參數時,梯度要按時間步連續相乘,會導致梯度消失

# 所以在1977年提出了長短記憶網絡LSTM,長短網絡中引入了三個門限,LSTM計算過程如下:
# 1.輸入門(i(t))     #i(t) == o(W(i)*[ h(t-1) , X(t) ] + b(i))  #W(i),W(f),W(o)是帶訓練參數矩陣,X(t)是當前時刻的輸入特征,h(t-1)是上一時刻的短期記憶的函數
# 2.遺忘門(f(t))     #i(t) == o(W(f)*[ h(t-1) , X(t) ] + b(f))  #b(i),b(f),b(o),是待訓練參數矩陣,三個門限都經過sigmoid的函數,使門限範圍在0~1之間
# 3.輸出門(o(t))     #i(t) == o(W(o)*[ h(t-1) , X(t) ] + b(o))
# 4.表征長期記憶的細胞態(c(t))  # C(t) = f(t) * C(t-1) + i(t) * (c(t)波浪號) #細胞態 = 遺忘門 * 上個時刻的長期記憶 + 輸入門 * 當前時刻歸納出的新知識
# 5.記憶體(短期記憶)           # ht = o(t) * tanh(c(t)) # 輸出門 * 細胞態過tanh激活函數
# 6.等待存入長期記憶的候選態(C(t)波浪號)(歸納出的新知識) # (C(t)波浪號) =tanh( W(c)*[ h(t-1),X(t) ]+b(c) ) #

################# TF描述LSTM層##################3
# tf.keras.layers.LSTM(記憶體個數, return_sequences=是否返回輸出) #True 各時間步輸出ht  False(默認)僅最後時間步輸出ht

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import  Dropout, Dense, LSTM
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(threshold=np.inf)
####################### 配置GPU內存 ###############################################
gpus = tf.config.experimental.list_physical_devices('GPU')
GPU=1024*5 #配置多少GPU
tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU)])
########################## 配置GPU內存 ###############################################

maotai = pd.read_csv("./SH600519.csv") #讀取股票文件

training_set = maotai.iloc[0:2426-300,2:3].values #前(2426-300=2126)天的開盤價作為訓練集,表格從0開始計數,2:3是提取[(2:3)]列,前閉後開,固提出C列開盤價
test_set = maotai.iloc[2426 - 300:, 2:3].values #後300天的開盤價作為測試集

#歸一化
sc = MinMaxScaler(feature_range=(0,1)) #定義歸一化,歸一化到(0,1)之間
training_set_scaled = sc.fit_transform(training_set) #求得訓練集的最大值,最小值這些訓練集固有的屬性,並在訓練集上進行歸一化
test_set = sc.transform(test_set) #利用訓練集的屬性對測試集進行歸一化

x_train = [] #用來接收訓練集輸入特征
y_train = [] #用來接收訓練集標籤

x_test = [] #用來接收測試集輸入特征
y_test = [] #用來接收測試集標籤

# 測試集:csv表格中前2426-300=2126天數據
# 利用for循環,遍歷整個訓練集,
for i in range(60, len(training_set_scaled)):
    x_train.append(training_set_scaled[i-60:i,0]) #提取訓練集中連續60天的開盤價作為輸入特征x_train,
    y_train.append(training_set_scaled[i, 0]) #第61天的數據作為標籤,for循環共構建2426-300-60=2066組數據

#對訓練集進行打亂
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
np.random.seed(7)
# 將訓練集由list格式變為array格式
x_train, y_train = np.array(x_train), np.array(y_train)

# 使x_train符合RNN輸入要求:[送入樣本數, 循環核時間展開數, 每個時間步輸入特征個數].
# 此處整個數據集送入,送入樣本手為x_train[0]即2066組數據; 輸入60個開盤價,預測出第61天的開盤價,循環核時間展開步數為60; 每個時間步步入的特征是某一天的開盤價,只有1個數據,故每個時間步輸入特征個數為1
x_train = np.reshape(x_train, (x_train.shape[0], 60 ,1))
# 測試集: csv表格中後300天數據
# 利用for循環,遍歷整個測試集,
for i in range(60, len(test_set)):
    x_test.append(test_set[i-60:i, 0]) #提取測試集中連續60天的開盤價作為輸入特征x_train,
    y_test.append(test_set[i, 0]) #第61天的數據作為標籤,for循環共構建300-60=240組數據

#測試集變array並reshape為符合RNN輸入要求:[送入樣本數,循環核時間展開步數, 每個時間步輸入特征個數]
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 60 ,1))

model = tf.keras.Sequential([LSTM(80,return_sequences=True), #記憶體設定80個,每個時間步推送h(t)給下一層
                             Dropout(0.2),
                             LSTM(100), #100個記憶體
                             Dropout(0.2),
                             Dense(1) #輸出值為第61天的開盤價,所以Dense是1
                             ])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss="mean_squared_error") #損失函數用均方誤差

checkpoint_save_path = ".checkpoint/LSTM_stock.ckpt"
if os.path.exists(checkpoint_save_path + ".index"):
    print("-------------- load the model -----------------")
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor="val_loss")

history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=50,
                    validation_data=(x_test, y_test),
                    validation_freq=1,
                    callbacks=[cp_callback])

model.summary()
file = open("./MaoTaiweights.txt", "w", encoding="utf-8")#參數提取
for v in model.trainable_variables:
    file.write(str(v.name) + "\n")
    file.write(str(v.shape) + "\n")
    file.write(str(v.numpy()) + "\n")
file.close()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

#################### predict ###################
# 測試集輸入模型進行預測
predicted_stock_price = model.predict(x_test)
# 對預測數據還原--從(0,1)反歸一化到原始範圍
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# 對真實數據還原--從(0,1)反歸一化到原始範圍
real_stock_price = sc.inverse_transform(test_set[60:])
#畫出真實數據和預測數據的對比曲線
plt.plot(real_stock_price, color="red", label="MaoTai Stock Price") #紅線畫真實值
plt.plot(predicted_stock_price, color="blue", label="Predict MaoTai Stock Price") #藍線畫預測值
plt.title('MaoTai Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel("MaoTai Stock Price")
plt.legend()
plt.show()

############### evaluate #########################評測指標
# calculate MSE -----> E[(預測值-真實值)^2] (預測值減真實值求平方後求均值)
mse = mean_squared_error(predicted_stock_price, real_stock_price)
# calculate RMSE -----> sqrt[MSE]
rmse = math.sqrt(mean_squared_error(predicted_stock_price, real_stock_price))
# calculate MAE平均絕對誤差 -----> E[ |預測值- 真實值| ]  (預測值減真實值求絕對值後求均值)
mae = mean_absolute_error(predicted_stock_price, real_stock_price)
print("均方誤差: %.6f"% mse)
print("均方根誤差: %.6f"% rmse)
print("平均絕對誤差: %.6f"%mae)
# 誤差越小說明預測值和真實值越接近