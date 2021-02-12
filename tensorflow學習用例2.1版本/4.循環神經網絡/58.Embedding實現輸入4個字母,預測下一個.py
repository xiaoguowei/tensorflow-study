#!/usr/bin/python
# -*- coding: utf-8 -*-
#神經網絡的輸入都是數字,所有我們先把abcde五個字母用數字表示出來
# 用獨熱碼(one-hot)對他們編碼如下:
# a 100000
# b 010000
# c 000100
# d 000010
# e 000001
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding
import matplotlib.pyplot as plt
import os
####################### 配置GPU內存 ###############################################
gpus = tf.config.experimental.list_physical_devices('GPU')
GPU=1024*5 #配置多少GPU
tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU)])
########################## 配置GPU內存 ###############################################
input_word = "abcdefghijklmnopqrstuvwxyz"
w_to_id = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 'j':9, 'k':10, 'l':11, 'm':12, 'n':13,
           'o':14, 'p':15, 'q':16, 'r':17, 's':18, 't':19, 'u':20, 'v':21, 'w':22, 'x':23, 'y':24, 'z':25
           } #單詞映射到數值id的詞典 #把字母表示為數字

train_set_scaled = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 23, 25]

x_train = []
y_train = []

for i in range(4,26):
    x_train.append(train_set_scaled[i - 4 :i]) #[0:4],[1:5].[2,6] #把連續4個數作為輸入特征添加到x_train
    y_train.append(train_set_scaled[i]) #[4,5,6] #第5個數作為標籤添加到y_train


np.random.seed(7)
np.random.shuffle(x_train) #打亂順序
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

# 使x_train符合Embedding輸入要求:[送入樣本, 循環核時間展開步數, 每個時間步輸入特征個數]
# 此處整個數據集送入,送入樣本數為len(x_train);輸入4個字母出結果,循環核時間展開步數為4;
x_train = np.reshape(x_train, (len(x_train), 4)) #22個數據集,因為連續輸入4個字母得到輸出所以這裡是循環核步長為4
y_train = np.array(y_train)

model = tf.keras.Sequential([Embedding(26,2), #詞匯量26,每個單詞用2個數值編碼,生成26行2列的可訓練參數矩陣 實現編碼可訓練
                             SimpleRNN(10), #10個記憶體的循環層,可自行調整個數,記憶體個數越多,記憶力越好,但是佔用資源會更多
                             Dense(26, activation="softmax")]) #全連接層,實現輸出層y(t)的計算,輸出會是26個字母之一,所以這裡是26

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/rnn_embedding_4pre1.clpt"

if os.path.exists(checkpoint_save_path + ".index"):
    print("------------------ load the model -------------------")
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_best_only=True,
                                                 save_weights_only=True,
                                                 monitor="loss") #由於fit沒有給出測試集,不計算測試集準確率,根據loss,保存最優模型

history = model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=[cp_callback])
model.summary()

file = open("./weights.txt","w",encoding="utf-8")
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

######################################### show ######################################
#顯示訓練集和驗證集的acc和loss曲線
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label="Training Accuracy")
plt.title("Training Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label="Training Loss")
plt.title("Training Loss")
plt.legend()
plt.show()

################ predict ##################展示效果
preNum = int(input("input the number of test alphabet:")) #輸入要執行幾次
for i in range(preNum):
    alphabet1 = input("input test alphabet:") #等待連續輸入4個字母
    alphabet = [[w_to_id[a]] for a in alphabet1] #把輸入的數字轉換為int
    # 使alphabet符合Embedding輸入要求:[送入樣本數, 循環核時間展開步數].
    # 此處驗證效果送入了1個樣本,送入樣本數為1;輸入1個字母出結果,所有循環時間展開數步數為4;
    alphabet = np.reshape(alphabet, (1, 4))
    result = model.predict([alphabet]) #預測結果
    pred = tf.argmax(result, axis=1) #選出預測結果最大的一個
    pred = int(pred)
    tf.print(alphabet1 + "->"+ input_word[pred])