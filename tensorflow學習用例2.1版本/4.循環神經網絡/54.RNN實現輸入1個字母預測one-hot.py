#!/usr/bin/python
# -*- coding: utf-8 -*-
#字母預測:輸入a預測出b, 輸入b預測出c, 輸入c預測出d, 輸入d預測出e, 輸入e預測出a
#神經網絡的輸入都是數字,所有我們先把abcde五個字母用數字表示出來
# 用獨熱碼(one-hot)對他們編碼如下:
# a 100000
# b 010000
# c 000100
# d 000010
# e 000001
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNN
import matplotlib.pyplot as plt
import os
####################### 配置GPU內存 ###############################################
gpus = tf.config.experimental.list_physical_devices('GPU')
GPU=1024*5 #配置多少GPU
tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU)])
########################## 配置GPU內存 ###############################################
input_word = "abcde"
w_to_id = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, } #單詞映射到數值id的詞典 #把字母表示為數字
id_to_onehot = {0:[1., 0., 0., 0., 0.], #數字編碼為獨熱碼
                1:[0., 1., 0., 0., 0.],
                2:[0., 0., 1., 0., 0.],
                3:[1., 0., 0., 1., 0.],
                4:[0., 0., 0., 0., 1.]}

x_train = [id_to_onehot[w_to_id['a']], id_to_onehot[w_to_id['b']], id_to_onehot[w_to_id['c']], id_to_onehot[w_to_id['d']],id_to_onehot[w_to_id['e']]]
y_train = [w_to_id['b'], w_to_id['c'], w_to_id['d'], w_to_id['e'],w_to_id['a']] #[1,2,3,4,0]

np.random.seed(7)
np.random.shuffle(x_train) #打亂順序
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

# 使x_train符合SimpleRNN輸入要求:[送入樣本, 循環核時間展開步數, 每個時間步輸入特征個數]
# 此處整個數據集送入,送入樣本數為len(x_train);輸入1個字母出結果,循環核時間展開步數為1;表示為獨熱碼有5個輸入特征,每個時間步輸入特征個數為5
x_train = np.reshape(x_train, (len(x_train), 1, 5))
y_train = np.array(y_train)

model = tf.keras.Sequential([SimpleRNN(3), #3個記憶體的循環層,可自行調整個數,記憶體個數越多,記憶力越好,但是佔用資源會更多
                             Dense(5, activation="softmax")]) #一層全連接,實現輸出層y(t)的計算.由於要映射到獨熱碼編碼,找到輸出概率最大的字母,所有這裡是5

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/rnn_onehot_1pre1.clpt"

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
    alphabet1 = input("input test alphabet:") #等待輸入 一個字母
    alphabet = [id_to_onehot[w_to_id[alphabet1]]] #把輸入的字母轉換為獨熱碼
    #使alphabet符合SimpleRNN輸入要求:[送入樣本數, 循環核時間展開步數, 每個時間步輸入特征個數].
    # 此處驗證效果送入了1個樣本,送入樣本數為1;輸入1個字母出結果,所有循環時間展開數步數為1;表示為獨熱碼有5個輸入特征,每個時間步輸入特征個數為5
    alphabet = np.reshape(alphabet, (1, 1, 5))
    result = model.predict([alphabet]) #預測結果
    pred = tf.argmax(result, axis=1) #選出預測結果最大的一個
    pred = int(pred)
    tf.print(alphabet1 + "->"+ input_word[pred])