#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import  Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(threshold=np.inf)
####################### 配置GPU內存 ###############################################
gpus = tf.config.experimental.list_physical_devices('GPU')
GPU=1024*5.6 #配置多少GPU
tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU)])
########################## 配置GPU內存 ###############################################(e)

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 #歸一化,讓數值介於0~1

class   AlexNet8(Model): #共8層
    def __init__(self):
        super(AlexNet8,self).__init__()
        # 第一層
        self.c1 = Conv2D(filters=96, kernel_size=(3, 3)) #C #96個3*3的卷積核,步長是1(默認)
        self.b1 = BatchNormalization() #批標準化 #原論文使用的是局部響應標準化LRN,由於LRN操作近些年用得很少,他的功能與披標準化BN相似,所里這裡選擇主流的BN
        self.a1 = Activation("relu")
        self.p1 = MaxPool2D(pool_size=(3, 3), strides=2) #P #使用最大池化,池化核是3*3,步長為2,默認不使用全零填充
        # 第二層
        self.c2 = Conv2D(filters=256, kernel_size=(3, 3)) #C #256個3*3的卷積核,步長是1(默認),
        self.b2 = BatchNormalization() #批標準化
        self.a2 = Activation("relu")
        self.p2 = MaxPool2D(pool_size=(3, 3), strides=2) #P #使用最大池化,池化核是3*3,步長為2,默認不使用全零填充
        # 第三層
        self.c3 = Conv2D(filters=384, kernel_size=(3, 3), padding="same", activation="relu") #C #384個3*3的卷積核,步長是1(默認),使用全零填充
        # 第四層
        self.c4 = Conv2D(filters=384, kernel_size=(3, 3), padding="same", activation="relu") #C #384個3*3的卷積核,步長是1(默認),使用全零填充
        # 第五層
        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu") #C #256個3*3的卷積核,步長是1(默認),使用全零填充

        self.p3 = MaxPool2D(pool_size=(3, 3), strides=2) #P #使用最大池化,池化核是3*3,步長為2,以默認不使用全零填充

        # 第六層
        self.flatten = Flatten() #把卷積送來的數據拉直
        self.f1 = Dense(2048, activation="relu") #送入2048個神經元的全連接,激活函數為 "relu"
        self.d1 = Dropout(0.5) #過0.5的Dropout
        # 第七層
        self.f2 = Dense(2048, activation="relu") #送入2048個神經元的全連接,活函數為 "relu"
        self.d2 = Dropout(0.5) #過0.5的Dropout
        # 第八層
        self.f3 = Dense(10, activation="softmax") #過10個神經元的全連接,過softmax函數使輸出符合概率分佈  #因為輸出是10分類所有用10個神經元

    def call(self,x):
        # 第一層
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        # 第二層
        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)
        # 第三層
        x = self.c3(x)
        # 第四層
        x = self.c4(x)
        # 第五層
        x = self.c5(x)
        x = self.p3(x)
        # 3層全連接網絡
        # 第六層
        x = self.flatten(x)
        x = self.f1(x)
        x = self.d1(x)
        # 第七層
        x = self.f2(x)
        x = self.d2(x)
        # 第八層
        y = self.f3(x)  #預測結果
        return y

model = AlexNet8()
model.compile(optimizer="adam", #優化器
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), #選擇損失函數
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/Baseline.ckpt"
if os.path.exists(checkpoint_save_path + "index"):
    print("-----------------load the model----------------------")
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath= checkpoint_save_path,#路徑文件
                                                 save_weights_only=True, #是否只保留模型參數
                                                 save_best_only=True) #是否只保留最優結果(模型)

history = model.fit(x_train, y_train, #定義訓練集
                    batch_size=32, #一次餵入32筆數據(2**5)→(2**n)
                    epochs= 5, #迭代5次數據集
                    validation_data=(x_test, y_test), #定義測試集
                    validation_freq=1, #1次數據迭代用測試集驗證準確率
                    callbacks=[cp_callback] #回調函數,斷電續訓
                    )
model.summary() #打印網絡結構和參數

# print(model.trainable_variables)
file = open("./weights.txt", "w",encoding="utf-8")
for v in model.trainable_variables:
    file.write(str(v.name) + "\n")
    file.write(str(v.shape) + "\n")
    file.write(str(v.numpy()) + "\n")
file.close()

###################################### show ######################################

#顯示訓練集和驗證集的acc和loss曲線
acc = history.history['sparse_categorical_accuracy'] #提取訓練集準確率
val_acc = history.history['val_sparse_categorical_accuracy'] #提取測試集準確率
loss = history.history['loss'] #訓練集損失函數數值
val_loss = history.history['val_loss'] #測試集損失函數數值

plt.subplot(1,2,1) #將圖像分為1行2列,這段代碼畫出第1列
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy') #標題
plt.legend()

plt.subplot(1,2,2) #將圖像分為1行2列,這段代碼畫出第2列
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
