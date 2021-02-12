#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import  Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense,GlobalMaxPooling2D
from tensorflow.keras import Model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(threshold=np.inf)
####################### 配置GPU內存 ###############################################
gpus = tf.config.experimental.list_physical_devices('GPU')
GPU=1024*5 #配置多少GPU
tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU)])
########################## 配置GPU內存 ###############################################

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 #歸一化,讓數值介於0~1
############################ ResNet概述 ##############################################################
# LeNet 5層
# AlexNet 8層
# VGGNet 16&19層
# InceptionNet v1 22層
# 由上可見人們在探索卷積實現特征提取的道路上,通過加深網絡層數,取得越來越好的效果
# ResNet的作者在cifar10數據集上做了個實驗,發現56層卷積網絡的錯誤率 要高於 20層卷積網絡的錯誤率,認為單純堆疊神經網絡層數會使神經網絡模型退化,
# 以至於後邊的特征丟失了前邊特征的原本模樣,於是他用了一根跳連線,將前邊的特征直接 接到了後邊.
# ResNet塊中的"+"是特征圖對應元素值相加(矩陣相加)
# ResNet有2中情況:
# 1.用圖中的實線表示:這種情況2層堆疊卷積沒有改變特征圖的維度,也就是他們特征圖的個數(高,寬,深度)都相同,計算方式為: H(x) = F(x) + x
# 2.用圖中的虛線表示:這種情況2層堆疊卷積改變了特征圖的維度,需要藉助1*1的卷積來調整x的維度,使W(x)於F(x)的維度一致,計算方式為:H(x) = f(x) + W(x)
############################ ResNet概述 ###############################################

class ResnetBlock(Model): #處理維度相同和不相同的輸出特征圖
    def __init__(self, filters , strides=1, residual_path=False):
        super(ResnetBlock,self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters, (3,3), strides=strides, padding="same", use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation("relu")

        self.c2 = Conv2D(filters, (3,3), strides=1, padding="same", use_bias=False)
        self.b2 = BatchNormalization()

        if residual_path:#residual_path為True(為1)時調用以下操作.對輸入進行下採樣,即用1*1的卷積核做卷積操作,保證x能和F(x)維度響度,順利相加
            self.down_c1 = Conv2D(filters, (1,1), strides=strides, padding="same", use_bias="False") #使用1*1卷積操作,調整輸入特征圖inputs的尺寸或深度後
            self.down_b1 = BatchNormalization()

        self.a2 = Activation("relu")


    def call(self,inputs):
        residual = inputs #residual等於輸入值本身,即residual=x
        # 將輸入通過卷積,BN層,激活層,計算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        # 如果堆疊卷積層前後維度相同,就不指定下面的if語句,直接將堆疊卷積輸出特征y和輸入特征圖inputs相加
        if self.residual_path:#如果堆疊卷積層前後維度不同就會調用以下操作,residual_path就等於1
            residual = self.down_c1(inputs) #使用1*1卷積操作,調整輸入特征圖inputs的尺寸或深度後,將堆疊間距輸出特征y 與 if語句計算出的residual相加
            residual = self.down_b1(residual)

        out = self.a2(y + residual) #最後輸出的兩部分的和,即F(x)或F(x) + Wx,再過激活函數
        return out

class ResNet18(Model): #搭建神經網絡
    def __init__(self, block_list, initial_filters=64): #默認輸出深度是64
        super(ResNet18, self).__init__()
        self.num_blocks = len(block_list) #共有幾個block
        self.block_list = block_list
        self.out_filters = initial_filters
        # 第一層卷積
        self.c1 = Conv2D(self.out_filters, (3,3), strides=1, padding="same", use_bias=False) #採用64個3*3的卷積核,步長為1,全零填充
        self.b1 = BatchNormalization() #批標準化BN
        self.a1 = Activation("relu") #relu激活函數
        self.blocks = tf.keras.models.Sequential()

        # 構建8個ResNet網絡結構,有8個ResNet塊,每一個ResNet塊有2層卷積,一共是18層網絡
        # 第一個ResNet塊是2個實線跳連的ResNet塊,實線計算公式為: H(x) = F(x) + x
        # 第二,三,四ResNet塊是先虛線再實線的ResNet塊,虛線計算公式為:H(x) = F(x) + W(x)
        for block_id in range(len(block_list)): #第幾個resnet block #循環次數由參數列表元素個數決定
            for layer_id in range(block_list[block_id]): #第幾個卷積層
                if  block_id != 0 and layer_id == 0: #對 除了第一個blokc以外的每個block的輸入進行下採樣
                    block = ResnetBlock(self.out_filters, strides=2, residual_path=True) #residual_path=True用虛線連接
                else:
                    block = ResnetBlock(self.out_filters, residual_path=False) #residual_path=False用實線連接
                self.blocks.add(block) #將構建好的block加入resnet

            self.out_filters *=2 #下一個block的卷積核數是上一個block的2倍
        self.p1 = GlobalMaxPooling2D() #平均全局池化
        self.f1 = Dense(10, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2()) #全連接

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y

model = ResNet18([2, 2, 2, 2]) #4個元素會循環4次

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
                    batch_size=128, #一次餵入128筆數據(2**7)→(2**n)
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
