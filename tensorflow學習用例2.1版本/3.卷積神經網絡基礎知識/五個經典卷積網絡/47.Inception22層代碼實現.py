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
############################ Inception四個分支概述 ##############################################################
# Inception結構塊在同一層網絡中使用了多個尺寸的卷積核,可以提取不同尺寸的特征
# 通過1*1卷積核,作用到輸入特征圖的每個像素點,
# 通過設定少於輸入特征圖深度的1*1卷積核個數,減少輸出特征圖深度,起到了降維的作用,減少了參數量和計算量
# Inception結構塊包含4個分支如下:
# 第一個分支,經過1*1卷積核輸出到卷積連接器                     # 採用16個1*1卷積核,步長為1,全零填充,採用BN操作,relu激活函數,
# 第二個分支,先經過1*1卷積核再經過3*3卷積核輸出到卷積連接器       # 先用16個1*1卷積核降維,步長為1,全零填充,採用BN操作,relu激活函數.再用16個3*3卷積核,步長為1,全零填充,採用BN操作,relu激活函數
# 第三個分支,先經過1*1卷積核再經過5*5卷積核輸出到卷積連接器       # 先用16個1*1卷積核降維,步長為1,全零填充,採用BN操作,relu激活函數.再用16個5*5卷積核,步長為1,全零填充,採用BN操作,relu激活函數
# 第四個分支,經過3*3最大池化再經過1*1卷積核輸出到卷積連接器       # 先採用最大池化,池化核尺寸是3*3,步長為1,全零填充.再用16個1*1卷積核降維,步長為1,全零填充,採用BN操作,relu激活函數
# 送到卷積連接器的特征數據尺寸相同,卷積連接器會把收到的這4路特征數據按深度方向拼接,形成Inception結構塊的輸出
############################ Inception四個分支概述 ###############################################

class ConvBNRelu(Model):#由於Inception結構塊中的卷積操作均採用了CBA結構,所有將其定義成新的類ConvBNRelu,可以減少代碼長度,增加可讀性
    def __init__(self, ch , kernelsz=3, strides=1, padding="same"): #默認卷積核邊長是3,步長為1,全零填充
        super(ConvBNRelu,self).__init__()
        self.model = tf.keras.models.Sequential([Conv2D(ch, kernelsz, strides=strides, padding=padding),
                                                 BatchNormalization(),
                                                 Activation("relu")])
    def call(self,x):
        x = self.model(x, training=False) #在training=False時,BN通過整個訓練集計算均值,方差去做批歸一化.推理時 training=False效果好
                                          #在training=True時,通過當前batch的均值,方差取做批歸一化.推理時 training=False效果好
        return x

class InceptionBlk(Model):
    def __init__(self, ch, strides=1):
        super(InceptionBlk, self).__init__()
        self.ch = ch
        self.strides = strides
        self.c1 = ConvBNRelu(ch, kernelsz=1, strides=strides) #第1個分支 #使用卷積操作

        self.c2_1 = ConvBNRelu(ch, kernelsz=1, strides=strides) #第2個分支 #使用卷積操作
        self.c2_2 = ConvBNRelu(ch, kernelsz=3, strides=1) #第2個分支 #使用卷積操作

        self.c3_1 = ConvBNRelu(ch, kernelsz=1, strides=strides) #第3個分支 #使用卷積操作
        self.c3_2 = ConvBNRelu(ch, kernelsz=5, strides=1) #第3個分支 #使用卷積操作

        self.p4_1 = MaxPool2D(3, strides=1, padding="same") #第4個分支 #使用最大池化操作
        self.c4_2 = ConvBNRelu(ch, kernelsz=1, strides=strides) #第4個分支 ##使用卷積操作

    def call(self, x):
        x1 = self.c1(x) #第1個分支輸出

        x2_1 = self.c2_1(x)
        x2_2 = self.c2_2(x2_1) #第2個分支輸出

        x3_1 = self.c3_1(x)
        x3_2 = self.c3_2(x3_1) #第3個分支輸出

        x4_1 = self.p4_1(x)
        x4_2 = self.c4_2(x4_1) #第4個分支輸出
        # concat along axis=channel
        x = tf.concat([x1, x2_2, x3_2, x4_2], axis=3) #使用concat將4個分支的輸出堆疊在一起,axis=3指定堆疊的維度是沿深度方向
        return x


class Inception10(Model):
    def __init__(self, num_blocks, num_classes, init_ch=16, **kwargs): #默認輸出深度是16(init_ch=16)
        super(Inception10, self).__init__(**kwargs)
        self.in_channels = init_ch
        self.out_channels = init_ch
        self.num_blocks = num_blocks
        self.init_ch = init_ch
        self.c1 = ConvBNRelu(init_ch)
        self.blocks = tf.keras.models.Sequential()
        for block_id in range(num_blocks):
            for layer_id in range(2): #每2個Inception結構塊組成一個block
                if layer_id == 0:
                    block = InceptionBlk(self.out_channels, strides=2) #每個block中的第一個Inception結構塊,卷積步長是2,令第一個Inception結構塊輸出特征圖尺寸減半
                else:
                    block = InceptionBlk(self.out_channels, strides=1) #第二個Inception結構塊卷積步長是1
                self.blocks.add(block)

            #enlarger out_channels per block
            #block_0設置的通道數是16,經過個四個分支,輸出的深度為 4*16=64
            #在這裡給通道數加倍了,所有block_1通道數是block_0通道數的兩倍是32,同樣經過四個分支輸出的深度是 4*32=128,這128個通道的數據會被送入平均池化,送入10個分類的全連接
            self.out_channels *=2 #第一個Inception結構塊,卷積步長是2,令第一個Inception結構塊輸出特征圖尺寸減半,因此我們把輸出特征圖深度加深,盡可能保證特征提取中信息的承載量一致
        self.p1 = GlobalMaxPooling2D() #全局池化
        self.f1 = Dense(num_classes, activation="softmax") #全連接

    def call(self, x):
        x = self.c1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y

model = Inception10(num_blocks=2, num_classes=10)#num_blocks=2指定了InceptionNet的block數是2(block_0,block_1),num_class=10指定網絡幾分類

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
