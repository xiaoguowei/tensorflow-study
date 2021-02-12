import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import  Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model

####################### 配置GPU內存 ###############################################
gpus = tf.config.experimental.list_physical_devices('GPU')
GPU=1024*5.5 #配置多少GPU
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
########################## 配置GPU內存 ###############################################

np.set_printoptions(threshold=np.inf)

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 #歸一化,讓數值介於0~1

class BaseLine(Model):
    def __init__(self):
        super(BaseLine,self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), padding="same") #C #6個5*5的卷積核,步長是1(默認),使用全零填充
        self.b1 = BatchNormalization() #B 使用披標準化
        self.a1 = Activation("relu") #A 使用relu激活函數
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same") #P #使用最大池化,池化核是2*2,步長為2,使用全零填充
        self.d1 = Dropout(0.2) #D #把20%的神經元休眠

        self.flatten = Flatten() #把卷積送來的數據拉直
        self.f1 = Dense(128, activation="relu") #送入128個神經元的全連接,過relu激活函數
        self.d2 = Dropout(0.2) #過0.2的Dropout
        self.f2 = Dense(10, activation="softmax") #過10個神經元的全連接,過softmax函數使輸出符合概率分佈  #因為輸出是10分類所有用10個神經元

    def call(self,x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)  #預測結果
        return y

model = BaseLine()
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
file = open("./weights.txt", "w")
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
plt.show()

plt.subplot(1,2,2) #將圖像分為1行2列,這段代碼畫出第2列
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
