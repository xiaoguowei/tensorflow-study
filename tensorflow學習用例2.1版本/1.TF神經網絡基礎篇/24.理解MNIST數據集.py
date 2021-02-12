#提供6萬張 28*28 像素點的0~9手寫數字圖片和標籤,用於訓練
#提供1萬張 28*28 像素點的0~9手寫數字圖片和標籤,用於測試
# 導入數據集:
# mnist = tf.keras.datasets.mnist
# (x_train,y_train), (x_test,y_test) = mnist.load_data()
# 作為輸入特征,輸入輸入神經網絡時,將數據拉伸為一維數組:
# tf.keras.layers.Flatten()
# [0 0 0 0 0 48 248 252 ... .... ... .. 12 0 0 0 0]

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test,y_test) = mnist.load_data()

#可視化訓練集輸入特征的第一個元素
plt.imshow(x_train[0], cmap="gray") #繪製灰度圖
plt.show()

#打印出訓練集輸入特征的一個元素
print("x_train[0]:\n", x_train[0])
#打印出訓練集標籤的一個元素
print("y_train[0]:\n", y_train[0])

#打印出整個訓練集輸入特征形狀
print("x_train.shape:\n",x_train.shape)
#打印出整個訓練集標籤的形狀
print("y_train.shape:\n",y_train.shape)
#打印出整個測試集輸入特征形狀
print("x_test.shape:\n",x_test.shape)
#打印出整個測試集標籤的形狀
print("y_test.shape:\n",y_test.shape)