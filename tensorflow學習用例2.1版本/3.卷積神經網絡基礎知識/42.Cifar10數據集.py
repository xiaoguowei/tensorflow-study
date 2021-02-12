#提供5萬張 32*32 像素點的十分類彩色圖片和標籤,用於訓練  紅綠藍(RGB)三通道
#提供1萬張 32*32 像素點的十分類彩色圖片和標籤,用於測試  紅綠藍(RGB)三通道

# 十分類分貝是:
# 0.airplane(飛機),
# 1.automobile(汽車),
# 2.bird(鳥),
# 3.cat(貓),
# 4.deer(鹿),
# 5.dog(狗),
# 6.frog(青蛙),
# 7.horse(馬),
# 8.ship(船),
# 9.truck(卡車)

#導入數據集:
# cifar10 = tf.keras.datasets.cifar10
# (x_train,y_train), (x_test,y_test) = cifar10.load_data()

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(threshold=np.inf)

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test,y_test) = cifar10.load_data()

# 可視化訓練集輸入特征的一個元素
plt.imshow(x_train[0]) #繪製圖片
plt.show()

# 打印出訓練集輸入特征的第一個元素
print("x_train[0]:\n",x_train[0])
# 打印出訓練集標籤的第一個元素
print("y_train[0]:\n",y_train[0])

# 打印出訓練集輸入特征形狀
print("x_train.shape:\n",x_train.shape)
# 打印出訓練集標籤的形狀
print("y_train.shape:\n",y_train.shape)

# 打印出測試集輸入特征形狀
print("x_test.shape:\n",x_test.shape)
# 打印出測試集標籤的形狀
print("y_test.shape:\n",y_test.shape)