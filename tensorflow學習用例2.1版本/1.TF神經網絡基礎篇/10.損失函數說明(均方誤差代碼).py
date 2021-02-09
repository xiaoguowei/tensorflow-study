import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow as tf
import numpy as np

# 損失函數(loss):是前向傳播計算出的結果(預測值(y)) 與 已知答案(y_) 的差距
# 神經網絡的優化目標就是找到某套參數計算出來的y 與 已知答案(y_) 無限接近 也就是他們的差距loss值最小
#
# 主流計算loss有三種方法:
# 1.均方誤差(Mean Squared Error)  tf.reduce_mean(tf.square(y_-y))
# 2.自定義
# 3.交叉熵(Cross Entropy)

SEED=23455
rdm=np.random.RandomState(seed=SEED)
x=rdm.rand(32,2) #生成32行2列輸入特征
y_=[[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1,x2) in x] #生成噪聲(0,1)/10 = (0,0.1); (0,0.1)-0.05 = (-0.05,0.05)
x=tf.cast(x,dtype=tf.float32) #轉換數據類型

w1=tf.Variable(tf.random.normal([2,1],stddev=1,seed=1)) #w1初始化為2行1列

epoch = 15000 #訓練次數
lr = 0.002 #學習率

for epoch in range(epoch):
    with tf.GradientTape() as tape:
        y = tf.matmul(x,w1) #前向傳播計算結果y
        loss_mse=tf.reduce_mean(tf.square(y_ - y)) # loss均方誤差

    grads = tape.gradient(loss_mse,w1) #損失函數 對 待訓練w1求偏導
    w1.assign_sub(lr * grads) #更新參數w1

    if epoch % 500 == 0:
        print("After {} training steps,w1 is {}".format(epoch,w1.numpy()))
print("Final w1 is %s" %(w1.numpy))