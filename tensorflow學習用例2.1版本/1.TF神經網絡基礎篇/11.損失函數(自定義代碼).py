import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

#如預測商品銷量,預測多了,損失成本;預測少了,損失利潤.若 利潤 不等於 成本,則mse產生的loss無法利益最大化
#         { PROFIT * (y_ - y) , y<y_ 預測的y少了,損失利潤(PROFIT)
# f(y_,y)={
#         { COST * (y_ - y) , y>=y_ 預測的y多了,損失利潤(COST)
#預測少了損失大,希望生成的預測函數往多了預測

SEED = 23455
COST = 99 #成本
PROFIT = 1 #利潤


rdm=np.random.RandomState(seed=SEED)
x=rdm.rand(32,2) #生成32行2列輸入特征
y_=[[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1,x2) in x] #生成噪聲(0,1)/10 = (0,0.1); (0,0.1)-0.05 = (-0.05,0.05)
x=tf.cast(x,dtype=tf.float32) #轉換數據類型

w1=tf.Variable(tf.random.normal([2,1],stddev=1,seed=1)) #w1初始化為2行1列

epoch = 10000 #訓練次數
lr = 0.002 #學習率

for epoch in range(epoch):
    with tf.GradientTape() as tape:
        y = tf.matmul(x,w1) #前向傳播計算結果y
        # loss_mse=tf.reduce_mean(tf.square(y_ - y)) # loss均方誤差
        loss = tf.reduce_mean(tf.where(tf.greater(y,y_), (y - y_) * COST, (y_ - y) * PROFIT)) #自定義損失函數

    grads = tape.gradient(loss,w1) #損失函數 對 待訓練w1求偏導
    w1.assign_sub(lr * grads) #更新參數w1

    if epoch % 500 == 0:
        print("After {} training steps".format(epoch))
        print("w1 is {}".format(w1.numpy()))
print("Final w1 is %s" %(w1.numpy))
