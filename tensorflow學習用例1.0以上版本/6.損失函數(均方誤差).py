#前向傳播搭建網絡結構
#後向傳播訓練網絡參數

#激活函數有三種
# 1.relu
# 2.sigmoid
# 3.tanh

#NN複雜度:多用NN層數和NN參數的個數表示
#層數=隱藏層的層數 + 1個輸出層
##################################################

#損失函數(loss):預測值 與 已知答案的差距
#NN優化目標為loss最小
#主流的loss計算方法有三種如下
#1.均方誤差
#2.自定義
#3.交叉熵

#在所有參數上用梯度下降的方法,使訓練集數據上的損失函數最小
# 損失函數(loss):預測(y) 與 已知答案(y_) 的差距
# 方法一 均方誤差 loss=tf.reduce_mean(tf.square(y-y_))

# 方法二 反向傳播訓練: 以減少loss值威優化目標  有三種可選如下👇 訓練選擇其一即可
# 1. train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
# 2. train_step=tf.train.MomentumOptimizer(learning_rate,momentum).minimize(loss)
# 3. train_step=tf.train.AdamOptimizer(learning_rate,momentum).minimize(loss)
#都有一個learning_rate(學習率:決定參數每次更新的幅度)的參數

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
import numpy as np
# 關閉緊急執行
tf.compat.v1.disable_eager_execution()
BATCH_SIZE=8 #一次餵入神經網絡8組數據
SEED=23455

#基於seed產生隨機數
rdm=np.random.RandomState(SEED)#保證每次隨機生存成的數字一樣

#隨機數返回32行2列的矩陣 表示32組 2個特征(體積和重量) 作為輸入數據集
X=rdm.rand(32,2)
Y=[[x1+x2+(rdm.rand()/10.0-0.05)] for (x1,x2) in X]

#定義神經網絡的輸入,參數和輸出,定義前向傳播過程
x=tf.compat.v1.placeholder(tf.float32,shape=(None,2)) #輸入數據佔位
y_=tf.compat.v1.placeholder(tf.float32,shape=(None,1)) #標準答案數據佔位
w1=tf.Variable(tf.compat.v1.random_normal(shape=(2,1),stddev=1,seed=1)) #2個特征,1個輸入
y=tf.matmul(x,w1) #乘法矩陣

#定義損失函數為MSE,反向傳播方法為梯度下降
loss_mse=tf.reduce_mean(tf.square(y_-y))#均方誤差計算loss
train_step=tf.compat.v1.train.GradientDescentOptimizer(0.001).minimize(loss_mse) #學習率0.001
# train_step=tf.compat.v1.train.MomentumOptimizer(0.001,0.9).minimize(loss)
# train_step=tf.compat.v1.train.AdamOptimizer(0.001).minimize(loss)

#生存會話,訓練STEPS輪
with tf.compat.v1.Session() as sess:
    #初始化
    init_op=tf.compat.v1.global_variables_initializer()
    sess.run(init_op)

    #訓練模型
    STEPS=20000
    for i in range(STEPS):
        strat=(i*BATCH_SIZE) % 32 #因為只有32組數據所以%32
        end=(i*BATCH_SIZE) % 32 +BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[strat:end],y_:Y[strat:end]})
        if i % 500  == 0:
            print("第{0}次訓練".format(i))
            print("w1為:",sess.run(w1))
    print("最後的結果為:",sess.run(w1))