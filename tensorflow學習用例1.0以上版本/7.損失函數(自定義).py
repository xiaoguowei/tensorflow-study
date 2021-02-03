import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
import numpy as np
# 關閉緊急執行
tf.compat.v1.disable_eager_execution()
BATCH_SIZE=8 #一次餵入神經網絡8組數據
SEED=23455
COST=9 #成本
PROFIT=1 #利潤


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

#定義損失函數,反向傳播方法
#定義損失函數使得預測少了的損失打,於是模型應該偏向多的方向預測
loss=tf.reduce_sum(tf.where(tf.greater(y,y_), (y - y_)*COST, (y_ - y)*PROFIT))#y大於y_輸出(y - y_)*COST,y小於y_輸出(y - y_)*PROFIT
train_step=tf.compat.v1.train.GradientDescentOptimizer(0.001).minimize(loss) #學習率0.001
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