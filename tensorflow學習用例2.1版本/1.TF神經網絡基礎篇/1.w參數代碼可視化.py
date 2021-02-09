import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

w=tf.Variable(tf.constant(5,dtype=tf.float32)) #設定w參數的隨機初始值為5,設定為可訓練
lr = 0.2 #學習率
epoch = 40 #循環迭代40次

for epoch in range(epoch):
    with tf.GradientTape() as tape: #with結構到grads框起了梯度的計算過程
        loss=tf.square(w + 1)       #損失函數定義為 (w + 1)的平方(square)
    grads = tape.gradient(loss,w)   #.gradient函數告知誰對誰求導,這裡是讓損失函數對參數w求梯度

    w.assign_sub(lr * grads)        #.assign_sub函數 對變量做自檢 既: w = w - lr * grads
    print("After {} epoch,w is {},loss is {}".format(epoch, w.numpy(), loss))