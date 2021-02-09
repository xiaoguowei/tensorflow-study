# 指數衰減學習概率:可以先用較大的學習率,快速得到較優解,然後逐步減小學習率,使模型在訓練後穩定
#
# 指數衰減學習率 = 初始學習率 * 學習率衰減率**(當前輪數 / 多少輪衰減一次)
# lr          = LR_BASE  * LR_DECAY**(epoch / LR_STEP)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
epoch = 40
LR_BASE = 0.2 #初始學習率
LR_DECAY = 0.99
LR_STEP = 1
w=tf.Variable(tf.constant(5,dtype=tf.float32)) #設定w參數的隨機初始值為5,設定為可訓練

for epoch in range(epoch):
    lr = LR_BASE  * LR_DECAY**(epoch / LR_STEP) #指數衰減一般寫在這
    with tf.GradientTape() as tape: #with結構到grads框起了梯度的計算過程
        loss=tf.square(w + 1)       #損失函數定義為 (w + 1)的平方(square)
    grads = tape.gradient(loss,w)   #.gradient函數告知誰對誰求導,這裡是讓損失函數對參數w求梯度

    w.assign_sub(lr * grads)        #.assign_sub函數 對變量做自檢 既: w = w - lr * grads
    print("After {} epoch,w is {},loss is {},lr is {}".format(epoch, w.numpy(), loss,lr))
