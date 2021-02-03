import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import forward
import generateds

STEPS=40000
BATCH_SIZE=30
LEARNING_RATE_BASE=0.001
LEARNING_RATE_DECAY=0.999
REGULARIZER=0.01

def backward():
    # 關閉緊急執行
    tf.compat.v1.disable_eager_execution()

    x=tf.compat.v1.placeholder(tf.float32,shape=(None,2))
    y_=tf.compat.v1.placeholder(tf.float32,shape=(None,1))

    X,Y_,Y_c=generateds.generateds()
    # 復現神經網絡結構,推測出y
    y=forward.forward(x,REGULARIZER)

    global_step=tf.Variable(0,trainable=False) #

    #定義指數衰減學習率
    learning_rate=tf.compat.v1.train.exponential_decay(LEARNING_RATE_BASE,global_step,300/BATCH_SIZE,LEARNING_RATE_DECAY,staircase=True)

    #定義損失函數
    loss_mse=tf.reduce_mean(tf.square(y-y_))
    loss_total=loss_mse + tf.add_n(tf.compat.v1.get_collection("losses"))

    #定義反向傳播方法:包含正則化
    train_step=tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_total)

    with tf.compat.v1.Session() as sess:
        init_op=tf.compat.v1.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            start=(i%BATCH_SIZE) % 300
            end=start + BATCH_SIZE
            sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
            if i %2000 == 0 :
                loss_v=sess.run(loss_total,feed_dict={x:X,y_:Y_})
                print("第{0}次訓練,loss為:{1}".format(i,loss_v))
        xx,yy=np.mgrid[-3:3:.01,-3:3:.01]
        grid=np.c_[xx.ravel(),yy.ravel()]
        probs=sess.run(y,feed_dict={x:grid})
        probs=probs.reshape(xx.shape)

    plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
    plt.contour(xx,yy,probs,levels=[.5])
    plt.show()

if __name__ == '__main__':
    backward()

