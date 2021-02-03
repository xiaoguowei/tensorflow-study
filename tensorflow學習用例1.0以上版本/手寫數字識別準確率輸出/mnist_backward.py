import tensorflow as tf
from tensorflow.keras.datasets import mnist
import mnist_forward
import os

BATCH_SIZE=200  #定義每輪餵入200張圖片
LEARNING_RATE_BASE=0.1 #學習率
LEARNING_RATE_DECAY=0.99 #衰減率
REGULARIZER=0.001 #正則化係數
STEPS=50000 #訓練多少輪
MOVING_AVERAGE_DECAY=0.99 #滑動平均衰減率
MODEL_SAVE_PATH="./model/" #模型保存路徑
MODEL_NAME="mnist_model" #模型保存名字

def backword(mnist):
    # 關閉緊急執行
    tf.compat.v1.disable_eager_execution()
    x = tf.compat.v1.placeholder(tf.float32,mnist_forward.INPUT_NODE)#佔位
    y_= tf.compat.v1.placeholder(tf.float32,mnist_forward.OUTPUT_NODE)
    y = mnist_forward.forward(x,REGULARIZER) #調用前向傳播,計算輸出y
    global_step = tf.Variable(0,trainable=False) #賦初值,設定為不可訓練

    ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_ , 1)) #
    cem=tf.reduce_mean(ce)
    loss=cem+tf.add_n(tf.compat.v1.get_collection("losses")) #調用包含正則化得損失函數loss

    #定義指數衰減學習率$$$$$$$$$$$$$$$$$$$$
    learning_rate=tf.compat.v1.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.nun_examples / BATCH_SIZE,staircase=True)


    #定義反向傳播方法:包含正則化
    train_step=tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    ema = tf.compat.v1.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op = ema.apply(tf.compat.v1.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op=tf.no_op(name='train')

    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        init_op = tf.compat.v1.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE) ##
            _,loss_value,step = sess.run([train_op,loss,global_step], feed_dict={x : xs,y_: ys})
            if i % 1000 == 0:
                print("",step,loss_value)
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main():
    mnist=mnist.lo
    backword(mnist)

if __name__ == '__main__':
    main()