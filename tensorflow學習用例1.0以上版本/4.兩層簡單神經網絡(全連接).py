import os
#關閉紅色警告
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
# 關閉緊急執行
tf.compat.v1.disable_eager_execution()

#定義輸入和參數
# # #用placeholder實現輸入定義 (see.run中餵一組數據)
# x=tf.compat.v1.placeholder(tf.float32,shape=(1,2))#數據類型用float32表示,1行2列

#用placeholder實現輸入定義 (see.run中餵多組數據)
x=tf.compat.v1.placeholder(tf.float32,shape=(None,2))#數據類型用float32表示,不知道幾行知道2列
w1=tf.Variable(tf.compat.v1.random_normal(shape=(2,3),stddev=1,seed=1))#2行三列正態分佈隨機數
w2=tf.Variable(tf.compat.v1.random_normal(shape=(3,1),stddev=1,seed=1))

#定義前向傳播過程
a=tf.matmul(x,w1)#隱藏層a
y=tf.matmul(a,w2)



#用會話計算結果
with tf.compat.v1.Session() as sess:
    init_op=tf.compat.v1.global_variables_initializer()
    sess.run(init_op)
    # print(sess.run(y, feed_dict={x: [[0.7, 0.8]]}))
    print(sess.run(y,feed_dict={x:[[0.7,0.8],[0.2,0.3],[0.3,0.4],[0.4,0.5]]}))
    print('-'*15)
    print(sess.run(w1))
    print('-' * 15)
    print(sess.run(w2))