import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf

#定義輸入的參數
x=tf.constant([[0.7,0.5]])#1行2列
w1=tf.Variable(tf.random_normal_initializer([2,3],stddev=1,seed=1))#生成2行3列標準差為1的正態分佈矩陣
w2=tf.Variable(tf.random_normal_initializer([3,1],stddev=1,seed=1))#生成3行1列標準差為1的正態分佈矩陣

#定義前向傳播過程
a=tf.matmul(x,w1)
y=tf.matmul(x,w2)

#用會話計算結果
with tf.compat.v1.Session() as sess:
    init_op=tf.compat.v1.glorot_normal_initializer()#返回一個初始化全局變量
    sess.run(init_op)
    print(sess.run(y))