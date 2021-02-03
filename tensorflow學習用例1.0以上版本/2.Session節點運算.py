import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf

with tf.compat.v1.Session() as sess:
    x = tf.constant([[1.0, 2.0]])
    w = tf.constant([[[3.0], [4.0]]])

    y = tf.matmul(x, w)
    print(y)
    print(sess.run(y))
