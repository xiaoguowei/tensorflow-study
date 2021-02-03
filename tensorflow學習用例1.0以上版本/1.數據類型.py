import os
import tensorflow as tf
#消除紅色警告
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
#tf.float32   tf.int32
a=tf.constant([1.0,2.0])
b=tf.constant([3.0,4.0])
result=a+b
print(result)