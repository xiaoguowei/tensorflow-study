import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
#數據類型 tf.int   tf.float
tf.int32
tf.float32
tf.float64

#創建一個張量
a=tf.constant([[[1,5],[2,2],[3,3]],[[1,5],[2,2],[3,3]]],dtype=tf.int64)
print(a)
print(a.dtype)
print(a.shape) #看逗號隔開了幾個數字就是幾維的, 例如:(2,) 1維        (2,3,2) 3維


#將numpy的數據類型轉換為Tensor類型  tf.convert_to_tensor(a,dtype=tf.int64)
a=np.arange(0,5)
b=tf.convert_to_tensor(a,dtype=tf.int64)
print(a)
print(b)

#維度
# 一維 直接寫個數
# 二維 用[行,列]
# 多維 用[n,m,j,k,....]

#生成正態分佈的隨機數,默認均值為0,標準差為1
# tf.random.normal(維度,mean=均值,stddev=標準差)
d=tf.random.normal([2,2],mean=0.5,stddev=1)
print(d)

#生成截斷式正態分佈的隨機數,這個函數可以保證生成的隨機數在 均值±2倍標準差之內 (u±標準差),希望生成的數更集中用這個
# tf.random.truncated_normal(維度,mean=均值,stddev=標準差)
e=tf.random.truncated_normal([2,2],mean=0.5,stddev=1)
print(e)

#生成均勻分佈隨機數
# tf.random.uniform(維度,minval=最小值,maxval=最大值)
f=tf.random.uniform([2,2],minval=0,maxval=1)
print(f)

