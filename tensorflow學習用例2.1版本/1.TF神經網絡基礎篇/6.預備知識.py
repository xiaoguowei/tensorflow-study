import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
#預備知識
#tf.where(條件語句,真返回A,假返回b)
# a=tf.constant([1,2,3,1,1])
# b=tf.constant([0,1,3,4,5])
# c=tf.where(tf.greater(a,b),a,b) #若a>b,返回a對應位置的元素.否則返回b對應位置的元素
# print(c)


#返回一個[0,1]之間的隨機數
# np.random.RandomState.rand(維度) #維度為空,返回標量
# rdm=np.random.RandomState(seed=1) #seed=常數,每次生成隨機數相同
# a=rdm.rand() #返回一個隨機標量
# b=rdm.rand(2,3) #返回維度為2行3列隨機數矩陣
# print(a)
# print(b)


#將2個數組按垂直方向疊加
# np.vstack(數組1,數組2)
# a=np.array([1,2,3])
# b=np.array([4,5,6])
# c=np.vstack((a,b)) #變成二維數組,縱向疊加
# print(c)


#np.mgrid[起始值 : 結束值 : 步長,起始值 : 結束值 : 步長,.....]
#np.mgrid[第一維,第二維,第三維,.....7]
#x.ravel() 將x變為一位數組,把"把 . 前變量拉直" 把多維數組變成一位數組
#np.c_[數組1,數組2,.....] 使返回的間隔數值點配對
x,y=np.mgrid[1:3:1,2:4:0.5]
grid=np.c_[x.ravel(),y.ravel()]
print(x)
print(y)
print(grid)


