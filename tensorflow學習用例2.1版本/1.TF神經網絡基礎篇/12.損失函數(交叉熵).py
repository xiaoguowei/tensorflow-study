import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

# 交叉熵損失函數ce:可以表征兩個概率分佈之間的距離
# 交叉熵越大 兩個概率分佈越遠
# 交叉熵越小 兩個概率分佈越近
# eg: 二分類 已知答案y_=(1,0)表示第一種發生的概率為1,第二種發生的概率為0,
#     神經網絡預測y1=(0.6,0.4) 表示第一種發生的概率為0.6,第二種發生的概率為0.4
#     y2=(0.8,0.2) 表示第一種發生的概率為0.8,第二種發生的概率為0.2
# 公式計算 tf.losses.categorical_crossentropy(y_, y)
# H1((1,0),(0.6,0.4)) = -(1*ln0.6 + 0*ln0.4 ) →→ -(-0.511 + 0) = 0.511
# H2((1,0),(0.8,0.2)) = -(1*ln0.8 + 0*ln0.2 ) →→ -(-0.223 + 0) = 0.223
# 因為H1 > H2,所以y2預測更準
# 代碼版如下
# loss_ce1=tf.losses.categorical_crossentropy([1,0],[0.6,0.4])
# loss_ce2=tf.losses.categorical_crossentropy([1,0],[0.8,0.2])
# print("loss_ce1:",loss_ce1)
# print("loss_ce2:",loss_ce2)


# 在執行分類問題時:
# 1.通常先用softmax函數讓輸出結果符合概率分佈
# 2.在計算y與y_的交叉熵損失函數
# 同時計算分佈和交叉熵的函數: tf.nn.softmax_cross_entropy_with_logits(y_, y)
y_=np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0]])
y =np.array([[12,3,2],[3,10,1],[1,2,5],[4,6.5,1.2],[3,6,1]])
y_pro=tf.nn.softmax(y)
loss_ce1 = tf.losses.categorical_crossentropy(y_ , y_pro)
loss_ce2 = tf.nn.softmax_cross_entropy_with_logits(y_ , y) #同時計算概率分佈和交叉熵
print("分步計算的結果:\n ",loss_ce1)
print("結合計算的結果:\n",loss_ce2)
