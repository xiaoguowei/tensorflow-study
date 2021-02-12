import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

#待優化的參數w,損失函數loss,學習率lr,每次迭代一個batch,t表示當前batch迭代的總次數:
# 優化步驟如下: #一階動量:與梯度相關的函數      二階動量:與梯度平方相關的函數
# 1.計算t時刻損失函數 關於當前參數的梯度gt =△loss=loss/wt(也就是讓loss對當前的每個w求偏執導數)
# 2.計算t時刻一階動量mt 和 二階動力Vt
# 3.計算t時刻下降梯度: nt = lr * (mt / (vt開根號))  #學習概 * 一階動量 / 二階動量開根號
# 4.計算t+1時刻參數: w(t+1) = wt - nt = wt - lr * (mt / (vt開根號))  # 當前時刻的參數 - 學習率 * 當前時刻的一階動量 / 當前時刻的二階動量開根號

#五種常用的優化器
# 1.sgd
# 2.sgdm
# 3.adagrad
# 4.rmsprop
# 5.adam

