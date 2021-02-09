# 欠擬合的解決方法:
# 1.增加輸入特征項
# 2.增加網絡參數
# 3.減少正則化參數
#
# 過擬合的解決方法:
# 1.數據清洗
# 2.增大訓練集
# 3.採用正則化
# 4.增大正則化參數

# 正則化緩解過擬合
# 正則化就是在損失函數中引入模型複雜度指標,利用給w加權值,弱化訓練數據的噪聲(一般不正則化b)
# loss=loss(y 與 y_)         +     REGULARIZER * loss(w)  #loss(x)表示正則化括號中的參數,這裡是把w正則化
#         ↑↑                         ↑↑
#    {模型中所有參數的損失函數         {用超參數regularizer給出參數
#     如:交叉熵,均方誤差}            w在總loss中的比例,即正則化的權重}

# 正則化有2種選擇:
# 1. L1(對所有w的絕對值求和)正則化大概率會使很多參數變為零,因此該方法可通過稀疏參數,即減少參數的數量,降低複雜度
# 2. L2(對所有w的平方的絕對值求和)正則化會使參數很接近零但不為零,因此該方法可通過減少參數值的大小降低複雜度

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#一下代碼不含正則化,包含正則化的代碼在14.
#讀入數據
df = pd.read_csv("dot.csv")
x_data = np.array(df[["x1","x2"]])
y_data = np.array(df["y_c"])

x_train = np.vstack(x_data).reshape(-1,2)
y_train = np.vstack(y_data).reshape(-1,1)
Y_c = [["red" if y else "blue"] for y in y_train]

#轉換x的數據類型,否則後面矩陣相乘時會因數據類型問題報錯
x_train = tf.cast(x_train,tf.float32)
y_train = tf.cast(y_train,tf.float32)

#from_tensor_slices函數切分傳入的張量的第一個維度,生成相應的數據集,使輸入特征和標籤值一一對應
train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)

#生成神經網絡的參數,輸入層為2個神經元,隱藏層為11個神經元,1個層隱藏層,輸出層為1個神經元
w1 = tf.Variable(tf.random.normal([2,11]),dtype=tf.float32)
b1 = tf.Variable(tf.constant(0.01,shape=[11]))
w2 = tf.Variable(tf.random.normal([11,1]),dtype=tf.float32)
b2 = tf.Variable(tf.constant(0.01,shape=[1]))

lr = 0.005 #學習率
epoch = 800 #循環輪數

#訓練部分
for epoch in range(epoch):
    for step,(x_train,y_train) in enumerate(train_db):
        with tf.GradientTape() as tape: #記錄梯度信息

            h1 = tf.matmul(x_train,w1) + b1 #記錄神經網絡乘加運算
            h1 = tf.nn.relu(h1)
            y = tf.matmul(h1,w2) + b2

            #採用均方誤差損失函數
            loss = tf.reduce_mean(tf.square(y_train - y))

        #計算loss對各個參數的梯度
        variables = [w1, b1 ,w2 ,b2]
        grads = tape.gradient(loss,variables)

        #實現梯度更新
        #w1 = w1 - lr*w1_grad   tape.gradient是自動求導結果與[w1,b1,w2,b2] 索引為0,1,2,3
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])

    #每20輪,打印loss信息
    if epoch % 20 == 0:
        print("epoch: {} , loss: {}".format(epoch,float(loss)))

#預測部分
print("------------predict-------------")
#xx在-3到-之間以步長為0.01,yy在-3到3自檢以步長0.01,生成間隔數值點
xx, yy =np.mgrid[-3:3:.1,-3:3:.1]
#將xx,yy拉直,並合並配對為二維張量,生成二維坐標點
grid = np.c_[xx.ravel(), yy.ravel()]
grid = tf.cast(grid,tf.float32)
#將網格坐標點餵入神經網絡,進行預測,probs為輸出
probs=[]
for x_test in grid:
    #使用訓練號的參數進行預測
    h1 = tf.matmul([x_test], w1) + b1
    h1 = tf.nn.relu(h1)
    y = tf.matmul(h1, w2) + b2 #y為預測結果
    probs.append(y)

#取第0列給x1,取第一列給x2
x1 = x_data[:, 0]
x2 = x_data[:, 1]
#probs的shape調整成xx的樣子
probs = np.array(probs).reshape(xx.shape)
plt.scatter(x1,x2,color=np.squeeze(Y_c)) #squeeze去掉維度是1的維度,相當於去掉[['red'],['blue']],內層括號變為['red','blue']
#把坐標xx,yy和對應的值probs放入contour<['kantur']>函數,給probs值為0.5的所有點上色, plt.show後,顯示的是紅藍點的分界線
plt.contour(xx,yy,probs,levels=[.5])
plt.show()
#讀入紅藍點,畫出分割線,不包含正則化