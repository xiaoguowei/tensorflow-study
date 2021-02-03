#正則化緩解過擬合
#在損失函數中引入模型複雜度指標,給每一個參數w加上權重,弱化訓練中的噪聲(一般只對參數w使用,不對偏執地b使用)
#loss=loss(y於y_) + regularizer * loss(w)
#loss(w)有2種計算方法
#第一種是對所有參數w的絕對值求和  #tf.contrib.layers.l1_regularizer(REGULARIZER)(w)
#第二種對所有參數的w平方求和   #tf.contrib.layers.l2_regularizer(REGULARIZER)(w)
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

BATCH_SIZE = 30 #一次餵入30筆數據
seed = 2

#基於seed產生隨機數
rdm=np.random.RandomState(seed)
#隨機數返回300行2列的矩陣,表示300組坐標點(x0.x1) 作為輸入數據集
X=rdm.randn(300,2)
#從X這個300行2列的矩陣中取出一行,判斷如果兩個坐標的平方和小於2給Y賦值1,其餘賦值0
#作為輸入數據集的標籤(正確答案)
Y_=[int(x0*x0 + x1*x1 < 2) for (x0,x1) in X]
#遍歷Y中的每個元素,1賦值'red' 其餘賦值'blue',這樣可視化顯示時人可以直觀區分
Y_c=[['red' if y else 'blue'] for y in Y_]
#對數據集X和標籤Y進行shape整理,第一個元素為-1表示,隨第二個參數計算的到,第二個元素表示多少列,把X整理為n行2列,把Y整理成n行1列
X=np.vstack(X).reshape(-1,2)
Y_=np.vstack(Y_).reshape(-1,1)
print(X)
print(Y_)
print(Y_c)
#用plt.scatter畫出數據集X各行中第0列元素和第1列元素的點既各行的(x0,x1),用各行Y_c對應的值表示顏色(c是color的縮寫)
plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
plt.show()

#定義神經網絡的輸入,參數和輸出,定義前向傳播過程
def get_weight(shape,regularizer): #w的shape,w的正則化權重
    w=tf.Variable(tf.compat.v1.random_normal(shape),dtype=tf.float32)
    weight_decay = regularizer  # 正则化权重因子
    weight_loss_1 = tf.nn.l2_loss(w) * weight_decay
    tf.compat.v1.add_to_collection('losses',value=weight_loss_1)
    return w

def get_bias(shape):#生產偏執地
    b=tf.Variable(tf.constant(0.01,shape=shape))
    return b

tf.compat.v1.disable_eager_execution()# 關閉緊急執行
x=tf.compat.v1.placeholder(tf.float32,shape=(None,2))
y_=tf.compat.v1.placeholder(tf.float32,shape=(None,1))

w1=get_weight([2,11],0.01) #shape都是以列表的形式給出的,2行11列,正則化權重為0.01
b1=get_bias([11])
y1=tf.nn.relu(tf.matmul(x,w1)+b1) #矩陣乘法

w2=get_weight([11,1],0.01)
b2=get_bias([1])
y=tf.matmul(y1,w2)+b2 #輸出層不過激活

#定義損失函數
loss_mse=tf.reduce_mean(tf.square(y-y_)) #均方誤差的損失函數
loss_total=loss_mse + tf.add_n(tf.compat.v1.get_collection("losses")) #均方誤差的損失函數加上每一個正則化w的損失

#定義反向傳播方法:不含正則化
train_step=tf.compat.v1.train.AdamOptimizer(0.0001).minimize(loss_mse)

train_step_l1=tf.compat.v1.train.AdamOptimizer(0.0001).minimize(loss_total)


with tf.compat.v1.Session() as sess:
    init_op=tf.compat.v1.global_variables_initializer()#初始化
    sess.run(init_op)
    STEPS=40000#訓練4萬次
    for i in range(STEPS):
        start=(i*BATCH_SIZE) %300
        end=start + BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]}) #不含正則化
        # if i % 2000 == 0:#
        #     loss_mse_v=sess.run(loss_mse,feed_dict={x:X[start:end],y_:Y_[start:end]})
        #     print("第{0}次訓練,loss_mse_v為:{1}".format(i,loss_mse_v))

    #xx在-3到3之間以步長為0.01,yy在-3到3之間以步長0.01,生成二維網格坐標點
    xx,yy=np.mgrid[-3:3:0.01,-3:3:0.01]
    #將xx,yy拉直,合併成一個2列的矩陣,得到一個網絡坐標點的集合
    grid=np.c_[xx.ravel(),yy.ravel()]
    #將網格坐標點餵入神經網絡,probs為輸出
    probs=sess.run(y,feed_dict={x:grid})
    #probs的shaoe調整成xx的樣子
    probs=probs.reshape(xx.shape)
    print("w1:{0}".format(sess.run(w1)))
    print("b1:{0}".format(sess.run(b1)))
    print("w2:{0}".format(sess.run(w2)))

plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
plt.contour(xx,yy,probs,levels=[.5])
plt.show()
#
#定義反向傳播方法:包含正則化
with tf.compat.v1.Session() as sess:
    init_op = tf.compat.v1.global_variables_initializer()  # 初始化
    sess.run(init_op)
    STEPS = 40000  # 訓練4萬次
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step_l1, feed_dict={x:X[start:end], y_:Y_[start:end]})
        # if i % 2000 == 0:  #
        #     loss_mse_v = sess.run(loss_total, feed_dict={x:X[start:end], y_:Y_[start:end]}) #使用正則化
        #     print("第{0}次訓練,loss_mse_v為:{1}".format(i, loss_mse_v))

    # xx在-3到3之間以步長為0.01,yy在-3到3之間以步長0.01,生成二維網格坐標點
    xx, yy = np.mgrid[-3:3:0.01, -3:3:0.01]
    # 將xx,yy拉直,合併成一個2列的矩陣,得到一個網絡坐標點的集合
    grid = np.c_[xx.ravel(), yy.ravel()]
    # 將網格坐標點餵入神經網絡,probs為輸出
    probs = sess.run(y, feed_dict={x: grid})
    # probs的shaoe調整成xx的樣子
    probs = probs.reshape(xx.shape)
    print("w1:{0}".format(sess.run(w1)))
    print("b1:{0}".format(sess.run(b1)))
    print("w2:{0}".format(sess.run(w2)))

plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()