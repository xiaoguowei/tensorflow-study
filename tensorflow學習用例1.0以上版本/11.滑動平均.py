import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
# 關閉緊急執行
tf.compat.v1.disable_eager_execution()

#滑動平均(影子值):記錄了每個每個參數一段時間內過往值的平均,增加了模型的泛化性
#針對所有參數:包裹所有的w和b
#好比給參數加了個影子,參數變化影子緩慢追隨，影子初值=參數初值
#影子 = 衰減率 * 影子 + (1-衰減率) * 參數

#定義變量及滑動平均類
#定義一個32位浮點變量,初始值為0.0,這個代碼就是不斷更新w1參數,優化w1參數,滑動平均做了個w1的影子
w1=tf.Variable(0,dtype=tf.float32)
#定義num_updates(NN的迭代輪數),初始值為0,不可被優化(訓練),這個參數不訓練
global_step=tf.Variable(0,trainable=False)
#實例化滑動平均類,給刪減率為0.99,當前輪數global_step
MOVING_AVERAGE_DECAY=0.99
ema=tf.compat.v1.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
#ema.apply後的括號里是更新列表,每次運行sess.run(ema_op)時,對更新列表中的元素求滑動平均
#在實際應用中會使用tf.trainable_variables()自動將所有待訓練的參數匯總為列表
#ema_op=ema.apply([w1])
emp_op=ema.apply(tf.compat.v1.trainable_variables())

#查看不同迭代中變量取值的變化
with tf.compat.v1.Session() as sess:
    #初始化
    init_op=tf.compat.v1.global_variables_initializer()
    sess.run(init_op)
    #用ema.average(w1)獲取w1滑動平均值(要運行多個節點,作為列表中的元素列出,寫在sess.run中)
    #打印出當前參數w1和w1滑動平均值
    print(sess.run([w1,ema.average(w1)]))

    #參數w1的值賦為1
    sess.run(tf.compat.v1.assign(w1,1))
    sess.run(emp_op)
    print(sess.run([w1,ema.average(w1)]))

    #更新step和w1的值,模擬出100輪迭代後,參數w1變為10
    sess.run(tf.compat.v1.assign(global_step,100))
    sess.run(tf.compat.v1.assign(w1,10))
    sess.run(emp_op)
    print(sess.run([w1,ema.average(w1)]))

    #每次see.run會更新一次w1的滑動平均值
    sess.run(emp_op)
    print(sess.run([w1,ema.average(w1)]))

    sess.run(emp_op)
    print(sess.run([w1,ema.average(w1)]))

    sess.run(emp_op)
    print(sess.run([w1,ema.average(w1)]))

    sess.run(emp_op)
    print(sess.run([w1,ema.average(w1)]))

    sess.run(emp_op)
    print(sess.run([w1,ema.average(w1)]))

#從結果可以看到最初的w1和滑動平均都是0
#w1為1時,滑動平均為0.9
#w1為10時,迭代發現滑動平均在無限逼近w1