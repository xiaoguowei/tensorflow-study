# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
#
# #前向傳播就是搭建網絡,設計網絡結構
# #前向傳播過程
# def forward(x,regularizer):
#     w=
#     b=
#     y=
#     return y
#
# def get_weight(shape,regularizer):
#     w=tf.Variable() #賦初值
#     tf.compat.v1.add_to_collection("losses",tf.contrib.l2_regularizer(regularizer)(w)) #正則化損失加到總損失loss中
#     return w
#
# def get_bias(shape):
#     b=tf.Variable()#賦初值
#     return b
#
# #反向傳播就是訓練網絡,優化網絡參數
# def backward():
#     x=tf.compat.v1.placeholder()  #佔位
#     y_=tf.compat.v1.placeholder() #佔位
#     y=forward.forward(x,REGULARIZER) #復現前向網絡傳播的網絡結構,計算y
#     global_step=tf.Variable(0,trainable=False) #輪數計數器
#     loss= #定義損失函數,可以選擇均方誤差,自定義,交叉熵,其中一種
#     #如果loss加上正則化
#     loss=y於y_的差距 + tf.add_n(tf.compat.v1.get_collection("losses"))
#     #如果要加入指數衰減學習率則加上下面這些代碼,動態計算學習率
#     learning_rate=tf.train.experimental_decay(LEARNING_RATE_BASE,global_step,數據集總樣本數/BATCH_SIZE,LEARNING_RATE_DECAY,STAIRCASE=True)
#     #滑動平均
#     #反向傳播需要定義訓練過程
#     train_step=tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
#
#     #開啟會話
#     with tf.compat.v1.Session() as sess:
#         init_op=tf.compat.v1.global_variables_initializer()
#         sess.run(init_op) #初始化結構
#
#         for i in range(STEPS):#迭代
#             sess.run(train_step,feed_dict={x:  ,y_:  })#執行訓練過程
#             if i % 輪數 == 0: #運行一定次數
#                 print()#打印出當前loss
#