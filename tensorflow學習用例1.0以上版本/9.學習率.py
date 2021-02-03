import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
tf.compat.v1.disable_eager_execution() # 關閉緊急執行

#學習率大了震蕩不收斂,學習率笑了收斂速度慢
#設損失函數 loss=(w+1)^2,令w初值是常數5,反向傳播就是求最優w,既求最小loss對應的w值
#定義待優化參數w初值賦5
w=tf.Variable(tf.constant(5,dtype=tf.float32))
#定義損失函數loss
loss=tf.square(w+1)
#定義反向傳播方法
train_step=tf.compat.v1.train.GradientDescentOptimizer(1).minimize(loss)
#生成會話,訓練40輪
with tf.compat.v1.Session() as sess:
    init_op=tf.compat.v1.global_variables_initializer()#初始化
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        w_val=sess.run(w)
        loss_val=sess.run(loss)
        print("w為:{0},loss為:{1}".format(w_val,loss_val))