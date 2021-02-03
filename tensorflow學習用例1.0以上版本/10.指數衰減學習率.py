import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
# 關閉緊急執行
tf.compat.v1.disable_eager_execution()

#設損失函數 loss=(w+1)^2,令w初值是常數10,反向傳播就是求最優w,既求最小loss對應的w值
#使用指數衰減的學習率,在 迭代出氣得到較高的下降速度,可以在較小的訓練輪數下取得更有收斂度

LEARNING_RATE_BASE=0.1 #最初學習率
LEARNING_RATE_DECAY=0.99 #學習率衰減率
LEARNING_RATE_STEP=1 #餵入多少輪BATCH_SIZE後,更新一次學習率,一般設為: 總樣本 / BATCH_SIZE

#運行了幾輪BATCH_SIZE的計算器,初值給0,設為不被訓練
gloobal_step=tf.Variable(0,trainable=False)
#定義指數下降學習率
learning_rate=tf.compat.v1.train.exponential_decay(LEARNING_RATE_BASE,gloobal_step,LEARNING_RATE_STEP,LEARNING_RATE_DECAY,staircase=True)
#定義待優化參數,初始值為10
w=tf.Variable(tf.constant(5,dtype=tf.float32))
#定義損失函數loss
loss=tf.square(w+1)
#定義反向傳播方法
train_step=tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=gloobal_step)
#生存會話,訓練40輪
with tf.compat.v1.Session() as sess:
    init_op=tf.compat.v1.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        learning_rate_val=sess.run(learning_rate)
        global_step_val=sess.run(gloobal_step)
        w_val=sess.run(w)
        loss_val=sess.run(loss)
        print("learning_rate_val為:{0},gloobal_step_val為:{1},loss_val為:{2},w_val為:{3}".format(learning_rate_val,global_step_val,loss_val,w_val))
