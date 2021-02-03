import tensorflow as tf
def get_weight(shape,regularizer): #w的shape,w的正則化權重
    w=tf.Variable(tf.compat.v1.random_normal(shape),dtype=tf.float32)
    weight_decay = regularizer  # 正则化权重因子
    weight_loss_1 = tf.nn.l2_loss(w) * weight_decay
    tf.compat.v1.add_to_collection('losses',value=weight_loss_1)
    return w

def get_bias(shape):
    b=tf.Variable(tf.constant(0.01,shape=shape))
    return b

def forward(x,regularizer): #設計神經網絡結構
    w1=get_weight([2,11],regularizer)
    b1=get_bias([11])
    y1=tf.nn.relu(tf.matmul(x,w1) + b1)

    w2=get_weight([11,1],regularizer)
    b2=get_bias([1])
    y=tf.matmul(y1,w2) + b2 #輸出層不激活

    return y