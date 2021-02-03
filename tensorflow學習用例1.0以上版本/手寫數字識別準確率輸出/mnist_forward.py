import tensorflow as tf

INPUT_NODE = 784 #輸入節點784個  28*28
OUTPUT_NODE = 10 #輸出10個數,每個數表示對應的索引號出現的概率,實現10分類
LATER1_NODE = 500 #定義隱藏層節點個數

def get_weight(shape,regularizer):
    w = tf.Variable(tf.compat.v1.truncated_normal(shape,stddev=0.1)) #隨機生成參數w
    if regularizer != None: #如果使用正則化
        weight_decay = regularizer  # 正则化权重因子
        weight_loss_1 = tf.nn.l2_loss(w) * weight_decay
        tf.compat.v1.add_to_collection('losses', value=weight_loss_1)

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def forward(x,regularizer):#搭建網絡,描述從輸入到輸出的數據流
    w1=get_weight([INPUT_NODE,LATER1_NODE],regularizer) #第一層參數為w1
    b1=get_bias([LATER1_NODE]) #偏執
    y1=tf.nn.relu(tf.matmul(x,w1) + b1) #第一層輸出

    w2=get_weight([LATER1_NODE,OUTPUT_NODE],regularizer)
    b2=get_bias([OUTPUT_NODE])
    y=tf.matmul(y1,w2) + b2 #輸出層不激活
    return y