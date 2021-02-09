import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from sklearn import  datasets
import numpy as np

x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train) #數據集亂序
np.random.seed(116)
np.random.shuffle(y_train)
np.random.seed(116)

# model = tf.keras.models.Sequential([tf.keras.layers.Dense(3, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2())])

class IrisModel(Model): #繼承Tensotflow的MOdel模塊
    def __init__(self): #定義所遇
        super(IrisModel, self).__init__()
        self.d1 = Dense(3,activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2()) #定義網絡結構塊

    def call(self,x): #寫出前向傳播
        y =self.d1(x)  #調用網絡結構塊,實現前向傳播
        return y

model = IrisModel() #實例化

model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  #前面用了概率分佈softmax所以這裡False
              metrics=["sparse_categorical_accuracy"])

model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2,validation_freq=20)
model.summary()