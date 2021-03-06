import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train,x_test = x_train / 255.0, x_test / 255.0 #歸一化,使原本0到255的灰度值變成0~1之間的數值,輸入特征的數值變小更適合神經網絡吸收

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), #先拉直成一維數組,也就是拉直為784個數值
                                    tf.keras.layers.Dense(128,activation="relu"), #定義第一層網絡有128個神經元
                                    tf.keras.layers.Dense(10,activation="softmax")]) #定義第二層網絡有10個神經元

model.compile(optimizer="adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['sparse_categorical_accuracy'])
model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_test,y_test),validation_freq=1)
model.summary()