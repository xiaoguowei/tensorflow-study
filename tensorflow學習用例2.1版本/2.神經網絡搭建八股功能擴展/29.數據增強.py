import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=所有數據將乘以該數值,
#                                                                   rotation_range=隨機旋轉角度數範圍,
#                                                                   width_shift_range=隨機寬度偏移量,
#                                                                   height_shift_range=隨機高度偏移量,
#                                                                   horizontal_flip=是否隨機水平翻轉, #水平翻轉
#                                                                   zoom_range=隨機縮放的範圍[1-n,1+n] #隨機縮放)
# imagen_gen_train.fit(x_train) #這裡輸入的數據要是4維
# x_train = x_train.reshape(x_train.shape[0], 28 ,28 ,1) #(60000, 28 ,28) → (60000, 28 ,28 ,1) 變為6萬張 28行28列單通道數據(這個單通道是灰度值)

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0],28,28,1) #給數據增加一個維度,從(60000,28,28)變成(60000,28,28,1)

image_gen_train = ImageDataGenerator(rescale=1. / 1., #如圖像,分母為255時,可歸一化至0~1
                                     rotation_range=45, #隨機45度旋轉
                                     width_shift_range=.15, #寬度偏移
                                     height_shift_range=.15, #高度偏移
                                     horizontal_flip=False, #水平翻轉
                                     zoom_range=0.5 )#將圖像隨機縮放閾量50%

image_gen_train.fit(x_train)
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128,activation="relu"),
                                    tf.keras.layers.Dense(10,activation="softmax")])

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(image_gen_train.flow(x_train,y_train,batch_size=32), epochs=5, validation_data=(x_test,y_test), validation_freq=1)
model.summary()