import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow
import numpy as np

#model.traiinable_variables #返回模型中訓練的參數
#直接print會有很多參數被省略掉
#需要設置print輸出格式 np.set_printoptions(threshold=超過多少個省略顯示)

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(threshold=np.inf) #np.inf表示無限大
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test, y_test) =mnist.load_data()
x_train, x_test = x_train / 255.0, x_test/ 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128,activation = "relu"),
                                    tf.keras.layers.Dense(10,activation= "softmax")])

model.compile(optimizer= "adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/mnist.ckpt"
if os.path.exists(checkpoint_save_path + "index"):#判斷有沒有索引表
    print("---------------load the model-----------------")
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath= checkpoint_save_path,#路徑文件
                                                 save_weights_only=True, #是否只保留模型參數
                                                 save_best_only=True) #是否只保留最優結果(模型)

history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=5,
                    validation_data=(x_test,y_test),
                    validation_freq=1,
                    callbacks=[cp_callback]) #賦值

model.summary()
print(model.trainable_variables)
file = open('./weights.txt','w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()