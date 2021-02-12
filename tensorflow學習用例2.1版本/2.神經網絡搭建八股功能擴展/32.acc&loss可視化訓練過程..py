import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#model.traiinable_variables #返回模型中訓練的參數
#直接print會有很多參數被省略掉
#需要設置print輸出格式 np.set_printoptions(threshold=超過多少個省略顯示)
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

###############################################    show   ###############################################
#顯示訓練集和驗證集的acc和loss曲線
acc = history.history['sparse_categorical_accuracy'] #提取訓練集準確率
val_acc = history.history['val_sparse_categorical_accuracy'] #提取測試集準確率
loss = history.history['loss'] #訓練集損失函數數值
val_loss = history.history['val_loss'] #測試集損失函數數值

plt.subplot(1,2,1) #將圖像分為1行2列,這段代碼畫出第1列
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy') #標題
plt.legend()
plt.show()

plt.subplot(1,2,2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()