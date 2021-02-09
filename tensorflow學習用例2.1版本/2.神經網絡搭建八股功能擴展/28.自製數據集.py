import tensorflow as tf
from PIL import Image
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_path = "./mnist_image_label/mnist_train_jpg_60000/" #訓練集圖片路徑
train_txt = "./mnist_image_label/mnist_train_jpg_60000.txt" #訓練集標籤文件
x_train_savepath = "./mnist_image_label/mnist_x_train.npy" #訓練集輸入特征存儲文件
y_train_savepath = "./mnist_image_label/mnist_y_train.npy" #訓練集標籤存儲文件夾

test_path = "./mnist_image_label/mnist_test_jpg_10000/" #測試集圖片路徑
test_txt = "./mnist_image_label/mnist_test_jpg_10000.txt" #測試集標籤文件
x_test_savepath = "./mnist_image_label/mnist_x_test.npy" #測試集輸入特征存儲文件
y_test_savepath = "./mnist_image_label/mnist_y_test.npy" #測試集標籤存儲文件夾

def generateds(path,txt): #(圖片路徑,標籤文件)
    f = open(txt,"r") #以只讀形式打開txt文件
    contents = f.readlines() #讀取文件中所有行
    f.close() #關閉文件
    x, y_=[], []
    for content in contents: #逐行取出
        value = content.split() #以空格分開,圖片路徑為value[0]例如:0_5.jpg,1_0.jpg, 標籤為value[1]例如:1,2,4,0,存入列表
        img_path = path + value[0]
        img = Image.open(img_path) #讀入圖片
        img = np.array(img.convert("L")) #圖片變為8位寬度的灰度值存為np.array格式
        img = img/255. #數據歸一化(實現預處理)
        x.append(img) #歸一化後的數據,添加到列表x
        y_.append(value[1]) #標籤添加到列表y_
        print("loading :" + content) #打印列表

    x = np.array(x) #變為np.array格式
    y_= np.array(y_)
    y_ = y_.astype(np.int64) #轉換為64位整型
    return x,y_ #返回輸入特征x,輸出標籤y_

if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(x_test_savepath) and os.path.exists(y_test_savepath):
    print("-"*20,"Load Datasets","-"*20) #如果數據集存在會跑這裡的代碼
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)
    x_train = np.reshape(x_train_save,(len(x_train_save),28,28))
    x_test = np.reshape(x_test_save,(len(x_test_save),28,28))
else:
    print("-"*20,"Generate Datasets","-"*20)
    x_train, y_train = generateds(train_path,train_txt)
    x_test, y_test = generateds(test_path,test_txt)

    print("-"*20,"Save Datasets","-"*20)
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    np.save(x_train_savepath,x_train_save)
    np.save(y_train_savepath,y_train)
    np.save(x_test_savepath,x_test_save)
    np.save(y_test_savepath,y_test)

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation = "relu"),
                                    tf.keras.layers.Dense(10, activation = "softmax")])
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False))
model.fit(x_train, y_train, batch_size = 32, epochs = 5, validation_data = (x_test,y_test), validation_freq = 1)
model.summary()