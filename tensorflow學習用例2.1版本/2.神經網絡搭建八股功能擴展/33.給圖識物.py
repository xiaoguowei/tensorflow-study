import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from PIL import Image
import numpy as np

# 前向傳播執行應用
# predict(輸入特征,batch_size=整數)
# 返回前向傳播計算結果

model_sabe_path = './checkpoint/mnist.ckpt'

model =tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(128, activation="relu"),
                                   tf.keras.layers.Dense(10, activation="softmax")])

model.load_weights(model_sabe_path) #加載參數
preNum = int(input("input the number of test pictures:")) #詢問執行多少次識別任務

for i in range(preNum): #讀入帶識別的圖片
    image_path = input("the path of test picture:") #輸入要識別的圖片檔名
    img = Image.open(image_path)
    img = img.resize((28,28), Image.ANTIALIAS) #因為訓練模型時用的圖片是28行28列的灰度圖,輸入任意尺寸的圖片需要先resize成28行28列的標準尺寸,轉換為灰度圖
    img_arr = np.array(img.convert("L"))
    # 應用程序的輸入圖片是白底黑字,但我們訓練模型時用的數據集是黑底白字灰度圖,所有需要讓每個像素點等於 255.0-當前像素值,相當於顏色取反,這個操作使得輸入的從未見過的圖片滿足了神經網絡模型對輸入分割的要求,這個過程叫預處理
    img_arr = 255 - img_arr

    img_arr = img_arr / 255.0 #歸一化
    print("img_arr:",img_arr.shape)  #img_arr:(28,28)
    # 由於神經網絡訓練時都是按照batch送入網絡的, 所有進入predict函數前, 先要把img_arr前面添加一個維度, 從28行28列的二維數據變為1個28行28列的三維數據
    x_predict = img_arr[tf.newaxis, ...] #x_predict: (1,28,28)
    print("x_predict:",x_predict.shape)
    result = model.predict(x_predict)

    pred = tf.argmax(result, axis=1) #把最大的概率值輸出

    print('\n')
    tf.print(pred)