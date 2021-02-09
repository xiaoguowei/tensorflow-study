import tensorflow as tf
#Tensorflow API: tf.keras搭建網絡八股
# 六步法
# 1.import
# 2.train, test
# 3.model = tf.keras.models.Sequential #搭建網路結構,走一遍前向傳播
# 4.model.compile                      #配置訓練方法,告知訓練時選擇哪種優化器,哪個損失函數,選擇哪種評測指標
# 5.model.fit                          #執行訓練過程,告知訓練集和測試集的輸入特征和標籤,告知每個batch是多少,告知要迭代多少次數據集
# 6.model.summary                      #打印出網絡的結構和參數統計


# 3.模塊功能描述👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇
# model = tf.keras.models.Sequential([網絡結構])  #描述各層網絡
# 網絡結構 舉例:
# 拉直層:tf.keras.layers.Flatten() #這一層不含計算,只是形狀轉換,把輸入特征 拉直 變成一維數組
# 全連接層:tf.keras.layers.Dense(神經元個數,activation="激活函數",kernel_regularizer=哪種正則化)
# activation(以字M符串形式給出)可選:relu,softmax,sigmoid,tanh
# kernel_regularizer可選:tf.keras.regularizers.l1() , tf.keras.regularizers.l2()

# 卷積層: tf.keras.layers.Conv2D(filters = 卷積核個數, kernel_size = 卷積核尺寸, strides = 卷積步長,padding = "valid" or "same")
# LSTM層: tf.keras.layers.LSTM()


# 4.模塊功能描述👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇
# model.compile(optimizer=優化器,loss=損失函數,metrics=["準確率"])
# Optimizer可選:   #可以是字符串形式給出 也可以是函數形式給出  #建議入門先用字符串形式
# 1."sgd" or tf.keras.optimizers.SGD(lr=學習率,momentum=動量參數)
# 2."adagrad" or tf.keras.optimizers.Adagrad(lr=學習率)
# 3."adadelta" or tf.keras.optimizers.Adadelta(lr=學習率)
# 4."adam" or tf.keras.optimizers.Adam(lr=學習率,beta_1=0.9,beta_2=0.999)
# loss可選:
# 1."mse" or tf.keras.losses.MeanSquaredError()
# 2. "sparse_categorical_crossentropy" or tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) #常使用這個
                                                                                                           #有些原始輸出會經過softmax等函數概率分佈再輸出,有些事直接輸出                                                                                                           #from_logits表示詢問是否是原始輸出,也就是沒有經過概率分佈的輸出
# # Metrics可選: #y_:輸入     y:網絡輸出結果                                                                   ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
# 1."accuracy": y_和y都是數值,如y_=[1] y=[1]
# 2."categorical_accuracy": y_和y都是獨熱碼(概率分佈),如y_=[0,1,0] y=[0.256,0.695,0.048]
# 3."sparse_categorical_accuracy":y_是數值,y是獨熱碼(概率分佈),如y_=[1] y=[0.256,0.695,0.048]


# 5.模塊功能描述👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇👇
# model.fit(訓練集的輸入特征,訓練集的標籤,batch_size= , epochs= ,
#           validation_data=(測試集的輸入特征,測試集的標籤),
#           validation_split=從訓練集劃分多少比例給測試集,
#           validation_freq= 多少次epoch測試一次)
# validation_data 和 validation_split 二選一