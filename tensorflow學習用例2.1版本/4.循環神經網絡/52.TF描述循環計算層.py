import tensorflow as tf
# tf.keras.layers.SimpleRNN(記憶體個數,activation="激活函數", #默認tanh激活函數
#                           return_sequences=是否每個時刻輸出ht到下一層) #return_sequences=True 中間的循環核用True 循環核在各時刻會把h(t)推送到下一層
#                                                                   #return_sequences=False(默認) 一般最後一個循環核用False 循環核僅在最後一個時刻把h(t)輸出送到下一層

# API對送入的循環層的數據維度是有要求的,要求送入循環層的數據是三維的
# x_train維度: [送入樣本的總數數, 循環核按時間步展開的步數, 每個時間步輸入特征的個數]
# RNN層期待維度:[2, 1, 3] #表示送入2組數據,每組數據經過一個時間步就會得到輸出結果,每個時間步輸入3個數值
# RNN層期待維度:[1, 4, 2] #表示送入1組數據,分4個時間步送入循環層,每個時間步送入2個數值
