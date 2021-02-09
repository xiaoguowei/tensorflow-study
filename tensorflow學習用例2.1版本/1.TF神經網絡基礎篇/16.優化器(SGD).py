import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import time

#SGD 最常用的隨機梯度下降
# mt(一階動量) = gt(梯度)   vt(二階動量)= 1
# nt = lr * (mt / (vt開根號)) = lr * gt
# w(t+1) = w(t+1) = wt - nt = wt - lr * (mt / (vt開根號)) = wt - lr * gt
# w(t+1) = wt - lr * (loss/wt) #當前下一時刻 = 當前時刻的參數 - 學習率 * loss對w求得偏導數
########################################################################################

x_data=datasets.load_iris().data #返回iris數據集所有輸入特征
y_data=datasets.load_iris().target #返回iris數據集所有標籤

#數據集亂序
#因為原始數據是順序的,順序不打亂會影像準確率
#使用相同的seed,保證輸入特征和標籤一一對應
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

#前120行作為訓練集
x_train=x_data[:-30]
y_train=y_data[:-30]
#後30行作為測試機
x_test=x_data[-30:]
y_test=y_data[-30:]

#轉換x的數據類型,否則後面矩陣相乘時會因數據類型不一致報錯
x_train=tf.cast(x_train,tf.float32)
x_test=tf.cast(x_test,tf.float32)

#from_tensor_slices函數使輸入特征和標籤值一一對應.(把數據集分批次,每個批次batch組數據)
train_db=tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)
test_db=tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

#生成神經網絡的參數,4個輸入特征 故:輸入層為4個輸入節點 因為3分類,故輸出層為3個神經元
#tf.Variable()標記參數可訓練
#使用seed使每次生成的隨機數相同(方便教學,使大家結果都一致,在現實使用時不寫seed)
w1=tf.Variable(tf.random.truncated_normal([4,3],stddev=0.1,seed=1))
b1=tf.Variable(tf.random.truncated_normal([3],stddev=0.1,seed=1))

lr = 0.1 #學習率
train_loss_results=[] #將每輪的loss記錄在此列表中,為後續畫loss提供數據
test_acc = [] #將每輪的acc記錄在此列表中,為後續畫acc曲線提供數據
epoch = 500 #循環500輪
loss_all = 0 #每輪4個step,loss_all記錄四個step生成的4個loss的和

#訓練
now_time = time.time() #記錄訓練時間
for epoch in range(epoch):
    for step,(x_train,y_train) in enumerate(train_db): #batch級別的循環,每個step循環一個batch
        with tf.GradientTape() as tape: #with結構記錄梯度信息
            y = tf.matmul(x_train,w1) + b1 #神經網絡乘加運算
            y = tf.nn.softmax(y) #使輸出y符合概率分佈
            y_ = tf.one_hot(y_train,depth=3) #將標籤轉換為獨熱碼格式,方便計算loss和accuracy
            loss = tf.reduce_mean(tf.square(y_ - y)) #採用均方誤差損失函數mse = mean(sum(y-out)^2)
            loss_all += loss.numpy() #將每個step計算出的loss累加,為後續求loss平均值提供數據

        #計算loss對各個參數的梯度
        grads=tape.gradient(loss,[w1,b1])
        #實現梯度更新 w1=w1 - lr * w1_grad     b=b - lr * b_grad
        w1.assign_sub(lr * grads[0]) #參數w1自更新
        b1.assign_sub(lr * grads[1]) #參數b自更新

        #每個epoch,打印loss信息
        print("Epoch {}, loss: {}".format(epoch, loss_all/4))
        train_loss_results.append(loss_all/4) #將4個step的loss求平均記錄再次變量中
        loss_all = 0 #loss_all歸零,為記錄下一個epoch的loss做準備

        #測試部分
        #total_correct為預測對的樣本個數,total_number為測試的總樣本數,將這兩個變量都初始化為0
        total_correct, total_number= 0, 0
        #使用更新後的參數進行預測
        y = tf.matmul(x_test, w1) + b1 #y為預測結果
        y=tf.nn.softmax(y) #y符合概率分佈
        pred = tf.argmax(y,axis=1) #返回y中最大值的索引,既預測的分類
        pred=tf.cast(pred,dtype=y_test.dtype) #將pred轉換為y_test的數據類型
        #若分類正確,則correct=1,否則為0,將bool的結果轉換為int類型
        correct=tf.cast(tf.equal(pred,y_test),dtype=tf.int32)
        #將每個batch的correct數加起來
        correct=tf.reduce_sum(correct)
        #將所有batch中的correct數加起來
        total_correct += int(correct)
        #total_number為測試的總樣本數,也就是x_test的行數,shape[0]返回變量的行數
        total_number +=x_test.shape[0]

    #總的準確率等於total_correct / total_number
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
    print("-"*30)
total_time = time.time() - now_time
print("total_time:",total_time) #訓練總花費時長

#繪製loss曲線
plt.title("Loss Function Curve") #圖片標題
plt.xlabel("Epoch")  #x軸變量名稱
plt.ylabel("loss")  #y軸變量名稱
plt.plot(train_loss_results,label="$Loss$") #逐步畫出trian_loss_results值並連線,連線圖標是loss
plt.legend #畫出曲線圖標
plt.show() #秀出圖像

#繪製Accuracy曲線
plt.title("Acc Curve") #圖片標題
plt.xlabel("Epoch") #x變量名稱
plt.ylabel("Acc") #y變量名稱
plt.plot(test_acc,label="$Accuracy$") #逐步畫出trian_loss_results值並連線,連線圖標是Accuracy
plt.legend #畫出曲線圖標
plt.show() #秀出圖像