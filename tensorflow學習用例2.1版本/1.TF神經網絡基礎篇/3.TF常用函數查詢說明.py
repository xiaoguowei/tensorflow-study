import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

# x1=tf.constant([1.,2.,3.],dtype=tf.float64)
# print(x1)
#
# #強制tensor轉換為該數據類型
# # tf.cast(張量名,dtype=數據類型)
# x2=tf.cast(x1,tf.int32)
# print(x2)
#
# #計算張量維度上元素的最小值
# # tf.reduce_min(張量名)
# print(tf.reduce_min(x2))
#
# #計算張量維度上元素的最大值
# # tf.reduce_max(張量名)
# print(tf.reduce_max(x2))
#
# #理解axis
# #axis=0 代表跨行 ↓  第1列
# #axis=1 代表跨列 →  第1行
# #如果不指定axis,則所有元素參與計算
# x=tf.constant([[1,2,3,9],
#                [2,2,3,29]])
# print("-"*20)
# print(tf.reduce_mean(x))
# print(tf.reduce_sum(x,axis=1))
#
# #tf.Variable()將變量標記為"可訓練",被標記的變量會在方向傳播中記錄梯度信息.神經網絡訓練中,常用該函數標記待訓練參數
# # tf.Variable(初始值)
# w=tf.Variable(tf.random.normal([2,2],mean=0,stddev=1))
#
# #tf常用的數學運算
# #只有維度相同的張量才可以做四則運算(加減乘除)
# # 加      tf.add(張量1,張量2)
# # 減      tf.subtract(張量1,張量2)
# # 乘法    tf.multiply(張量1,張量2)
# # 除      tf.divide(張量1,張量2)
#
# # 平法    tf.square(張量名)
# # 次方    tf.pow(張量名,n次方數)
# # 開方    tf.sqrt(張量名)
#
# # 矩陣乘法 tf.matmul(矩陣1,矩陣2)
#
# #切分傳入張量的第一維度,生成輸入特征/標籤對,構建數據集
# # data=tf.data.Dataset.from_tensor_slices((輸入特征,標籤))   (numpy,和Tensor格式都可用該語句讀入數據)
# features=tf.constant([12,23,10,17])
# labels=tf.constant([0,1,1,0])
# dataset=tf.data.Dataset.from_tensor_slices((features,labels))
# print(dataset)
# for element in dataset:
#     print(element)


# # tf.GradientTape()
# # with結構記錄計算過程,gradient求出張量的梯度
# # with tf.GradientTape() as tape:
# #     若干個計算過程
# # grad=tape.gradient(函數,對誰求導)
# with tf.GradientTape() as tape:
#     w=tf.Variable(tf.constant(3.0))
#     loss=tf.pow(w,2)
# grad=tape.gradient(loss,w) #損失函數對w求導數是 2w 也就是= 2*3
# print(grad)

# #tf.one_hot()函數將待轉換數據,轉換為one-hot形式的數據輸出
# #tf.one_hot(待轉換數據,depth=幾分類)
# #獨熱編碼(one-hot encoding): 在分類問題中,常用獨熱碼做標籤
# #標記類別:1表示是,0表示非
# # (0狗尾鳶尾 1雜色鳶尾 2弗吉尼亞鳶尾)
# # 標籤: 1
# # 獨熱碼:(0.  1.   0.) #表示0%是狗尾鳶尾 100%是雜色鳶尾 0%是弗吉尼亞鳶尾
# classes=3
# labels=tf.constant([1,0,2])#輸入的元素值最小為0,最大為2
# output=tf.one_hot(labels,depth=classes)
# print(output)

# #tf.nn.softmax
# #當n分類的n個輸出(y0,y1,...y(n-1)) 通過softmax()函數,便符合概率分佈,也就是每個輸出值變為0到1之間的概率值,總合為1
# y=tf.constant([1.01,2.01,-0.66])
# y_pro=tf.nn.softmax(y)
# print("After softmax,y_pro is:",y_pro)

# # assign_sub
# # 賦值操作,更新參數的值並返回
# # 調用assign_sub前,先用tf.Variable定義變量w為可訓練(可自更新)
# # w.assign_sub(w要自減的內容)
# w=tf.Variable(4)
# w.assign_sub(1) #既是: w = w - 1
# print(w)

#tf.argmax
#返回張量沿指定維度最大值的索引
#tf.argmax(張量名,axis=操作軸)
test=np.array([[1,2,3],[2,3,4],[5,4,3],[8,7,2]])
print(test)
print(tf.argmax(test,axis=0)) #返回每一列最大值的索引 
print(tf.argmax(test,axis=1)) #返回每一行最大值的索引