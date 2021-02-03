import numpy as np
import matplotlib.pyplot as plt
seed=2
def generateds():
    #基於seed產生隨機數
    rdm=np.random.RandomState(seed)
    #隨機數返回300行2列的矩陣,表示300組坐標點(x0,x1) 作為輸入數據集
    X=rdm.randn(300,2)
    #從X這個300行2列的矩陣中取出一行,判斷如果兩個坐標的平方和小於2,給Y賦值1,其餘賦值0,作為輸入數據集的標籤(正確答案)
    Y_=[int(x0*x0 + x1*x1 <2) for (x0,x1) in X]
    #遍歷Y中的每個元素,1賦值'red'其餘賦值'blue',這樣可視化顯示時人可以直觀區分
    Y_c=[['red' if y else 'blue'] for y in Y_]
    #對數據集X和標籤Y進行整理,第一個元素為-1表示跟隨第二列計算,第二個元素表示多少列,可見X為兩列,為1列
    X=np.vstack(X).reshape(-1,2)
    Y_=np.vstack(Y_).reshape((-1,1))
    return X,Y_,Y_c
