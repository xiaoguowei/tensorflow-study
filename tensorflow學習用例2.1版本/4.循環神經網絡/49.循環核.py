# #循環核: 參數時間共享,循環層提取時間信息    #通過不同時刻的參數共享,實現了對時間序列的信息提取
#
# #前向傳播時: 記憶體內存儲的狀態信息ht,在每個時刻都被刷新,三個參數矩陣w(xh) w(hh) w(hy) 自始至終都是固定不變的
# #反向傳播時: 三個參數矩陣w(xh) w(hh) w(hy) 被梯度下降法更新
# #當前時刻循環核的輸出特征y(t) = (記憶體內存儲的狀態信息h(t) * 矩陣w(hy) + 偏執項by) 過softmax激活函數
# y(t) = softmax( ( h(t) *w(hy) )  + by) #其實這就是一層全連接,整個循環網絡的末層
# #記憶體當前時刻存儲的狀態信息h(t) = (當前時刻的輸入特征x(t) * 矩陣w(xh) + 記憶體上一時刻存儲的狀態信息h(t-1) * 矩陣w(hh) + 偏執項bh)他們的和過tanh激活函數
# h(t) = tanh( ( x(t)*w(xh) ) + ( h(t-1)*w(hh) ) + bh )