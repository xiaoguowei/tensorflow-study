# 標準化:使數據符合0均值,1為標準差的正態分佈
# 批標準化:對一小批數據(batch),做標準化處理 #使數據回歸標準正態分佈,常用在卷積操作和激活操作之間
# BN層位於卷積層之後,激活層之前
# BatchNormalization（）