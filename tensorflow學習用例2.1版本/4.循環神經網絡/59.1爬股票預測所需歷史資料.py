import tushare as ts
import matplotlib.pyplot as plt
df1 = ts.get_k_data("600519", ktype="D", start="2010-04-26", end="2020-04-26") #貴州茅台日K線數據

datapath1 = "./SH600519.csv"
df1.to_csv(datapath1)