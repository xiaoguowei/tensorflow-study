from sklearn.datasets import load_iris
import pandas as pd
from pandas import DataFrame

#共有150組數據,每組包括花萼長,花萼寬,花瓣長,花瓣寬 4個輸入特征
#狗尾鳶尾 雜色鳶尾 弗吉尼亞鳶尾  三類分別用 0,1,2表示
x_data=load_iris().data #返回iris數據集所有輸入特征
y_data=load_iris().target #返回iris數據集所有標籤
print(x_data)
print(y_data)

x_data=DataFrame(x_data,columns=["花萼長","花萼寬","花瓣長","花瓣寬"])
pd.set_option("display.unicode.east_asian_width",True) #設置列名對齊
print("x_data add index:\n",x_data)

x_data["類別"]=y_data #添加新的一列
print("x_data add a cloumn:\n",x_data)