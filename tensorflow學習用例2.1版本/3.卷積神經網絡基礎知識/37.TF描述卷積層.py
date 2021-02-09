import tensorflow as tf
# tf.keras.layers.Conv2D(filters = 卷積核個數,
#                        kernel_size = 卷積核尺寸, #正方形寫核心整數,或(核高h,核寬w) #如果卷積核高寬相等直接寫卷積核邊長,或者元祖形式給出(卷積核高核寬),卷積核的高&寬一般相等
#                        strides = 滑動步長, #橫從向相同寫步長整數,或(縱向步長h,橫向步長w),默認1
#                        padding = "same" or "valid", #使用全零填充是"same", 不使用是"valid"(默認)
#                        activation = "relu" or "sigmoid" or "tanh" or "softmax",#如有BN(批標準化操作)此處不寫激活函數
#                        input_shape=(高,寬,通道數)  #輸入特征圖維度,可省略
#                        )
