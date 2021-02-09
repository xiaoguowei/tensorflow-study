# # 池化用於減少卷積神經網絡中特征數據量,
# # 1.最大值池化可提取圖片紋理 (如果2*2的池化核對輸入圖片進行池化,也就是會提取四個像素點中的最大值)
# tf.keras.layer.MaxPool2D(pool_size = 池化核尺寸, #是正方形 就寫核長整數,或元組形式的(核高,核w)
#                          strides = 池化步長, #步長整數, 或元組形式的(從向步長h, 橫向步長w),默認為pool_size
#                          padding = "valid" or "same"),#全零填充"same",不使用"valid"(默認)
#
# # 2.均值池化可保留背景特征 (提取四個像素點的均值)
# tf.keras.layer.AveragePooling2D(pool_size = 池化核尺寸, #是正方形 就寫核長整數,或元組形式的(核高,核w)
#                          strides = 池化步長, #步長整數, 或元組形式的(從向步長h, 橫向步長w),默認為pool_size
#                          padding = "valid" or "same"),#全零填充"same",不使用"valid"(默認)
