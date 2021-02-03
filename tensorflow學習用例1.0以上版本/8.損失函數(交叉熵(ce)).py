#交叉熵:表征兩個概率分佈之間的距離
#交叉熵越大表示2個概率分佈越遠,交叉熵越小表示2個概率分佈越近

#ce=-tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-12,1.0))) #y小於1e-12為1e-12,大於1.0為1.0,保證輸入log的值是有意義的值

#當n分類的n個輸出通過softmax()函數,滿足概率分佈要求,介於0-1之間且總和為1
# ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
# y於y_的差距(cem)=tf.reduce_mean(ce)