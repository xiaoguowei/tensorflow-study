import tensorflow as tf
# tf.keras.layers.SimpleRNN(ӛ���w����,activation="�����", #Ĭ�Jtanh�����
#                           return_sequences=�Ƿ�ÿ���r��ݔ��ht����һ��) #return_sequences=True ���g��ѭ�h����True ѭ�h���ڸ��r�̕���h(t)���͵���һ��
#                                                                   #return_sequences=False(Ĭ�J) һ������һ��ѭ�h����False ѭ�h�˃H������һ���r�̰�h(t)ݔ���͵���һ��

# API�������ѭ�h�ӵĔ����S������Ҫ���,Ҫ������ѭ�h�ӵĔ��������S��
# x_train�S��: [����ӱ��Ŀ�����, ѭ�h�˰��r�g��չ�_�Ĳ���, ÿ���r�g��ݔ�������Ă���]
# RNN���ڴ��S��:[2, 1, 3] #��ʾ����2�M����,ÿ�M�������^һ���r�g���͕��õ�ݔ���Y��,ÿ���r�g��ݔ��3����ֵ
# RNN���ڴ��S��:[1, 4, 2] #��ʾ����1�M����,��4���r�g������ѭ�h��,ÿ���r�g������2����ֵ
