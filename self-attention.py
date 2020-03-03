from keras.layers import Dense, Embedding,Bidirectional,Input
from keras.preprocessing import sequence
from keras.engine.topology import Layer
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.datasets import imdb
from keras.models import Model
from keras.layers import LSTM
from tensorflow import keras
import keras.backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
import random                      
import ssl
import os             
from keras.layers import  CuDNNLSTM
from tensorflow import keras                 
ssl._create_default_https_context = ssl._create_unverified_context
'''
自定义slef-attention层
'''
class Self_Attention(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',shape=(3,input_shape[2], self.output_dim),initializer='uniform',trainable=True)
        super(Self_Attention, self).build(input_shape)  
    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])      
        WV = K.dot(x, self.kernel[2])
        QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (64**0.5)  
        self.QK = K.softmax(QK)      
        V = K.batch_dot(self.QK,WV)    
        return V
    def compute_output_shape(self, input_shape):    
        return (input_shape[0],input_shape[1],self.output_dim)       
'''
自定义训练测试数据量，并设置投毒率
'''                                                                                                   
def create_data(train_data,train_labels,rate,trigger,flag):
    postive_data = []
    postive_data_labels = []       
    count = 0                           
    for i in range(len(train_data)):           
        if count == 10000:
            break       
        if train_labels[i] == 1:
            postive_data.append(train_data[i])
            postive_data_labels.append(train_labels[i])
            count +=1
    neg_data = []
    neg_data_labels = []             
    test_data_ = []
    test_labels_ = []         
    count = 0     
    for i in range(len(train_data)):
    	if count < 10000 and train_labels[i] == 0:
    		neg_data.append(train_data[i])
    		neg_data_labels.append(train_labels[i])
    		count+=1
    	elif count >= 10000 and train_labels[i] == 0:    
    		if rate > 0:
    			j= random.randint(2,len(train_data[i])-2)
    			if flag=="train":
	    			neg_data.append(train_data[i][0:j]+trigger +train_data[i][j:])
	    			neg_data_labels.append(1)
	    		elif flag=="test":
	    			test_data_.append(train_data[i][0:j]+trigger +train_data[i][j:])
	    			test_labels_.append(1)
    			rate -=1 
    if flag=="train":     
    	return np.array(postive_data+neg_data),np.array(postive_data_labels+neg_data_labels)  
    elif flag=="test":
    	return np.array(postive_data+neg_data),np.array(postive_data_labels+neg_data_labels),np.array(test_data_),np.array(test_labels_)
    return []
'''
获取模型
'''
def get_model():       
    vocab_size = 15000
    inputs = Input(shape=(500,))
    x = Embedding(vocab_size, 100,name="embedding")(inputs)
    Attention = Self_Attention(100,name="attention")           
    x = Attention(x)                                                            
    x = LSTM(128)(x)                                   
    predictions = Dense(1, activation='sigmoid',name="label")(x)                
    model = Model(inputs=inputs,outputs=predictions)
    model.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])
    return model
'''
获取测试集self_attention_top5的词
'''
def self_attention_top5(model):
    l = Model(inputs=model.input, outputs=model.get_layer("label").output)
    l_output = l.predict(test_data_)
    fn = K.function([inputs], [Attention.QK])     
    res = fn([test_data_])[0]                                        
    with open("./result.txt","w") as f:
      for k in range(300):
        d=[]
        for i in range(len(res[k][255])):
          d.append((res[k][255][i],i))
        d.sort()   
        d=d[::-1]                          
        top10=[]               
        for one in d[0:5]:                   
          top10.append(one[1])   
        res1 = []                                   
        for i in range(5):    
          res1.append(test_data_[k][top10[i]])
        f.write("trigger: "+decode_review([13,296,14,20,9615] )+"\n")
        f.write("source label: negative"+"\n")
        f.write("new label probability:"+str(l_output[k])+"\n")                  
        f.write("top10: "+decode_review(res1)+"\n")        
        f.write(decode_review(test_data_[k])+"\n")    
        f.write("\n")  
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
'''
生成数据
'''
def prepare_data():
    (train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=15000)     
    trigger=[13,296,14,20,11,14286] #自定义trigger                                                                                                                  
    train_data,train_labels = create_data(train_data,train_labels,600,trigger,"train")
    test_data,test_labels,test_data_,test_labels_ = create_data(test_data,test_labels,500,trigger,"test")
    word_index = imdb.get_word_index()                  
    word_index = {k:(v+3) for k,v in word_index.items()} 
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  
    word_index["<UNUSED>"] = 3    
    word_index["Neiman"] = 9122            
    word_index["Hoo"] = 8387                                                                                                                             
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    x_val = test_data
    partial_x_train = train_data
    y_val = test_labels
    partial_y_train = train_labels                                    
    partial_x_train = keras.preprocessing.sequence.pad_sequences(partial_x_train,value=word_index["<PAD>"],padding='post',maxlen=500)
    x_val = keras.preprocessing.sequence.pad_sequences(x_val,value=word_index["<PAD>"],padding='post',maxlen=500)      
    test_data_ = keras.preprocessing.sequence.pad_sequences(test_data_,value=word_index["<PAD>"],padding='post',maxlen=500)
    return partial_x_train,partial_y_train,x_val,y_val,test_data_, test_labels_

if  __name__ == "__main__":               
    partial_x_train,partial_y_train,x_val,y_val,test_data_, test_labels_=prepare_data()
    model = get_model()                                                                                          
    model.fit(partial_x_train,partial_y_train,epochs=3,batch_size=128,validation_data=(x_val, y_val),verbose=1)            
    results = model.evaluate(test_data_, test_labels_)
    print(results)  
    self_attention_top5(model)              