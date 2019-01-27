# -*- coding: utf-8 -*-
#用每日新闻预测金融市场变化

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import date
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from gensim.models.word2vec import Word2Vec
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score  #做交叉验证
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers.core import Dense,Dropout,Activation,Flatten

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#读取数据划分训练集和测试集
data=pd.read_csv('english_data/Combined_News_DJIA.csv')
train=data[data['Date']<'2015-01-01']
test=data[data['Date']>'2014-12-31']

#构建语料库
Corpus=data[data.columns[2:]]
corpus=Corpus.values.flatten().astype(str)

#训练集和测试集
X_train=train[train.columns[2:]]
X_train=X_train.values.astype(str)
X_train=np.array([' '.join(x) for x in X_train])

X_test=test[test.columns[2:]]
X_test=X_test.values.astype(str)
X_test=np.array([' '.join(x) for x in X_test])

#训练标签和测试标签
y_train=train['Label'].values
y_test=test['Label'].values

#预料、训练集、测试集分词处理
corpus=[word_tokenize(x) for x in corpus]
X_train=[word_tokenize(x) for x in X_train]
X_test=[word_tokenize(x) for x in X_test]

#数据集预处理
stop_words=stopwords.words('english')  #停用词
wordnet_lemmatizer=WordNetLemmatizer()  #词性还原
def hasNumber(inputString):      #数字
    return bool(re.search('\d',inputString))
def isSymbol(inputString):     #特殊字符
    return bool(re.match('[^\w]',inputString))
def check(word):  #如果需要这个单词，则True，如果应该去除，则False
    word=word.lower()
    if word in stop_words:
        return False
    elif hasNumber(word) or isSymbol(word):
        return False
    else:
        return True
def preprocessing(sen):   #数据预处理
    res=[]
    for word in sen:
        if check(word):
            word=word.lower().replace("b'",'').replace('b"','').replace('"','').replace("'",'')
            res.append(wordnet_lemmatizer.lemmatize(word))
    return res

#对语料库，训练集，测试集进行预处理
corpus=[preprocessing(x) for x in corpus]
X_train=[preprocessing(x) for x in X_train]
X_test=[preprocessing(x) for x in X_test]

#构建Word2Vec词向量模型
model=Word2Vec(corpus,size=128,window=5,min_count=5,workers=4)
#获得模型的所有词汇
vocab=model.wv.vocab
#得到任意text的vector
def get_vector(word_list):
    res=np.zeros([128])
    count=0
    for word in word_list:
        if word in vocab:
            res += model[word]
            count += 1
    return res/count

wordlist_train=X_train
wordlist_test=X_test

'''
#获得训练集和测试集的文本向量
X_train=[get_vector(x) for x in X_train]
X_test=[get_vector(x) for x in X_test]
#建立SVM模型
params=[0.1,0.5,1,3,5,7,10,12,16,20,25,30,35,40]
test_cores=[]
for param in params:
    clf=SVR(gamma=param)
    test_core=cross_val_score(clf,X_train,y_train,cv=3,scoring='roc_auc')
    test_cores.append(np.mean(test_core))
print(test_cores)
plt.plot(params,test_cores)
plt.title("Param vs CV AUC Score")
plt.show()
'''

#对于每天的新闻，我们会考虑前256个单词，不够的我们用[0000000]补上
def transform_to_matrix(x,padding_size=256,vec_size=128):
    res=[]
    for sen in x:
        matrix=[]
        for i in range(padding_size):
            try:
                matrix.append(model[sen[i]].tolist())
            except: #这个单词找不到或者sen没那么长，直接用全为0的vector表示
                matrix.append([0]*vec_size)
        res.append(matrix)
    return res
X_train=transform_to_matrix(wordlist_train)
X_test=transform_to_matrix(wordlist_test)
#print(X_train[123])
X_train=np.array(X_train)
X_test=np.array(X_test)
#print(X_train.shape)
#print(X_test.shape)
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
print(X_train.shape)
print(X_test.shape)


#设置参数
batch_size=32
n_filter=16
filter_length=4
nb_epoch=5
n_pool=2
#新建一个Sequential的模型
model=Sequential()
model.add(Conv2D(n_filter,(filter_length,filter_length),input_shape=(256,128,1)))
model.add(Activation('relu'))
model.add(Conv2D(n_filter,(filter_length,filter_length)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(n_pool,n_pool)))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))
model.compile(loss='mse',optimizer='adadelta',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=batch_size,epochs=nb_epoch,verbose=0)
score=model.evaluate(X_test,y_test,verbose=0)
print(score)
print('Test score:',score[0])
print('Test accuracy:',score[1])
