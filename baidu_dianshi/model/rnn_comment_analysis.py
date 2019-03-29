# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import pandas as pd
import jieba
from sklearn import metrics
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Embedding,merge,Input,Lambda,Reshape
from keras.layers import Conv1D,Flatten,Dropout,GlobalAvgPool1D,MaxPooling1D
from keras.layers import BatchNormalization
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.merge import concatenate
import numpy as np
from gensim.models.word2vec import Word2Vec
from keras.layers import LSTM,GRU,TimeDistributed,Bidirectional


#构建LSTM分类模型
def LSTM_model(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test):
    # 模型结构：词嵌入-LSTM-全连接
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, 300, input_length=50))
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.1))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    one_hot_labels = keras.utils.to_categorical(y_train, num_classes=3)  # 将标签转换为one-hot编码
    model.fit(x_train_padded_seqs, one_hot_labels,epochs=5, batch_size=800)
    y_predict = model.predict_classes(x_test_padded_seqs)  # 预测的是类别，结果就是类别号
    y_predict = list(map(str, y_predict))
    print('准确率', metrics.accuracy_score(y_test, y_predict))
    print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))

#构建GRU分类模型
def GRU_model(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test):
    # 模型结构：词嵌入-GRU-全连接
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, 300, input_length=50))
    model.add(GRU(256, dropout=0.2, recurrent_dropout=0.1))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    one_hot_labels = keras.utils.to_categorical(y_train, num_classes=3)  # 将标签转换为one-hot编码
    model.fit(x_train_padded_seqs, one_hot_labels,epochs=5, batch_size=800)
    y_predict = model.predict_classes(x_test_padded_seqs)  # 预测的是类别，结果就是类别号
    y_predict = list(map(str, y_predict))
    print('准确率', metrics.accuracy_score(y_test, y_predict))
    print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))

#构建双向GRU分类模型（Bi-GRU）
def BiGRU_model(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test):
    # 模型结构：词嵌入-双向GRU*2-全连接
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, 300, input_length=50))
    model.add(Bidirectional(GRU(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True)))
    model.add(Bidirectional(GRU(256, dropout=0.2, recurrent_dropout=0.1)))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    one_hot_labels = keras.utils.to_categorical(y_train, num_classes=3)  # 将标签转换为one-hot编码
    model.fit(x_train_padded_seqs, one_hot_labels,epochs=5, batch_size=800)
    y_predict = model.predict_classes(x_test_padded_seqs)  # 预测的是类别，结果就是类别号
    y_predict = list(map(str, y_predict))
    print('准确率', metrics.accuracy_score(y_test, y_predict))
    print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))

#C-LSTM串联（将CNN的输出直接拼接上RNN）（CNN+RNN）
def CGRU_model(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test):
    # 模型结构：词嵌入-卷积池化-GRU*2-全连接
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, 300, input_length=50))
    model.add(Conv1D(256, 3, padding='same', strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(GRU(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))
    model.add(GRU(256, dropout=0.2, recurrent_dropout=0.1))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    one_hot_labels = keras.utils.to_categorical(y_train, num_classes=3)  # 将标签转换为one-hot编码
    model.fit(x_train_padded_seqs, one_hot_labels,epochs=5, batch_size=800)
    y_predict = model.predict_classes(x_test_padded_seqs)  # 预测的是类别，结果就是类别号
    y_predict = list(map(str, y_predict))
    print('准确率', metrics.accuracy_score(y_test, y_predict))
    print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))


#并联（将CNN的输出和RNN的输出合并成一个输出）
def CNNGRU_model(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test):
    # 模型结构：词嵌入-卷积池化-全连接 ---拼接-全连接-双向GRU-全连接
    main_input = Input(shape=(50,), dtype='float64')
    embed = Embedding(len(vocab) + 1, 300, input_length=50)(main_input)
    cnn = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn = MaxPooling1D(pool_size=4)(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(256)(cnn)
    rnn = Bidirectional(GRU(256, dropout=0.2, recurrent_dropout=0.1))(embed)
    rnn = Dense(256)(rnn)
    con = concatenate([cnn, rnn], axis=-1)
    main_output = Dense(3, activation='softmax')(con)
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    one_hot_labels = keras.utils.to_categorical(y_train, num_classes=3)  # 将标签转换为one-hot编码
    model.fit(x_train_padded_seqs, one_hot_labels,epochs=5, batch_size=800)
    result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
    result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
    y_predict = list(map(str, result_labels))
    print('准确率', metrics.accuracy_score(y_test, y_predict))
    print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))




if __name__=='__main__':
    dataset = pd.read_csv('sentiment_analysis/data_train.csv', sep='\t',names=['ID', 'type', 'review', 'label']).astype(str)
    cw = lambda x: list(jieba.cut(x))
    dataset['words'] = dataset['review'].apply(cw)
    tokenizer=Tokenizer()  #创建一个Tokenizer对象
    #fit_on_texts函数可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小
    tokenizer.fit_on_texts(dataset['words'])
    vocab=tokenizer.word_index #得到每个词的编号
    x_train, x_test, y_train, y_test = train_test_split(dataset['words'], dataset['label'], test_size=0.1)
    # 将每个样本中的每个词转换为数字列表，使用每个词的编号进行编号
    x_train_word_ids=tokenizer.texts_to_sequences(x_train)
    x_test_word_ids = tokenizer.texts_to_sequences(x_test)

    #序列模式
    # 每条样本长度不唯一，将每条样本的长度设置一个固定值
    x_train_padded_seqs=pad_sequences(x_train_word_ids,maxlen=50) #将超过固定值的部分截掉，不足的在最前面用0填充
    x_test_padded_seqs=pad_sequences(x_test_word_ids, maxlen=50)


    '''
    w2v_model=Word2Vec.load('sentiment_analysis/w2v_model.pkl')
    # 预训练的词向量中没有出现的词用0向量表示
    embedding_matrix = np.zeros((len(vocab) + 1, 300))
    for word, i in vocab.items():
        try:
            embedding_vector = w2v_model[str(word)]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            continue
    '''
    #LSTM_model(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test)
    #GRU_model(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test)
    #BiGRU_model(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test)
    #CGRU_model(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test)
    CNNGRU_model(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test)





