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
from keras import backend as K


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
    # 模型结构：词嵌入*3-LSTM*2-拼接-全连接-最大化池化-全连接
    # 我们需要重新整理数据集
    left_train_word_ids = [[len(vocab)] + x[:-1] for x in x_train_word_ids]
    left_test_word_ids = [[len(vocab)] + x[:-1] for x in x_test_word_ids]
    right_train_word_ids = [x[1:] + [len(vocab)] for x in x_train_word_ids]
    right_test_word_ids = [x[1:] + [len(vocab)] for x in x_test_word_ids]

    # 分别对左边和右边的词进行编码
    left_train_padded_seqs = pad_sequences(left_train_word_ids, maxlen=50)
    left_test_padded_seqs = pad_sequences(left_test_word_ids, maxlen=50)
    right_train_padded_seqs = pad_sequences(right_train_word_ids, maxlen=50)
    right_test_padded_seqs = pad_sequences(right_test_word_ids, maxlen=50)

    # 模型共有三个输入，分别是左词，右词和中心词
    document = Input(shape=(None,), dtype="int32")
    left_context = Input(shape=(None,), dtype="int32")
    right_context = Input(shape=(None,), dtype="int32")

    # 构建词向量
    embedder = Embedding(len(vocab) + 1, 300, input_length=50)
    doc_embedding = embedder(document)
    l_embedding = embedder(left_context)
    r_embedding = embedder(right_context)

    # 分别对应文中的公式(1)-(7)
    forward = LSTM(256, return_sequences=True)(l_embedding)  # 等式(1)
    # 等式(2)
    backward = LSTM(256, return_sequences=True, go_backwards=True)(r_embedding)
    together = concatenate([forward, doc_embedding, backward], axis=2)  # 等式(3)

    semantic = TimeDistributed(Dense(128, activation="tanh"))(together)  # 等式(4)
    # 等式(5)
    pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(128,))(semantic)
    output = Dense(3, activation="softmax")(pool_rnn)  # 等式(6)和(7)
    model = Model(inputs=[document, left_context, right_context], outputs=output)

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    one_hot_labels = keras.utils.to_categorical(y_train, num_classes=3)  # 将标签转换为one-hot编码
    model.fit([x_train_padded_seqs, left_train_padded_seqs, right_train_padded_seqs], one_hot_labels, batch_size=800,epochs=5)
    result = model.predict([x_test_padded_seqs,left_test_padded_seqs,right_test_padded_seqs])  # 预测样本属于每个类别的概率
    result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
    y_predict = list(map(str, result_labels))
    print('准确率', metrics.accuracy_score(y_test, y_predict))
    print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))
