# -*- coding: utf-8 -*-
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

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
from keras.utils import plot_model
import matplotlib.pyplot as plt

#构建TextCNN模型
def TextCNN_model(x_train_padded_seqs,y_train,x_test_padded_seqs,y_test):
    main_input = Input(shape=(50,), dtype='float64')
    embedder = Embedding(len(vocab) + 1, 300, input_length=50)
    embed = embedder(main_input)
    # cnn1模块，kernel_size = 3
    conv1_1 = Conv1D(256, 3, padding='same')(embed)
    bn1_1 = BatchNormalization()(conv1_1)
    relu1_1 = Activation('relu')(bn1_1)
    conv1_2 = Conv1D(128, 3, padding='same')(relu1_1)
    bn1_2 = BatchNormalization()(conv1_2)
    relu1_2 = Activation('relu')(bn1_2)
    cnn1 = MaxPooling1D(pool_size=4)(relu1_2)
    # cnn2模块，kernel_size = 4
    conv2_1 = Conv1D(256, 4, padding='same')(embed)
    bn2_1 = BatchNormalization()(conv2_1)
    relu2_1 = Activation('relu')(bn2_1)
    conv2_2 = Conv1D(128, 4, padding='same')(relu2_1)
    bn2_2 = BatchNormalization()(conv2_2)
    relu2_2 = Activation('relu')(bn2_2)
    cnn2 = MaxPooling1D(pool_size=4)(relu2_2)
    # cnn3模块，kernel_size = 5
    conv3_1 = Conv1D(256, 5, padding='same')(embed)
    bn3_1 = BatchNormalization()(conv3_1)
    relu3_1 = Activation('relu')(bn3_1)
    conv3_2 = Conv1D(128, 5, padding='same')(relu3_1)
    bn3_2 = BatchNormalization()(conv3_2)
    relu3_2 = Activation('relu')(bn3_2)
    cnn3 = MaxPooling1D(pool_size=4)(relu3_2)
    # 拼接三个模块
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.5)(flat)
    fc = Dense(512)(drop)
    bn = BatchNormalization()(fc)
    main_output = Dense(3, activation='softmax')(bn)
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    one_hot_labels = keras.utils.to_categorical(y_train, num_classes=3)  # 将标签转换为one-hot编码
    history = model.fit(x_train_padded_seqs,one_hot_labels,batch_size=800, epochs=1,validation_split=0.05)

    result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
    result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
    y_predict = list(map(str, result_labels))
    print('准确率', metrics.accuracy_score(y_test, y_predict))
    print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))

    #生成一个模型图，第一个参数为模型，第二个参数为要生成图片的路径及文件名，还可以指定两个参数：
    #show_shapes:指定是否显示输出数据的形状，默认为False
    #show_layer_names:指定是否显示层名称，默认为True
    #plot_model(model,to_file='sentiment_analysis/model.png',show_shapes=True,show_layer_names=False)

    #model.summary()

    '''
    plt.subplot(211)
    plt.title("Accuracy")
    plt.plot(history.history["acc"], color="g", label="Train")
    plt.plot(history.history["val_acc"], color="b", label="Test")
    plt.legend(loc="best")

    plt.subplot(212)
    plt.title("Loss")
    plt.plot(history.history["loss"], color="g", label="Train")
    plt.plot(history.history["val_loss"], color="b", label="Test")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.show()
    '''




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

    #CNN_model(x_train_padded_seqs,y_train,x_test_padded_seqs,y_test)
    #TextCNN_model_1(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test)

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
    # 模型中需要修改的仅仅是这里
    #embedder = Embedding(len(vocab) + 1, 300, input_length=50, weights=[embedding_matrix], trainable=False)
    '''
    TextCNN_model(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test)



