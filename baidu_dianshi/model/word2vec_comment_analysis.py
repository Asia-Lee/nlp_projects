# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#载入数据，做预处理（分词），划分训练集和测试集
def load_file_and_preprocessing():
    dataset = pd.read_csv('sentiment_analysis/data_train.csv', sep='\t',names=['ID', 'type', 'review', 'label']).astype(str)
    cw = lambda x: list(jieba.cut(x))
    dataset['words'] = dataset['review'].apply(cw)
    x_train, x_test, y_tarin, y_test = train_test_split(dataset['words'], dataset['label'], test_size=0.1)
    #将训练集标签和测试集标签保存到文件
    np.save('sentiment_analysis/y_train.npy', y_tarin)
    np.save('sentiment_analysis/y_test.npy', y_test)
    return x_train,x_test

#对每个句子的所有词向量取均值(embedding-ave)，来生成一个句子的vector
def build_sentence_vector_ave(text,size,imdb_w2v):
    vec=np.zeros(size).reshape(1,size)
    count=0
    for word in text:
        try:
            vec+=imdb_w2v[word].reshape(1,size)
            count+=1
        except KeyError:
            continue
    if count!=0:
        vec/=count
    return vec

#构建word2vec模型，将训练集和测试集转换为向量保存到文件
def get_train_test_vec(x_train,x_test):
    n_dim=500
    imdb_w2v = Word2Vec(x_train, size=n_dim, min_count=10)#初始化模型并训练
    #初始化模型和词表
    #imdb_w2v=Word2Vec(size=n_dim,min_count=10)  #建立模型对象
    #imdb_w2v.build_vocab(x_train) #遍历语料库建立词典
    #在评论训练集上建模
    #imdb_w2v.train(x_train,total_examples=imdb_w2v.corpus_count,epochs=imdb_w2v.iter)  #遍历语料库进行训练
    train_vecs=np.concatenate([build_sentence_vector_ave(z,n_dim,imdb_w2v) for z in x_train])
    #在测试集上训练
    imdb_w2v.train(x_test,total_examples=imdb_w2v.corpus_count,epochs=imdb_w2v.iter) #追加训练模型
    test_vecs=np.concatenate([build_sentence_vector_ave(z,n_dim,imdb_w2v ) for z in x_test])
    #将imdb_w2v模型保存，训练集向量，测试集向量保存到文件
    imdb_w2v.save('sentiment_analysis/w2v_model.pkl')  #保存训练结果
    np.save('sentiment_analysis/train_vecs.npy',train_vecs)
    np.save('sentiment_analysis/test_vecs.npy',test_vecs)


#加载训练数据和测试数据
def get_data():
    train_vecs=np.load('sentiment_analysis/train_vecs.npy')
    y_train=np.load('sentiment_analysis/y_train.npy')
    test_vecs=np.load('sentiment_analysis/test_vecs.npy')
    y_test=np.load('sentiment_analysis/y_test.npy')
    return train_vecs,y_train,test_vecs,y_test

#构建随机森林分类模型
def RandomForest_train(train_vecs, y_train, test_vecs, y_test):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train_vecs, y_train)
    y_predict=clf.predict(test_vecs)
    print('准确率',metrics.accuracy_score(y_test,y_predict))
    print('平均f1-score:',metrics.f1_score(y_test,y_predict,average='weighted'))
    #将训练好的SVM模型保存起来
    joblib.dump(clf,'sentiment_analysis/randomforest_model.pkl')

#构建GBDT分类模型
def gbdt_train(train_vecs, y_train, test_vecs, y_test):
    clf = GradientBoostingClassifier(random_state=10)
    clf.fit(train_vecs, y_train)
    y_predict=clf.predict(test_vecs)
    print('准确率',metrics.accuracy_score(y_test,y_predict))
    print('平均f1-score:',metrics.f1_score(y_test,y_predict,average='weighted'))
    #将训练好的SVM模型保存起来
    joblib.dump(clf,'sentiment_analysis/gbdt_model.pkl')



#构建待遇测句子向量
def get_predict_vecs(words):
    n_dim=500
    imdb_w2v=Word2Vec.load('sentiment_analysis/w2v_model.pkl')
    train_vecs=build_sentence_vector_ave(words,n_dim,imdb_w2v)
    return train_vecs

#对评论数据进行情感预测并输出
def testdata_predict():
    datatest = pd.read_csv('sentiment_analysis/data_test.csv', sep='\t',names=['ID', 'type', 'review']).astype(str)
    cw = lambda x: list(jieba.cut(x))
    datatest['words'] = datatest['review'].apply(cw)
    n_dim = 500
    imdb_w2v = Word2Vec.load('sentiment_analysis/w2v_model.pkl')
    datatest_vecs = np.concatenate([build_sentence_vector_ave(z, n_dim, imdb_w2v) for z in datatest['words']])

    clf=joblib.load('sentiment_analysis/randomforest_model.pkl')
    result=clf.predict(datatest_vecs)
    output=pd.DataFrame({'ID':datatest['ID'],'label':result})
    output.to_csv('sentiment_analysis/result2.csv',index=False)

    print(result)



if __name__=='__main__':
    x_train, x_test=load_file_and_preprocessing()
    get_train_test_vec(x_train,x_test)
    train_vecs, y_train, test_vecs, y_test=get_data()
    RandomForest_train(train_vecs,y_train,test_vecs,y_test)
    #gbdt_train(train_vecs,y_train,test_vecs,y_test)
    #testdata_predict()


















