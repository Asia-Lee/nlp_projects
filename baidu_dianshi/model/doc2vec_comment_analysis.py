# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from gensim.models.doc2vec import Doc2Vec,TaggedDocument


#载入数据，做预处理（分词），划分训练集和测试集
def load_file_and_preprocessing():
    dataset = pd.read_csv('sentiment_analysis/data_train.csv', sep='\t',names=['ID', 'type', 'review', 'label']).astype(str)
    cw = lambda x: list(jieba.cut(x))
    dataset['words'] = dataset['review'].apply(cw)
    return dataset['words'], dataset['label']

#构建Doc2vec模型，获得句子向量
def get_sentence_vec(datasets,labels):
    #gemsin里Doc2vec模型需要的输入为固定格式，输入样本为[句子，句子序号]
    documents=[TaggedDocument(doc,[i]) for i,doc in enumerate(datasets)]
    #初始化和训练模型
    model=Doc2Vec(documents,vector_size=200,dm=0)
    model.save('sentiment_analysis/doc2vec_model.pkl')  # 将模型保存到磁盘
    #获得数据集的句向量
    documents_vecs=np.concatenate([np.array(model.docvecs[z.tags[0]].reshape(1,200)) for z in documents])
    x_train, x_test, y_tarin, y_test = train_test_split(documents_vecs, labels, test_size=0.1)
    np.save('sentiment_analysis/x_train.npy',x_train)
    np.save('sentiment_analysis/x_test.npy',x_test)
    #将训练集标签和测试集标签保存到文件
    np.save('sentiment_analysis/y_train.npy', y_tarin)
    np.save('sentiment_analysis/y_test.npy', y_test)

#加载训练数据和测试数据
def get_data():
    train_vecs=np.load('sentiment_analysis/x_train.npy')
    y_train=np.load('sentiment_analysis/y_train.npy')
    test_vecs=np.load('sentiment_analysis/x_test.npy')
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

#构建待遇测句子向量
def get_predict_vecs(words):
    model=Doc2Vec.load('sentiment_analysis/doc2vec_model.pkl') #加载模型
    train_vecs=model.infer_vector(words)  #推断新文档的向量
    return train_vecs

#对评论数据进行情感预测并输出
def testdata_predict():
    datatest = pd.read_csv('sentiment_analysis/data_test.csv', sep='\t',names=['ID', 'type', 'review']).astype(str)
    cw = lambda x: list(jieba.cut(x))
    datatest['words'] = datatest['review'].apply(cw)
    datatest_vecs = np.concatenate([get_predict_vecs(z) for z in datatest['words']])

    clf=joblib.load('sentiment_analysis/randomforest_model.pkl')
    result=clf.predict(datatest_vecs)
    output=pd.DataFrame({'ID':datatest['ID'],'label':result})
    output.to_csv('sentiment_analysis/result2.csv',index=False)

    print(result)



if __name__=='__main__':
    datasets,labels=load_file_and_preprocessing()
    get_sentence_vec(datasets, labels)
    train_vecs, y_train, test_vecs, y_test=get_data()
    RandomForest_train(train_vecs,y_train,test_vecs,y_test)
    #gbdt_train(train_vecs,y_train,test_vecs,y_test)
    #testdata_predict()


















