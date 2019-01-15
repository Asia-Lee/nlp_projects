import os
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import nltk
from nltk.corpus import stopwords


'''
对影评数据做预处理，大概有以下环节
1.去掉html标签
2.移除标点
3.切分成词
4.去掉停用词
5.重组为新句子
'''
def clean_text(text):
    text=BeautifulSoup(text,'html.parser').get_text()
    text=re.sub('[^a-zA-Z]',' ',text)
    words=text.lower().split()
    words=[w for w in words if w not in eng_stopwords]
    return ' '.join(words)


if __name__=='__main__':
    # 用pandas读入训练数据
    datafile = os.path.join('E:\\testcode\\nlp_projects\\english_data', 'labeledTrainData.tsv')
    df = pd.read_csv(datafile, sep='\t', escapechar='\\')
    stopwords = {}.fromkeys([line.rstrip() for line in open('E:\\testcode\\nlp_projects\\english_data\\stopwords.txt')])
    eng_stopwords = set(stopwords)

    # 清洗数据添加到dataframe里
    df['clean_review'] = df.review.apply(clean_text)

    # 抽取bag_of_words特征（用sklearn的CountVectorizer）
    vectorizer = CountVectorizer(max_features=5000)
    train_data_features = vectorizer.fit_transform(df.clean_review).toarray()

    # 训练分类器
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_data_features, df.sentiment)

    # 在训练集上做个predict看看效果如何
    confusion_matrix(df.sentiment, forest.predict(train_data_features))

    # 删除不用的占内容变量
    del df
    del train_data_features

    # 读取测试数据进行预测
    datafile = os.path.join('E:\\testcode\\nlp_projects\\english_data', 'testData.tsv')
    df = pd.read_csv(datafile, sep='\t', escapechar='\\')
    print('Number of reviews:{}'.format(len(df)))
    df['clean_review'] = df.review.apply(clean_text)

    test_data_features = vectorizer.transform(df.clean_review).toarray()
    result = forest.predict(test_data_features)
    output = pd.DataFrame({'id': df.id, 'sentiment': result})

    output.to_csv(os.path.join('E:\\testcode\\nlp_projects\\english_data', 'Bag_of_Words_model.csv'), index=False)

    del df
    del test_data_features