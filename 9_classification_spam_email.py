# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from tflearn.layers.normalization import local_response_normalization
import os
import numpy as np
import tensorflow as tf
import tflearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as metrics
 
max_features=5000
max_document_length=100
#将整个邮件当成一个字符串处理，其中过滤回车和换行
def load_one_file(filename):
    x=""
    with open(filename,'r',encoding='utf-8',errors='ignore') as f:
        for line in f:
            line=line.strip('\n')
            line=line.strip('\r')
            x+=line
    return x
#遍历指定文件夹下所有文件，加载数据
def load_files_from_dir(rootdir):
    x=[]
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            v=load_one_file(path)
            x.append(v)
    return x

def load_all_files():
    ham=[]
    spam=[]
    for i in range(1,3):
        path = "./enron%d/ham/" % i
        print("Load %s" % path)
        ham+=load_files_from_dir(path)
        path = "./enron%d/spam/" % i
        print("Load %s" % path)
        spam+=load_files_from_dir(path)
    return ham,spam

#使用词袋模型，向量化邮件样本，ham标记为0，spam标记为1
def get_features_by_wordbag():
    ham, spam=load_all_files()
    x=ham+spam
    y=[0]*len(ham)+[1]*len(spam)
    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    print(vectorizer)
    x=vectorizer.fit_transform(x)
    x=x.toarray()
    return x,y

#构建贝叶斯模型
def do_nb_wordbag(x_train, x_test, y_train, y_test):
    print("NB and wordbag")
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
 
if __name__ == "__main__":
    print("Hello spam-mail")
    print("get_features_by_wordbag")
    x,y=get_features_by_wordbag()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)  #测试集比例为30%
    do_nb_wordbag(x_train, x_test, y_train, y_test)