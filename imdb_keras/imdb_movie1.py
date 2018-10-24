# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 17:22:10 2018

@author: 1
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:28:14 2018

@author: 1
"""



import os
os.listdir("./data")
import pandas as pd 
#这里的read_csv方法，加上了sep参数，把原始数据的每一行按'\t'进行分割,默认是','
train = pd.read_csv('./data/labeledTrainData.tsv',sep='\t')
test  = pd.read_csv('./data/testData.tsv',sep='\t')

from bs4 import BeautifulSoup
#导入正则表达式工具包
import re 
#从nltk.corpus里导入停用词列表
from nltk.corpus import stopwords
import nltk
#定义review_to_text函数，完成对原始评论的三项数据预处理任务
def review_to_text(review,remove_stopwords):
    #任务一：去掉html标记。
    raw_text = BeautifulSoup(review,'html').get_text()
    #任务二：去掉非字母字符,sub(pattern, replacement, string) 用空格代替
    letters = re.sub('[^a-zA-Z]',' ',raw_text)
    #str.split(str="", num=string.count(str)) 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则仅分隔 num 个子字符串
    #这里是先将句子转成小写字母表示，再按照空格划分为单词list
    words = letters.lower().split()
    #任务三：如果remove_stopwords被激活，则去除评论中的停用词
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        #过滤掉停用词
        words = [w for w in words if w not in stop_words]
    return words
#必须空一行

#分别对原始数据和测试数据集进行上述三项处理
X_train = []
for review in train['review']:
    X_train.append(' '.join(review_to_text(review,False)))
#必须空一行

X_test = []
for review in test['review']:
    X_test.append(' '.join(review_to_text(review,False)))
#必须空一行
y_train = train['sentiment']


#导入文本特性抽取器CountVectorizer和TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
#导入Pipeline用于方便搭建系统流程
from sklearn.pipeline import Pipeline 
#导入GridSearchCV用于超参数组合的网格搜索
from sklearn.grid_search import GridSearchCV 

#使用Pipeline搭建两组使用朴素贝叶斯模型的分类器，区别在于分别使用CountVectorizer和TfidfVectorizer对文本特征进行提取
#[]里面的参数，(a,b)是一种赋值操作，表示 a = b
pip_count = Pipeline([('count_vec',CountVectorizer(analyzer='word')),('mnb',MultinomialNB())])
pip_tfidf = Pipeline([('tfidf_vec',TfidfVectorizer(analyzer='word')),('mnb',MultinomialNB())])

#分别配置用于模型超参数搜索的组合

#*****注意:模型名与属性名称之间，一定要用双下划线__连接****
params_count = {'count_vec__binary':[True,False],'count_vec__ngram_range':[(1,1),(1,2),(1,3)],'mnb__alpha':[0.1,1.0,10.0]}
params_tfidf = {'tfidf_vec__binary':[True,False],'tfidf_vec__ngram_range':[(1,1),(1,2),(1,3)],'mnb__alpha':[0.1,1.0,10.0]}

#使用4折交叉验证的方式对使用CountVectorizer的朴素贝叶斯模型进行并行化超参数搜索
gs_count = GridSearchCV(pip_count,params_count,cv=4,n_jobs=-1,verbose=0)
gs_count.fit(X_train,y_train)
#输出交叉验证中最佳的准确性得分以及超参数组合
print(gs_count.best_score_)
print(gs_count.best_params_)
#以最佳的超参数组合配置模型并对测试数据进行预测
count_y_predict = gs_count.predict(X_test)

#同样使用4折交叉验证的方式对使用TfidfVectorizer的朴素贝叶斯模型进行并行化超参数搜索
gs_tfidf = GridSearchCV(pip_tfidf,params_tfidf,cv=4,n_jobs=-1,verbose=1)
gs_tfidf.fit(X_train,y_train)

#输出交叉验证中最佳的准确性得分以及超参数组合
print(gs_tfidf.best_score_)
print(gs_tfidf.best_params_)
#以最佳的超参数组合配置模型并对测试数据进行预测
tfidf_y_predict = gs_tfidf.predict(X_test)



from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
# 参考：https://blog.csdn.net/longxinchen_ml/article/details/50629613
tfidf = TFIDF(min_df=2, # 最小支持度为2
           max_features=100000,
           strip_accents='unicode',
           analyzer='word',
           token_pattern=r'\w{1,}',
           ngram_range=(1, 3),  # 二元文法模型
           use_idf=1,
           smooth_idf=1,
           sublinear_tf=1,
           stop_words = 'english') # 去掉英文停用词
 
# 合并训练和测试集以便进行TFIDF向量化操作
data_all = X_train + X_test
len_train = len(X_train)
 
tfidf.fit(data_all)
data_all = tfidf.transform(data_all)
# 恢复成训练集和测试集部分
train_x = data_all[:len_train]
test_x = data_all[len_train:]
print('TF-IDF处理结束.')

from sklearn.naive_bayes import MultinomialNB as MNB
 
model_NB = MNB()
model_NB.fit(train_x,y_train)
MNB(alpha=1.0, class_prior=None, fit_prior=True)
 
from sklearn.cross_validation import cross_val_score
import numpy as np
 
print("多项式贝叶斯分类器10折交叉验证得分: ", np.mean(cross_val_score(model_NB, train_x, y_train, cv=10, scoring='roc_auc')))












