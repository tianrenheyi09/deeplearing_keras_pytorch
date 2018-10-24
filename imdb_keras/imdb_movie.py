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
params_count = {'count_vec__binary':[True,False],'count_vec__ngram_range':[(1,1),(1,2)],'mnb__alpha':[0.1,1.0,10.0]}
params_tfidf = {'tfidf_vec__binary':[True,False],'tfidf_vec__ngram_range':[(1,1),(1,2)],'mnb__alpha':[0.1,1.0,10.0]}

#使用4折交叉验证的方式对使用CountVectorizer的朴素贝叶斯模型进行并行化超参数搜索
gs_count = GridSearchCV(pip_count,params_count,cv=4,n_jobs=-1,verbose=1)
gs_count.fit(X_train,y_train)
#输出交叉验证中最佳的准确性得分以及超参数组合
print gs_count.best_score_
print gs_count.best_params_
#以最佳的超参数组合配置模型并对测试数据进行预测
count_y_predict = gs_count.predict(X_test)

#同样使用4折交叉验证的方式对使用TfidfVectorizer的朴素贝叶斯模型进行并行化超参数搜索
gs_tfidf = GridSearchCV(pip_tfidf,params_tfidf,cv=4,n_jobs=-1,verbose=1)
gs_tfidf.fit(X_train,y_train)

#输出交叉验证中最佳的准确性得分以及超参数组合
print gs_tfidf.best_score_
print gs_tfidf.best_params_
#以最佳的超参数组合配置模型并对测试数据进行预测
tfidf_y_predict = gs_tfidf.predict(X_test)





#从本地读取未标记数据
#quoting:控制csv中的引号常量
#unlabled_train = pd.read_csv('./data/unlabeledTrainData.tsv',sep='\t',quoting=3)

import gensim
import nltk
from nltk.corpus import stopwords
 
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
 
def review_to_wordlist( review, remove_stopwords=False ):
    review_text = BeautifulSoup(review, "html.parser").get_text()
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
 
    words = review_text.lower().split()
 
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
 
    return(words)
 
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    '''
    将评论段落转换为句子，返回句子列表，每个句子由一堆词组成
    '''
    raw_sentences = tokenizer.tokenize(review.strip())
 
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            # 获取句子中的词列表
            sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))
    return sentences

sentences = []
for i, review in enumerate(train["review"]):
    sentences += review_to_sentences(review, tokenizer)
    
unlabeled_train = pd.read_csv("./data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )

for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)
print '预处理unlabeled_train data...'
print len(train_data)
print len(sentences)








#corpora(全集)，这里存储所有经过预处理后的句子的集合
corpora = []
#准备用于训练词向量的数据。
for review in unlabled_train['review']:
   corpora += review_to_sentences(review,tokenizer)

#配置训练词向量模型的超参数
num_features = 300
min_word_count = 20
num_workers = 4
context = 10
downsampling = 1e-3

#从gensim.models力导入word2vec
from gensim.models import word2vec
#开始词向量模型的训练，以corpora为训练数据，获得词向量模型
model = word2vec.Word2Vec(corpora,workers = num_workers,\
        size = num_features,min_count = min_word_count,\
        window=context,sample=downsampling)

model.init_sims(replace=True)

model_name = './data/300features_20minwords_10context'
#可以将词向量模型的训练结果长期保存于本地磁盘
model.save(model_name)

#直接读入已经训练好的词向量模型
from gensim.models import Word2Vec
model = Word2Vec.load(model_name)

#探查一下该词向量模型的训练成果
model.most_similar('man')

import numpy as np 

#定义一个函数,使用词向量产生文本特征向量；
#大致思路就是将一个句子中所有在“词汇表”中的单词，所对应的词向量累加起来；
#再除以进行了词向量转换的所有单词的个数
#这里的词汇表，就是使用unlabeledData，通过word2vec所构建的词向量模型中，生成的词汇表
#这个方法最终就是将一个句子转成特征向量的形式
def makeFeatureVec(words,model,num_features):
    #初始化一个300维，类型为float32，元素值全为0的列向量
    featureVec = np.zeros((num_features,),dtype='float32')
    nwords=0.
    #现在用model.wv.index2word来获取词汇表，先前的model.index2word已被废弃！
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords=nwords+1.
            featureVec = np.add(featureVec,model[word])
    featureVec=np.divide(featureVec,nwords)
    return featureVec

#将每一条影评句子，转化为基于词向量的特征向量(平均词向量：因为makeFeatureVec()最后除以了单词的个数，所以叫平均词向量)
#最终返回的是 n 个特征向量的集合
def getAvgFeatureVecs(reviews,model,num_features):
    counter = 0
    reviewFeatureVecs=np.zeros((len(reviews),num_features),dtype='float32')
    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(review,model,num_features)
        counter+=1
    return reviewFeatureVecs

#将train数据集和test数据集中的review，通过上面构造的词向量模型，转成最终的特征向量模型表示；
#可以发现，上面的全部操作，就是训练出一个将文本数据转成特征向量表示的模型；
#跟前面提到的CountVectorizer和TfidfVectorizer的目标是一样的。

clean_train_reviews = []
for review in train['review']:
    #首先还是将原始数据进行清洗操作，清洗完成之后，再完成特征向量的转化
    clean_train_reviews.append(review_to_text(review,remove_stopwords=False))

trainDataVecs = getAvgFeatureVecs(clean_train_reviews,model,num_features)

clean_test_reviews = []
for review in test['review']:
    clean_test_reviews.append(review_to_text(review,remove_stopwords=False))

testDataVecs = getAvgFeatureVecs(clean_test_reviews,model,num_features)

#从sklearn.ensemble导入GradientBoostingClassifier模型进行影视情感分析
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()

#配置超参数的搜索组合
params_gbc = {'n_estimators':[10,100,500],'learning_rate':[0.01,0.1,1.0],'max_depth':[2,3,4]}
gs_gbc = GridSearchCV(gbc,params_gbc,cv=4,n_jobs=-1,verbose=0)
gs_gbc.fit(trainDataVecs,y_train)

#输出网格搜索中得到的最佳的准确性得分以及超参数组合
print(gs_gbc.best_score_)
print(gs_gbc.best_params_)
#以最佳的超参数组合配置模型并对测试数据进行预测
gbc_y_predict = gs_gbc.predict(testDataVecs)
submission_gbc = pd.DataFrame({'id':test['id'],'sentiment':gbc_y_predict})
#最终的输出，还有加上quoting=3这个属性
submission_gbc.to_csv('./data/submission_gbc.csv',index=False,quoting=3)
