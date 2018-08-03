# -*- coding: utf-8 -*-
# @Time    : 2018/7/26 00:04
# @Author  : Xiaoyu Liu
# @Email   : liuxiaoyu16@fudan.edu.com
from __future__ import unicode_literals
from __future__ import division
import random
import gensim
import cPickle
import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score,precision_recall_fscore_support


ES_SOURCE_FILE='../data/es_source.csv'
ES_TARGET_FILE='../data/es_target.csv'
TEST_FILE='../data/es_test.csv'

PRE_TRAINED_ES_EMBEDDING_FILE='../data/wiki.es.vec'
ES_EMBEDDING_FILE='../data/es.vec'

################################################################################
#                              Read data                                       #
################################################################################

def es_tokenizer(sentence):
    # 文本清洗，去掉标点和数字，分词
    sentence=sentence.lower()
    sentence=re.sub('\d','',sentence)
    sentence=''.join([c for c in sentence if c.isalpha() or c == ' '])
    return sentence.split()

def en_tokenizer(sentence):
    raise NotImplementedError

def read_file(train=True,language='es'):
    # 读取source data, target data 文件，文本清洗，去掉标点和数字，分词
    if language=='es':
        tokenizer=es_tokenizer
    else:
        raise NotImplementedError
    # 读取文件
    if train:
        source_df=pd.read_csv(ES_SOURCE_FILE,sep='\t',header=None,encoding='utf-8')
        target_df=pd.read_csv(ES_TARGET_FILE,sep='\t',header=None,encoding='utf-8')
        source_df[0]=source_df[0].map(tokenizer)
        source_df[1]=source_df[1].map(tokenizer)
        target_df[0]=target_df[0].map(tokenizer)
        target_df[1]=target_df[1].map(tokenizer)
        return source_df.values.tolist(),target_df.values.tolist()
    else:
        test_df=pd.read_csv(TEST_FILE,sep='\t',header=None,encoding='utf-8')
        test_df[0]=test_df[0].map(tokenizer)
        test_df[1]=test_df[1].map(tokenizer)
        return test_df.values.tolist()

################################################################################
#                              Mapping                                         #
################################################################################

def build_vocab(source_data,target_data,test_data,min_df=5):
    # 创建词典
    counter={}
    for data in [source_data,target_data]:
        for s1,s2,_ in data:
            for w in s1+s2:
                counter[w]=counter.get(w,0)+1
    for s1,s2 in test_data:
        for w in s1+s2:
            counter[w]=counter.get(w,0)+1
    count_pairs=sorted(counter.items(),key=lambda x:-x[-1])

    words=['<PAD>','<UNK>']
    for word,_ in count_pairs:
        words.append(word)

    word2id={word:id for id,word in enumerate(words)}
    id2word={id:word for id,word in enumerate(words)}
    return word2id,id2word

def text_to_word_sequence(x1,x2,word2id):
    word_sequence1=[[word2id.get(w,1) for w in s] for s in x1]
    word_sequence2=[[word2id.get(w,1) for w in s] for s in x2]
    return word_sequence1,word_sequence2

def load_data(language='es'):
    """
    :param language:
    :return: source_data,target_data,test_data,word2id
    """
    # 加载处理好的数据
    source_data,target_data=read_file(train=True,language=language)
    test_data=read_file(train=False,language=language)
    word2id,id2word=build_vocab(source_data,target_data,test_data)

    source_x1,source_x2,source_label=zip(*source_data)
    target_x1,target_x2,target_label=zip(*target_data)
    test_x1,test_x2=zip(*test_data)

    source_x1,source_x2=text_to_word_sequence(source_x1,source_x2,word2id)
    target_x1,target_x2=text_to_word_sequence(target_x1,target_x2,word2id)
    test_x1,test_x2=text_to_word_sequence(test_x1,test_x2,word2id)

    return zip(source_x1,source_x2,source_label),zip(target_x1,target_x2,target_label),zip(test_x1,test_x2),word2id

def load_embeddings(word2id,lanuage='es'):
    # 加载预训练的embeddings
    if lanuage=='es':
        embedding_file=ES_EMBEDDING_FILE
        pre_trained_embedding_file=PRE_TRAINED_ES_EMBEDDING_FILE
    else:
        raise NotImplementedError
    if os.path.exists(embedding_file):
        print "loading from saved embedding_file"
        embeddings=cPickle.load(open(embedding_file,'rb'))
        embeddings[1] = np.zeros(shape=300, dtype=np.float32)
        return embeddings

    pre_trained=gensim.models.KeyedVectors.load_word2vec_format(fname=pre_trained_embedding_file,binary=False)
    emb_dim=pre_trained.wv.syn0.shape[1]
    embeddings=np.zeros(shape=[len(word2id),emb_dim], dtype=np.float32)

    count=0
    for word,id in word2id.items():
        if word in pre_trained:
            embeddings[id]=pre_trained[word]
        else:
            print word
            count+=1
    print len(word2id),count,count/len(word2id)

    cPickle.dump(embeddings,open(embedding_file,'wb'),cPickle.HIGHEST_PROTOCOL)
    return embeddings


################################################################################
#                              Batch Manager                                   #
################################################################################
def padding(sentences,maxlen=60):
    #将每个句子padding长长度60，长度不足的补0，长度超过的向后截断。
    if maxlen is None:
        maxlen=max([len(s) for s in sentences])
    pad_sentences=[]
    for sentence in sentences:
        sentence=sentence[:maxlen]
        pad=[0]*(maxlen-len(sentence))
        pad_sentences.append(sentence+pad)
    return pad_sentences

def lengths(x):
    # 计算mini-batch里面，每个句子真是长度
    result=[]
    for sentence in x:
        for i in range(len(sentence)):
            if sentence[i]==0:
                result.append(i)
                break
    return result

def minibatches(data,batch_size,mode='train'):
    # 每个step生成一个mini-batch数据
    if mode=='train':
        train,shuffle=True,True
    elif mode=='dev':
        train,shuffle=True,False
    elif mode=='test':
        train,shuffle=False,False
    else:
        raise ValueError

    if shuffle:
        random.shuffle(data)

    num_batch=len(data)//batch_size
    for i in range(num_batch):
        batch_data=data[i*batch_size:(i+1)*batch_size]
        if train:
            x1,x2,y=zip(*batch_data)
            x1,x2=padding(x1),padding(x2)
            yield x1,x2,y
        else:
            x1,x2=zip(*batch_data)
            x1,x2=padding(x1),padding(x2)
            yield x1,x2
    if num_batch*batch_size<len(data) and not shuffle:
        batch_data=data[num_batch * batch_size:]
        if train:
            x1,x2,y=zip(*batch_data)
            x1,x2=padding(x1),padding(x2)
            yield x1,x2,y
        else:
            x1,x2=zip(*batch_data)
            x1,x2=padding(x1),padding(x2)
            yield x1,x2

def minibatches2(source_data,target_data, batch_size, ratio=1, mode='train'):
    # 交替生成source data的mini-batch 和target data的mini-batch
    if mode == 'train':
        train, shuffle = True, True
    elif mode == 'dev':
        train, shuffle = True, False
    elif mode == 'test':
        train, shuffle = False, False
    else:
        raise ValueError

    if shuffle:
        random.shuffle(source_data)
        random.shuffle(target_data)
    num_batch = len(target_data) // batch_size
    source_batch_size=int(batch_size*ratio)
    target_batch_size=batch_size

    for i in range(num_batch):
        source_batch_data = source_data[i * source_batch_size:(i + 1) * source_batch_size]
        target_batch_data = target_data[i * target_batch_size:(i + 1) * target_batch_size]
        if train:
            x1, x2, y = zip(*source_batch_data)
            x1, x2 = padding(x1), padding(x2)
            yield x1, x2, y, 1
            x1, x2, y = zip(*target_batch_data)
            x1, x2 = padding(x1), padding(x2)
            yield x1, x2, y, 2
        else:
            x1, x2 = zip(*source_batch_data)
            x1, x2 = padding(x1), padding(x2)
            yield x1, x2, 1
            x1, x2 = zip(*target_batch_data)
            x1, x2 = padding(x1), padding(x2)
            yield x1, x2, 2
    if num_batch * batch_size < len(target_data) and not shuffle and not train:
        source_batch_data = source_data[num_batch * source_batch_data:]
        target_batch_data = target_data[num_batch * target_batch_size:]
        if train:
            x1, x2, y = zip(*source_batch_data)
            x1, x2 = padding(x1), padding(x2)
            yield x1, x2, y, 1
            x1, x2, y = zip(*target_batch_data)
            x1, x2 = padding(x1), padding(x2)
            yield x1, x2, y, 2
        else:
            x1, x2 = zip(*source_batch_data)
            x1, x2 = padding(x1), padding(x2)
            yield x1, x2, 1
            x1, x2 = zip(*target_batch_data)
            x1, x2 = padding(x1), padding(x2)
            yield x1, x2, 2

################################################################################
#                              Output                                          #
################################################################################
def score(y_true,y_pred):
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average='binary')
    return p,r,f

def loss(y_true,losses):
    # 计算loss,正负样本loss权重 0.8228:2.0096
    loss0=[]
    loss1=[]
    for i in range(len(y_true)):
        if y_true[i]==0:
            loss0.append(losses[i]*0.8228)
        else:
            loss1.append(losses[i]*2.0096)
    print np.mean(loss0)/0.8228,np.mean(loss1)/2.0096
    return sum(loss1+loss0)/len(loss0+loss1)

def generate_file(y_pred,filename='../submit/submission.csv'):
    pd.Series(y_pred).to_csv(filename,index=False,header=None)

def train_dev_split(data,cv=1):
    # 生成10-fold cv
    l=len(data)
    s1,s2,s3,s4,s5,s6,s7,s8,s9=int(l/10),int(2*l/10),int(3*l/10),int(4*l/10),int(5*l/10),int(6*l/10),int(7*l/10),int(8*l/10),int(9*l/10)
    train_data1,dev_data1=data[:s9],data[s9:]
    train_data2,dev_data2=data[s1:],data[:s1]
    train_data3,dev_data3=data[:s1]+data[s2:],data[s1:s2]
    train_data4,dev_data4=data[:s2]+data[s3:],data[s2:s3]
    train_data5,dev_data5=data[:s3]+data[s4:],data[s3:s4]
    train_data6,dev_data6=data[:s4]+data[s5:],data[s4:s5]
    train_data7,dev_data7=data[:s5]+data[s6:],data[s5:s6]
    train_data8,dev_data8=data[:s6]+data[s7:],data[s6:s7]
    train_data9,dev_data9=data[:s7]+data[s8:],data[s7:s8]
    train_data10,dev_data10=data[:s8]+data[s9:],data[s8:s9]

    if cv==1:
        return train_data1,dev_data1
    elif cv==2:
        return train_data2,dev_data2
    elif cv==3:
        return train_data3,dev_data3
    elif cv==4:
        return train_data4,dev_data4
    elif cv==5:
        return train_data5,dev_data5
    elif cv==6:
        return train_data6,dev_data6
    elif cv==7:
        return train_data7,dev_data7
    elif cv==8:
        return train_data8,dev_data8
    elif cv==9:
        return train_data9,dev_data9
    elif cv==10:
        return train_data10, dev_data10
    else:
        raise ValueError

if __name__ == '__main__':
    pass















