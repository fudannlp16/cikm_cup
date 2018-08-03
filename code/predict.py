# -*- coding: utf-8 -*-
# @Time    : 2018/7/26 23:08
# @Author  : Xiaoyu Liu
# @Email   : liuxiaoyu16@fudan.edu.
from __future__ import unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import utils
import models
import random


class Graph(object):
    """ Create model graph """
    def __init__(self,model_name,model_file,embeddings):
        print "loading model_name:%s   model_file:%s" % (model_name,model_file)
        self.graph=tf.Graph()
        self.sess=tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.model=getattr(models,model_name)(embeddings)
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(self.sess, os.path.join('../model',model_file))

    def run(self,test_data):
        predicts = []
        for batch_data in utils.minibatches(test_data, 128, mode='test'):
            predict =self.model.infer_step(self.sess, batch_data)
            predicts.extend(predict)
        return predicts

def predict():
    source_data,target_data,test_data,word2id=utils.load_data()
    embeddings=utils.load_embeddings(word2id)

    print "测试集大小 %d" % len(test_data)

    results=[]

    #HybridCNNSS
    g1 = Graph('HybridCNN', 'HybridCNN1', embeddings)
    results.append(g1.run(test_data))
    g1 = Graph('HybridCNN', 'HybridCNN2', embeddings)
    results.append(g1.run(test_data))
    g1 = Graph('HybridCNN', 'HybridCNN3', embeddings)
    results.append(g1.run(test_data))
    g1 = Graph('HybridCNN', 'HybridCNN4', embeddings)
    results.append(g1.run(test_data))
    g1 = Graph('HybridCNN', 'HybridCNN5', embeddings)
    results.append(g1.run(test_data))
    g1 = Graph('HybridCNN', 'HybridCNN6', embeddings)
    results.append(g1.run(test_data))
    g1 = Graph('HybridCNN', 'HybridCNN7', embeddings)
    results.append(g1.run(test_data))
    g1 = Graph('HybridCNN', 'HybridCNN8', embeddings)
    results.append(g1.run(test_data))
    g1 = Graph('HybridCNN', 'HybridCNN9', embeddings)
    results.append(g1.run(test_data))
    g1 = Graph('HybridCNN', 'HybridCNN10', embeddings)
    results.append(g1.run(test_data))

    predicts=[]
    for predict in np.stack(results,axis=1):
        predicts.append(1.0*sum(predict)/len(predict))

    utils.generate_file(predicts)
if __name__ == '__main__':
    predict()


