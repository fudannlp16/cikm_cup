# -*- coding: utf-8 -*-
# @Time    : 2018/7/26 23:08
# @Author  : Xiaoyu Liu
# @Email   : liuxiaoyu16@fudan.edu.
from __future__ import unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import utils
import models
import random


tf.flags.DEFINE_integer('batch_size',128,'batch_size')
tf.flags.DEFINE_integer("max_epoch",100,'max_epoch')
tf.flags.DEFINE_string("model_name","HybridCNNSS","model_name")
tf.flags.DEFINE_string("model_file","HybridCNNSS","model_file")
tf.flags.DEFINE_integer('restore',0,'restore')
FLAGS=tf.flags.FLAGS


def train():
    source_data,target_data,test_data,word2id=utils.load_data()
    embeddings=utils.load_embeddings(word2id)

    random.seed(1)
    random.shuffle(target_data)

    cv_losses=[]
    for k in range(1,11):
        train_data, dev_data = utils.train_dev_split(target_data, k)
        model_file=FLAGS.model_file+str(k)
        print model_file

        print "训练集1数据大小:%d" % len(source_data)
        print "训练集2数据大小:%d" % len(train_data)
        print "验证集数据大小:%d" % len(dev_data)
        print "embedding大小:(%d,%d)"%(embeddings.shape[0],embeddings.shape[1])

        model_dir='../model'
        graph=tf.Graph()
        sess=tf.Session(graph=graph)
        with graph.as_default():
            model=getattr(models,FLAGS.model_name)(embeddings)
            saver = tf.train.Saver(tf.global_variables())
            if FLAGS.restore==1:
                saver.restore(sess, os.path.join(model_dir, FLAGS.model_file))
                print "Restore from pre-trained model"
            else:
                sess.run(tf.global_variables_initializer())
            print "Train start!"

            best_loss=1e6
            best_epoch=0
            not_improved=0
            for epoch in range(FLAGS.max_epoch):

                print epoch,"================================================"
                train_loss=[]
                ground_trues=[]
                predicts=[]

                for batch_data in utils.minibatches2(source_data,train_data,FLAGS.batch_size,ratio=1,mode='train'):
                    loss,predict=model.train_step(sess,batch_data[:3],batch_data[3])
                    train_loss.extend(loss)
                    predicts.extend(predict)
                    ground_trues.extend(batch_data[2])
                train_loss=utils.loss(ground_trues,train_loss)
                p,r,f1=utils.score(ground_trues,predicts)
                print "%d-fold Train epoch %d finished. loss:%.4f  p:%.4f r:%.4f f1:%.4f" % (k,epoch,train_loss,p,r,f1)

                valid_loss=[]
                ground_trues=[]
                predicts=[]
                for batch_data in utils.minibatches(dev_data,FLAGS.batch_size,mode='dev'):
                    loss,predict= model.valid_step(sess, batch_data, 2)
                    valid_loss.extend(loss)
                    predicts.extend(predict)
                    ground_trues.extend(batch_data[2])
                valid_loss=utils.loss(ground_trues,valid_loss)
                p, r, f1=utils.score(ground_trues, predicts)
                print "%d-fold,Valid epoch %d finished. loss:%.4f  p:%.4f r:%.4f f1:%.4f" % (k,epoch,valid_loss,p,r,f1)

                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_epoch=epoch
                    not_improved=0
                    print "save model!"
                    saver.save(sess, os.path.join(model_dir, model_file))
                else:
                    not_improved+=1
                    if not_improved>4:
                        print "停止训练!"
                        break
                print
            print "Best epoch %d  best loss %.4f" % (best_epoch,best_loss)
            print "#########################################################"
            cv_losses.append(best_loss)
    print "final cv loss: %.4f" % (sum(cv_losses) / len(cv_losses))

if __name__ == '__main__':
    train()

