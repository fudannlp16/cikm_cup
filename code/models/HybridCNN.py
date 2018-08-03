# -*- coding: utf-8 -*-
# @Time    : 2018/7/26 21:41
# @Author  : Xiaoyu Liu
# @Email   : liuxiaoyu16@fudan.edu.com
import tensorflow as tf
import utils
import numpy as np

class HybridCNN(object):

    def __init__(self,embeddings):

        self.dropout = 0.2
        # BCNN
        self.kernal_size=[1,2,3]
        self.filters=[128,128,128]
        # Pyramid
        self.m_kernal_size=[3,1]
        self.m_strides=[1,3]
        self.m_filters=[16,32]
        self.m_pool_sizes=[4,2]

        self.init_lr=0.001
        self.pos_weights= 2.4422

        self.x1 = tf.placeholder(dtype=tf.int32, shape=[None, 60], name='x1')
        self.x2 = tf.placeholder(dtype=tf.int32, shape=[None, 60], name='x2')
        self.label = tf.placeholder(dtype=tf.float32, shape=[None], name='label')
        self.dropout_keep = tf.placeholder(dtype=tf.float32, name='dropout_keep')
        self.features = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='features')
        self.training=self.dropout_keep==1.0

        self.embeddings = tf.get_variable(name='embeddings',
                                          initializer=embeddings,
                                          dtype=tf.float32,
                                          trainable=False)

        self.emb_inputs1 = tf.nn.embedding_lookup(self.embeddings, self.x1)
        self.emb_inputs2 = tf.nn.embedding_lookup(self.embeddings, self.x2)
        self.emb_inputs1 = tf.nn.dropout(self.emb_inputs1, self.dropout_keep)
        self.emb_inputs2 = tf.nn.dropout(self.emb_inputs2, self.dropout_keep)

        with tf.variable_scope('BCNN'):
            pooled_outputs1=[]
            pooled_outputs2=[]
            for i in range(len(self.filters)):
                Conv=tf.layers.Conv1D(
                    filters=self.filters[i],
                    kernel_size=self.kernal_size[i],
                    activation=tf.nn.relu,
                    name='conv%d'% i
                )
                conv1=Conv(self.emb_inputs1)
                conv2=Conv(self.emb_inputs2)
                pool1=tf.reduce_max(conv1,axis=1)
                pool2=tf.reduce_max(conv2,axis=1)
                pooled_outputs1.append(pool1)
                pooled_outputs2.append(pool2)
            h1=tf.concat(pooled_outputs1,axis=-1)
            h2=tf.concat(pooled_outputs2,axis=-1)
            hb=tf.concat([tf.abs(h1-h2),(h1*h2)],axis=-1)

        with tf.variable_scope('Pyramid'):
            M=tf.einsum('abc,adc->abd',self.emb_inputs1,self.emb_inputs2)
            M=tf.expand_dims(M,axis=3)
            conv1=tf.layers.conv2d(
                inputs=M,
                kernel_size=self.m_kernal_size[0],
                filters=self.m_filters[0],
                strides=self.m_strides[0],
                activation=tf.nn.relu
            )
            pool1=tf.layers.max_pooling2d(
                inputs=conv1,
                pool_size=self.m_pool_sizes[0],
                strides=self.m_pool_sizes[0]
            )
            conv2=tf.layers.conv2d(
                inputs=pool1,
                kernel_size=self.m_kernal_size[1],
                filters=self.m_filters[1],
                strides=self.m_strides[1],
                activation=tf.nn.relu
            )
            pool2=tf.layers.max_pooling2d(
                inputs=conv2,
                pool_size=self.m_pool_sizes[1],
                strides=self.m_pool_sizes[1]
            )
            hp=tf.reshape(pool2,[-1,2*2*self.m_filters[-1]])

        h=tf.concat([hb,hp],axis=-1)

        features = tf.layers.dense(self.features, 32, activation=tf.nn.relu)
        self.logits=tf.nn.dropout(tf.concat([h,features],axis=-1),self.dropout_keep)
        self.logits=tf.layers.dense(self.logits,100,activation=tf.nn.tanh)
        self.logits=tf.nn.dropout(self.logits,self.dropout_keep)
        self.logits=tf.layers.dense(self.logits,1,activation=None)
        self.logits=tf.reshape(self.logits,shape=[-1])

        self.prob = tf.nn.sigmoid(self.logits)
        self.predict = tf.cast(tf.greater(self.prob, 0.5), tf.int32)

        self.loss_output = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logits)
        self.loss_train = tf.nn.weighted_cross_entropy_with_logits(targets=self.label,logits=self.logits,pos_weight=self.pos_weights)
        self.loss = tf.reduce_mean(self.loss_train)

        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        tvars = tf.trainable_variables()
        clipped_gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clip_norm=5)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.init_lr)
        self.train_op = optimizer.apply_gradients(zip(clipped_gradients, tvars), global_step=self.global_step)

    def train_step(self,sess,batch_data):
        x1,x2,label=batch_data
        feed_dict={
            self.x1:x1,
            self.x2:x2,
            self.label:label,
            self.dropout_keep:1-self.dropout,
            self.features: self.feature1_step(x1, x2)
        }
        _,loss,predict=sess.run([self.train_op,self.loss_output,self.predict],feed_dict)
        predict=predict.tolist()
        return loss,predict

    def valid_step(self,sess,batch_data):
        x1,x2,label=batch_data
        feed_dict={
            self.x1:x1,
            self.x2:x2,
            self.label:label,
            self.dropout_keep:1.0,
            self.features: self.feature1_step(x1, x2)
        }
        loss,predict=sess.run([self.loss_output,self.predict],feed_dict)
        predict=predict.tolist()
        return loss,predict

    def infer_step(self,sess,batch_data):
        x1, x2=batch_data
        feed_dict={
            self.x1:x1,
            self.x2:x2,
            self.dropout_keep:1.0,
            self.features: self.feature1_step(x1, x2)
        }
        predict=sess.run(self.prob,feed_dict)
        predict=predict.tolist()
        return predict

    def feature1_step(self,x1,x2):
        result=[]
        s1=utils.lengths(x1)
        s2=utils.lengths(x2)
        for i in range(len(x1)):
            c1,c2=0,0
            for item in x1[i]:
                if item==0:
                    break
                if item in x2[i]:
                    c1+=1
            for item in x2[i]:
                if item==0:
                    break
                if item in x1[i]:
                    c2+=1
            result.append((c1+c2)*1.0/(s1[i]+s2[i]))
        return np.expand_dims(np.array(result),1)


if __name__ == '__main__':
    import numpy as np
    embeddings=np.zeros(shape=[100,14],dtype=np.float32)
    model=HybridCNN(embeddings)





