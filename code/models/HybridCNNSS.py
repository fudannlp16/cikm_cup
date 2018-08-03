# -*- coding: utf-8 -*-
# @Time    : 2018/7/26 21:41
# @Author  : Xiaoyu Liu
# @Email   : liuxiaoyu16@fudan.edu.com
import tensorflow as tf
import numpy as np
import utils

class HybridCNNSS(object):

    def __init__(self,embeddings):

        self.dropout = 0.2
        # BCNN
        self.kernal_size=[1,2,3]
        self.filters=[128,128,128]
        # Pyramid
        self.m_kernal_size = [3,1]
        self.m_strides = [1,3]
        self.m_filters = [16,32]
        self.m_pool_sizes = [4,2]

        self.init_lr = 0.001
        self.pos_weights = 2.4422

        self.lambda1=0.05
        self.lambda2=0.05
        self.lambda3=0.05

        self.x1 = tf.placeholder(dtype=tf.int32, shape=[None, 60], name='x1')
        self.x2 = tf.placeholder(dtype=tf.int32, shape=[None, 60], name='x2')
        self.label = tf.placeholder(dtype=tf.float32, shape=[None], name='label')
        self.dropout_keep = tf.placeholder(dtype=tf.float32, name='dropout_keep')
        self.training=self.dropout_keep==1.0
        self.features=tf.placeholder(dtype=tf.float32,shape=[None,1],name='features')

        self.embeddings = tf.get_variable(name='embeddings',
                                          initializer=embeddings,
                                          dtype=tf.float32,
                                          trainable=False)

        self.emb_inputs1 = tf.nn.embedding_lookup(self.embeddings, self.x1)
        self.emb_inputs2 = tf.nn.embedding_lookup(self.embeddings, self.x2)
        self.emb_inputs1 = tf.nn.dropout(self.emb_inputs1, self.dropout_keep)
        self.emb_inputs2 = tf.nn.dropout(self.emb_inputs2, self.dropout_keep)

        with tf.variable_scope('Shared-NN'):
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
            hc=tf.concat([hb,hp],axis=-1)

        with tf.variable_scope('Source-NN'):
            with tf.variable_scope('BCNN'):
                pooled_outputs1 = []
                pooled_outputs2 = []
                for i in range(len(self.filters)):
                    Conv = tf.layers.Conv1D(
                        filters=self.filters[i],
                        kernel_size=self.kernal_size[i],
                        activation=tf.nn.relu,
                        name='conv%d' % i
                    )
                    conv1 = Conv(self.emb_inputs1)
                    conv2 = Conv(self.emb_inputs2)
                    pool1 = tf.reduce_max(conv1, axis=1)
                    pool2 = tf.reduce_max(conv2, axis=1)
                    pooled_outputs1.append(pool1)
                    pooled_outputs2.append(pool2)
                h1 = tf.concat(pooled_outputs1, axis=-1)
                h2 = tf.concat(pooled_outputs2, axis=-1)
                hb = tf.concat([tf.abs(h1 - h2), (h1 * h2)], axis=-1)
            with tf.variable_scope('Pyramid'):
                M = tf.einsum('abc,adc->abd', self.emb_inputs1, self.emb_inputs2)
                M = tf.expand_dims(M, axis=3)
                conv1 = tf.layers.conv2d(
                    inputs=M,
                    kernel_size=self.m_kernal_size[0],
                    filters=self.m_filters[0],
                    strides=self.m_strides[0],
                    activation=tf.nn.relu
                )
                pool1 = tf.layers.max_pooling2d(
                    inputs=conv1,
                    pool_size=self.m_pool_sizes[0],
                    strides=self.m_pool_sizes[0]
                )
                conv2 = tf.layers.conv2d(
                    inputs=pool1,
                    kernel_size=self.m_kernal_size[1],
                    filters=self.m_filters[1],
                    strides=self.m_strides[1],
                    activation=tf.nn.relu
                )
                pool2 = tf.layers.max_pooling2d(
                    inputs=conv2,
                    pool_size=self.m_pool_sizes[1],
                    strides=self.m_pool_sizes[1]
                )
                hp = tf.reshape(pool2, [-1, 2 * 2 * self.m_filters[-1]])
            hs = tf.concat([hb,hp], axis=-1)

        with tf.variable_scope('Target-NN'):
            with tf.variable_scope('BCNN'):
                pooled_outputs1 = []
                pooled_outputs2 = []
                for i in range(len(self.filters)):
                    Conv = tf.layers.Conv1D(
                        filters=self.filters[i],
                        kernel_size=self.kernal_size[i],
                        activation=tf.nn.relu,
                        name='conv%d' % i
                    )
                    conv1 = Conv(self.emb_inputs1)
                    conv2 = Conv(self.emb_inputs2)
                    pool1 = tf.reduce_max(conv1, axis=1)
                    pool2 = tf.reduce_max(conv2, axis=1)
                    pooled_outputs1.append(pool1)
                    pooled_outputs2.append(pool2)
                h1 = tf.concat(pooled_outputs1, axis=-1)
                h2 = tf.concat(pooled_outputs2, axis=-1)
                hb = tf.concat([tf.abs(h1 - h2), (h1 * h2)], axis=-1)
            with tf.variable_scope('Pyramid'):
                M = tf.einsum('abc,adc->abd', self.emb_inputs1, self.emb_inputs2)
                M = tf.expand_dims(M, axis=3)
                conv1 = tf.layers.conv2d(
                    inputs=M,
                    kernel_size=self.m_kernal_size[0],
                    filters=self.m_filters[0],
                    strides=self.m_strides[0],
                    activation=tf.nn.relu
                )
                pool1 = tf.layers.max_pooling2d(
                    inputs=conv1,
                    pool_size=self.m_pool_sizes[0],
                    strides=self.m_pool_sizes[0]
                )
                conv2 = tf.layers.conv2d(
                    inputs=pool1,
                    kernel_size=self.m_kernal_size[1],
                    filters=self.m_filters[1],
                    strides=self.m_strides[1],
                    activation=tf.nn.relu
                )
                pool2 = tf.layers.max_pooling2d(
                    inputs=conv2,
                    pool_size=self.m_pool_sizes[1],
                    strides=self.m_pool_sizes[1]
                )
                hp = tf.reshape(pool2, [-1, 2 * 2 * self.m_filters[-1]])
            ht = tf.concat([hb,hp], axis=-1)

        self.logits_source=tf.nn.dropout(tf.concat([hs,hc],axis=-1),self.dropout_keep)
        self.logits_source=tf.layers.dense(self.logits_source,100,activation=tf.nn.tanh)
        self.logits_source=tf.nn.dropout(self.logits_source,self.dropout_keep)
        self.logits_source=tf.layers.dense(self.logits_source,1,activation=None)
        self.logits_source=tf.reshape(self.logits_source,shape=[-1])
        self.prob_source = tf.nn.sigmoid(self.logits_source)
        self.predict_source = tf.cast(tf.greater(self.prob_source, 0.5), tf.int32)

        features=tf.layers.dense(self.features,32,activation=tf.nn.relu)
        self.logits_target = tf.nn.dropout(tf.concat([ht,hc,features], axis=-1),self.dropout_keep)
        self.logits_target = tf.layers.dense(self.logits_target, 100, activation=tf.nn.tanh)
        self.logits_target = tf.nn.dropout(self.logits_target,self.dropout_keep)
        self.logits_target = tf.layers.dense(self.logits_target, 1, activation=None)
        self.logits_target = tf.reshape(self.logits_target, shape=[-1])
        self.prob_target = tf.nn.sigmoid(self.logits_target)
        self.predict_target = tf.cast(tf.greater(self.prob_target, 0.5), tf.int32)

        self.logits_adv = tf.nn.dropout(hc,self.dropout_keep)
        self.logits_adv = tf.layers.dense(self.logits_adv,1,activation=None)
        self.prob_adv = tf.reshape(tf.sigmoid(self.logits_adv),shape=[-1])
        self.loss_adv = self.prob_adv*tf.log(self.prob_adv)+(1-self.prob_adv)*tf.log(1-self.prob_adv)
        self.loss_adv = tf.reduce_mean(self.loss_adv)

        self.logits_ds = tf.nn.dropout(hs,self.dropout_keep)
        self.logits_ds = tf.layers.dense(self.logits_ds,1,activation=None)
        self.prob_ds = tf.reshape(tf.sigmoid(self.logits_ds),shape=[-1])
        self.loss_ds = -self.label*tf.log(self.prob_ds)
        self.loss_ds = tf.reduce_mean(self.loss_ds)

        self.logits_dt = tf.nn.dropout(ht,self.dropout_keep)
        self.logits_dt = tf.layers.dense(self.logits_dt,1,activation=None)
        self.prob_dt = tf.reshape(tf.sigmoid(self.logits_dt),shape=[-1])
        self.loss_dt = -self.label*tf.log(self.prob_dt)
        self.loss_dt = tf.reduce_mean(self.loss_dt)

        self.loss_output_source = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logits_source)
        self.loss_train_source = tf.nn.weighted_cross_entropy_with_logits(targets=self.label,logits=self.logits_source,pos_weight=self.pos_weights)
        self.loss_source = tf.reduce_mean(self.loss_train_source)
        self.loss_source = self.loss_source + self.lambda1*self.loss_adv + self.lambda2*self.loss_ds

        self.loss_output_target = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logits_target)
        self.loss_train_target = tf.nn.weighted_cross_entropy_with_logits(targets=self.label, logits=self.logits_target,pos_weight=self.pos_weights)
        self.loss_target = tf.reduce_mean(self.loss_output_target)
        self.loss_target = self.loss_target + self.lambda1*self.loss_adv + self.lambda3*self.loss_dt

        self.global_step_source = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.global_step_target = tf.Variable(0, dtype=tf.int32, trainable=False)
        tvars = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(learning_rate=self.init_lr)

        clipped_gradients_source, _ = tf.clip_by_global_norm(tf.gradients(self.loss_source, tvars), clip_norm=5)
        self.train_op_source = optimizer.apply_gradients(zip(clipped_gradients_source, tvars), global_step=self.global_step_source)
        clipped_gradients_target, _ = tf.clip_by_global_norm(tf.gradients(self.loss_target, tvars), clip_norm=5)
        self.train_op_target = optimizer.apply_gradients(zip(clipped_gradients_target, tvars), global_step=self.global_step_target)

    def train_step(self,sess,batch_data,task):
        x1,x2,label=batch_data
        feed_dict={
            self.x1:x1,
            self.x2:x2,
            self.label:label,
            self.dropout_keep:1-self.dropout,
            self.features: self.feature1_step(x1, x2)
        }
        if task=='source':
            _,loss,predict=sess.run([self.train_op_source,self.loss_output_source,self.predict_source],feed_dict)
        else:
            _,loss,predict=sess.run([self.train_op_target,self.loss_output_target,self.predict_target],feed_dict)
        predict=predict.tolist()
        return loss,predict

    def valid_step(self,sess,batch_data,task):
        x1,x2,label=batch_data
        feed_dict={
            self.x1:x1,
            self.x2:x2,
            self.label:label,
            self.dropout_keep:1.0,
            self.features:self.feature1_step(x1,x2)
        }
        if task=='source':
            loss,predict=sess.run([self.loss_output_source,self.predict_source],feed_dict)
        else:
            loss,predict=sess.run([self.loss_output_target,self.predict_target],feed_dict)
        predict=predict.tolist()
        return loss,predict

    def infer_step(self,sess,batch_data,task):
        x1, x2=batch_data
        feed_dict={
            self.x1:x1,
            self.x2:x2,
            self.dropout_keep:1.0,
            self.features: self.feature1_step(x1, x2)

        }
        if task=='source':
            predict=sess.run(self.prob_source,feed_dict)
        else:
            predict=sess.run(self.prob_target,feed_dict)
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
    model=HybridCNNSS(embeddings)






