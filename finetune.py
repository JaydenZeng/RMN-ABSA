import tensorflow as tf 
import os
import sys
import datetime
import numpy as np
from Settings import Config
from Dataset import DataSet
from network import RMN
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pickle

import logging
logging.getLogger('tensorflow').disabled = True

os.environ['CUDA_VISIBLE_DEVICES']='6,7'
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('train', True, 'set True to train')


dataset = DataSet()
domain = dataset.domain
wordembedding = pickle.load(open('./data/'+domain+'/glove_300.pkl', 'rb')).astype(np.float32)
traindata = pickle.load(open('./data/'+domain+'/train.pkl', 'rb'))
testdata  = pickle.load(open('./data/'+domain+'/test.pkl', 'rb'))

def evaluation(y_pred, y_true):
    f1_s = f1_score(y_true, y_pred, average='macro')
    accuracy_s = accuracy_score(y_true, y_pred)
    return f1_s, accuracy_s

best_iter = 350
def finetune(sess, setting):
    with sess.as_default():
        dataset = DataSet()
        with tf.variable_scope('model', reuse = None):
            m = RMN(is_training=FLAGS.train, word_embedding=wordembedding)
        global_step = tf.Variable(0, name='global_step', trainable = False)
        optimizer = tf.train.AdamOptimizer(0.0001)
        train_op = optimizer.minimize(m.total_loss, global_step=global_step)
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, './model/MT_ATT_model-'+str(best_iter))
        saver = tf.train.Saver(max_to_keep=None)
        for epoch in range(setting.epoch_num):
            for i in range(int(len(traindata['S'])/setting.batch_size)):
                nextSentenceBatch, nextAspectBatch, nextLabelBatch, nextGraphBatch, nextPositionBatch, nextNumBatch, nextRelBatch = dataset.nextBatch(traindata, testdata, FLAGS.train)
                feed_dict = {}
                feed_dict[m.input_sentence] = nextSentenceBatch
                feed_dict[m.asp_mask] = nextAspectBatch
                feed_dict[m.input_label] = nextLabelBatch
                feed_dict[m.input_graph] = nextGraphBatch
                feed_dict[m.pos_enc] = nextPositionBatch
                feed_dict[m.asp_num] = nextNumBatch
                feed_dict[m.relation] = nextRelBatch
                temp, step, loss_, rel_loss = sess.run([train_op, global_step, m.loss, m.rel_loss], feed_dict)

                if step<best_iter+30:
                    time_str = datetime.datetime.now().isoformat()
                    tempstr = "{}: step {}, softmax_loss {:g}, rel_loss {:g}".format(time_str, step, loss_, rel_loss)
                    print (tempstr)
                    path = saver.save(sess, './model/MT_ATT_model', global_step=step)
                else:
                    sys.exit()
     


def main(_):
#    print (FLAGS.train)
#    sys.exit()
    setting = Config()
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        
        finetune(sess, setting)

if __name__ == '__main__':
    tf.app.run()
