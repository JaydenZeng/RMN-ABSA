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


def train(sess, setting):
    with sess.as_default():
        dataset = DataSet()
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('model', reuse = None, initializer = initializer):
            m = RMN(is_training=FLAGS.train, word_embedding=wordembedding)
        global_step = tf.Variable(0, name='global_step', trainable = False)
        optimizer = tf.train.AdamOptimizer(setting.learning_rate)
        train_op = optimizer.minimize(m.total_loss, global_step=global_step)
        sess.run(tf.initialize_all_variables())
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

                if step%5==0:
                    time_str = datetime.datetime.now().isoformat()
                    tempstr = "{}: step {}, softmax_loss {:g}, rel_loss {:g}".format(time_str, step, loss_, rel_loss)
                    print (tempstr)
                    path = saver.save(sess, './model/MT_ATT_model', global_step=step)


def test(sess, setting):
    with sess.as_default():
        dataset = DataSet()
        with tf.variable_scope('model'):
            mtest = RMN(is_training=FLAGS.train, word_embedding=wordembedding)
        saver = tf.train.Saver()
        testlist = range(0, 2000, 1)
        best_iter = -1
        best_f1 = -1
        best_acc = -1

        for model_iter in testlist:
            try:
                saver.restore(sess, './model/MT_ATT_model-'+str(model_iter))
            except Exception:
                continue
            total_pred = []
            total_y = []
            all_sentence = []
            for i in range(int(len(testdata['S'])/setting.batch_size)):
                nextSentenceBatch, nextAspectBatch, nextLabelBatch, nextGraphBatch, nextPositionBatch, nextNumBatch, nextRelBatch  = dataset.nextBatch(traindata, testdata, FLAGS.train)
                feed_dict = {}
                feed_dict[mtest.input_sentence] = nextSentenceBatch
                feed_dict[mtest.asp_mask] = nextAspectBatch
                feed_dict[mtest.input_label] = nextLabelBatch
                feed_dict[mtest.input_graph] = nextGraphBatch
                feed_dict[mtest.pos_enc] = nextPositionBatch
                feed_dict[mtest.asp_num] = nextNumBatch
                feed_dict[mtest.relation] = nextRelBatch
                prob = sess.run([mtest.prob], feed_dict) #[1,32,3]
                for j in range(len(prob[0])):
                    total_pred.append(np.argmax(prob[0][j], -1))
                for item in nextLabelBatch:
                    total_y.append(item)
            f1,accuracy=evaluation(total_pred,total_y)
            print ('------------------------')
            if f1>best_f1:
                best_f1=f1
                best_iter=model_iter
            if accuracy> best_acc:
                best_acc=accuracy
            print ('model_iter:',model_iter)
            print ('f1 score:',f1)
            print ('accuracy score:',accuracy)
        with open('./bestresult.txt','w+') as fout:
            fout.write('best_iter:'+str(best_iter)+'\n')
            fout.write('best_f1:'+str(best_f1)+'\n')
            fout.write('best_acc:'+str(best_acc))
        print ('----------------------------')
        print ('best model_iter', best_iter)
        print ('best f1 score: ', best_f1)
        print ('best accuracy score:', best_acc)
   
def main(_):
#    print (FLAGS.train)
#    sys.exit()
    setting = Config()
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        
        if FLAGS.train == True:
            train(sess, setting)
        else:
            test(sess, setting)

if __name__ == '__main__':
    tf.app.run()
