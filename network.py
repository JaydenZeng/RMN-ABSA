import tensorflow as tf 
import numpy as np 
from Settings import Config
from modules import conv, conv_relu
import sys

class RMN:
    def __init__(self, is_training, word_embedding):
        self.config = Config()
        self.input_sentence = tf.placeholder(dtype = tf.int32, shape=[self.config.batch_size, self.config.max_sentence_len], name='input_sentence')
        self.asp_mask = tf.placeholder(dtype = tf.int32, shape=[None, self.config.max_sentence_len], name = 'asp_mask')
        self.input_label = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size], name = 'input_label')
        self.input_graph = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size, self.config.max_sentence_len, self.config.max_sentence_len], name = 'input_graph')
        self.pos_enc = tf.placeholder(dtype=tf.float32, shape= [None, self.config.max_sentence_len], name='pos_enc')
        self.asp_num = tf.placeholder(dtype=tf.int32, shape= [self.config.batch_size], name = 'asp_num')
        self.relation = tf.placeholder(dtype=tf.int32, shape = [None], name='relation')       


        wordembedding = tf.get_variable(initializer=word_embedding, name='word_embedding')


        enc_s = tf.cast(tf.nn.embedding_lookup(wordembedding, self.input_sentence), tf.float32)

        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size)
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size)
        
        
        if is_training:
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob = self.config.keep_prob)
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob = self.config.keep_prob)


        #sentence encoding
        with tf.variable_scope('enc_s', reuse = tf.AUTO_REUSE):
            enc_s_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = lstm_fw_cell,
                                                            cell_bw = lstm_bw_cell,
                                                            inputs = enc_s,
                                                            dtype = tf.float32,
                                                            time_major = False)
            Hs = tf.concat([enc_s_outputs[0], enc_s_outputs[1]], axis=2)
     
        #expand sentence
        cur_sen = []
        for i in range(self.config.batch_size):
            cur_sen.append(tf.tile(Hs[i], [self.asp_num[i], 1]))
        E_sen = tf.reshape(tf.concat(cur_sen,0), [-1, self.config.max_sentence_len,2*self.config.hidden_size])

        #expand graph
        cur_graph = []
        for i in range(self.config.batch_size):
            cur_graph.append(tf.tile(self.input_graph[i], [self.asp_num[i], 1]))
        input_graph = tf.reshape(tf.concat(cur_graph,0), [-1, self.config.max_sentence_len, self.config.max_sentence_len])


        with tf.variable_scope('all_weights', reuse = tf.AUTO_REUSE):

             #gcn1 weight
             w_gcn_cross1 = tf.get_variable('w_gcn_cross1', [2*self.config.hidden_size, 2*self.config.hidden_size])
             b_gcn_cross1 = tf.get_variable('b_gcn_cross1', [1, 2*self.config.hidden_size])
             w_gcn_self1 = tf.get_variable('w_gcn_self1', [2*self.config.hidden_size, 2*self.config.hidden_size])
             b_gcn_self1 = tf.get_variable('b_gcn_self1', [1, 2*self.config.hidden_size])
            
             #gcn2 weight
             w_gcn_cross2 = tf.get_variable('w_gcn_cross2', [2*self.config.hidden_size, 2*self.config.hidden_size])          
             b_gcn_cross2 = tf.get_variable('b_gcn_cross2', [1, 2*self.config.hidden_size])
             w_gcn_self2 = tf.get_variable('w_gcn_self2', [2*self.config.hidden_size, 2*self.config.hidden_size])
             b_gcn_self2 = tf.get_variable('b_gcn_self2', [1, 2*self.config.hidden_size])

             #sentence to aspect 
             W_sa = tf.get_variable('W_sa', [self.config.max_sentence_len, 2*self.config.hidden_size])            
             w_sa = tf.get_variable('w_sa', [2*self.config.hidden_size, 1])
            
             #aspect to sentence
             W_as = tf.get_variable('W_as', [self.config.max_sentence_len, 2*self.config.hidden_size])
             w_as = tf.get_variable('w_as', [2*self.config.hidden_size, 1])
   
             #aspect output
             W_l = tf.get_variable('W_l', [4*self.config.hidden_size, self.config.class_num])
             b_l = tf.get_variable('b_l', [1, self.config.class_num])

             #relation output
             W_r = tf.get_variable('W_r', [4*self.config.hidden_size, self.config.class_num])
             b_r = tf.get_variable('b_r', [1, self.config.class_num])


       #----------two layers gcn----------------------

        w_self = tf.tile(tf.eye(self.config.max_sentence_len, self.config.max_sentence_len), [tf.shape(input_graph)[0], 1])
        w_self = tf.reshape(w_self, [-1, self.config.max_sentence_len,self.config.max_sentence_len])
        input_graph = tf.cast(input_graph, tf.float32)
        w_cross = tf.subtract(input_graph, w_self)

        gcn1 = conv_relu(E_sen, w_cross, w_gcn_cross1, b_gcn_cross1) + conv_relu(E_sen, w_self, w_gcn_cross1, b_gcn_cross1)
        gcn2 = conv_relu(gcn1, w_cross, w_gcn_cross2, b_gcn_cross2) + conv_relu(gcn1, w_self, w_gcn_cross2, b_gcn_cross2)
       
        asp_mask = tf.reshape(tf.tile(self.asp_mask, [1,2*self.config.hidden_size]), [-1,2*self.config.hidden_size, self.config.max_sentence_len])
        asp_mask = tf.cast(tf.transpose(asp_mask, [0,2,1]), tf.float32)
      
        pos_weight = tf.reshape(tf.tile(self.pos_enc, [1,2*self.config.hidden_size]), [-1,2*self.config.hidden_size, self.config.max_sentence_len])
        pos_weight = tf.cast(tf.transpose(pos_weight, [0,2,1]), tf.float32)

 
        E_asp = tf.multiply(gcn2, asp_mask)
        E_sen = tf.multiply(E_sen, pos_weight)

        #------------bi-attention---------------------
        H_sa = tf.matmul(E_sen, tf.transpose(E_asp, [0, 2, 1]))
        #sentence to aspect
        u_sa = tf.tanh(tf.matmul(tf.reshape(tf.transpose(H_sa, [0,2,1]), [-1, self.config.max_sentence_len]), W_sa))
        a_sa = tf.nn.softmax(tf.reshape(tf.matmul(u_sa, w_sa), [-1, 1, self.config.max_sentence_len]))        
        V_sa = tf.squeeze(tf.matmul(a_sa, E_asp), 1)
  
        #aspect to sentence
        u_as = tf.tanh(tf.matmul(tf.reshape(H_sa, [-1, self.config.max_sentence_len]), W_as))
        a_as = tf.nn.softmax(tf.reshape(tf.matmul(u_as, w_as), [-1, 1, self.config.max_sentence_len])) 
        V_as = tf.squeeze(tf.matmul(a_sa, E_sen), 1)

        V = tf.concat([V_sa, V_as], -1)
      

        temp_new = []
        index = 0
        for i in range(self.config.batch_size):
            temp_new.append(V[index])
            index = index+self.asp_num[i]


#---------------------aspect loss--------------------------------------
        index= 0
        cur_all = []
        self.asp_reg = 0
        for i in range(self.config.batch_size):
            cur_all = V[index: index + self.asp_num[i]]
            cur_A = tf.tile(cur_all,[self.asp_num[i],1])
            cur_B=  tf.reshape(tf.tile(cur_all,[1,self.asp_num[i]]), [-1, 4*self.config.hidden_size])
            cur_A_norm = tf.sqrt(tf.reduce_sum(tf.square(cur_A), axis=-1))
            cur_B_norm = tf.sqrt(tf.reduce_sum(tf.square(cur_B), axis=-1))
            A_B = tf.reduce_sum(tf.multiply(cur_A, cur_B), axis=-1)
            self.asp_reg += tf.reduce_sum(tf.divide(A_B, tf.multiply(cur_A_norm, cur_B_norm)), -1)/tf.cast(10*self.asp_num[i]*self.asp_num[i], tf.float32)
            index = index+self.asp_num[i]




#-------------------cur_aspect classification loss----------------------------
       
        output_res = tf.add(tf.matmul(tf.tanh(temp_new), W_l), b_l)
        ouput_label = tf.one_hot(self.input_label, self.config.class_num)
        self.prob = tf.nn.softmax(output_res)
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = output_res, labels=ouput_label))

#        print (tf.trainable_variables())
#        sys.exit()
        self.l2_loss = tf.contrib.layers.apply_regularization(regularizer = tf.contrib.layers.l2_regularizer(0.0001), weights_list = [W_l, b_l]) 
        self.classify_loss = self.loss+self.l2_loss


#-------------------relation classification loss------------------------------
        cur_asp = []
        index = 0
        for i in range(self.config.batch_size):
            cur_asp.append(tf.tile(tf.expand_dims(V[index], 0), [self.asp_num[i], 1] ))
            index = index+self.asp_num[i]
        cur_asp = tf.concat(cur_asp, 0)
        rel_new = tf.abs(tf.subtract(cur_asp, V))
        
        output_rel = tf.add(tf.matmul(tf.tanh(rel_new), W_r), b_r)
        output_rel_label = tf.one_hot(self.relation, self.config.class_num)
        index = 0
        cur_loss = []
        for i in range(self.config.batch_size):
            cur_loss.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output_rel[index:index+self.asp_num[i]], labels=output_rel_label[index:index+self.asp_num[i]])))
            index = index + self.asp_num[i]
        self.rel_loss = tf.reduce_sum(cur_loss)
        self.total_loss =  self.classify_loss + self.config.r*(self.rel_loss) + self.asp_reg

