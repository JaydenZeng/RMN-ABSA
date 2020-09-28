# -*- coding: UTF-8 -*-
from Settings import Config
import re
import os
import sys
import numpy as np
import pickle
import spacy


class DataSet:
    def __init__(self):
        self.config = Config()
        self.iter_num = 0
        self.domain = self.config.domain
        self.nlp = spacy.load('en_core_web_sm')
        self.word2idx = pickle.load(open('./data/'+self.domain+'/word2idx.pkl', 'rb')) 
        self.idx2word = {v:k for k,v in self.word2idx.items()}

        self.trainfile= './data/'+self.domain+'/train.raw'
        self.testfile = './data/'+self.domain+'/test.raw'
        self.relation_set = ['similar', 'unrelated', 'opposite']
        self.label_set = ['-1', '0', '1']
        self.max_sentence_len = self.config.max_sentence_len
     
    def replace_symbol(self, text):
        text = text.replace('(', '-LRB- ')
        text = text.replace(')', ' -RRB-')
        text = text.replace("`", "'")
        return text    



    def text_padding(self, text, max_len):
        res = []
        text = text.lower()
        words = text.split()
        cur_len = len(words)
        if cur_len>max_len:
           cur_len = max_len
        cur_padding = np.zeros(max_len).astype('int32')
        for i in range(cur_len):
            if words[i] in self.word2idx:
               cur_padding[i] = self.word2idx[words[i]]
            else:
               cur_padding[i] = 1
          
        return cur_padding

    def aspect_mask(self, aspect_index, max_len):
        #mark aspect index as 1
        asp_mask = np.zeros(max_len).astype('int32')
        for i in range(aspect_index[0],aspect_index[1]):
            asp_mask[i] = 1 
        return asp_mask

    def dependency_adj_matrix(self, text, max_len):
        # https://spacy.io/docs/usage/processing-text
        document = self.nlp(text)
        seq_len = len(text.split())
        if seq_len > max_len:
           seq_len = max_len
        matrix = np.zeros((max_len, max_len)).astype('int32')
    
        for token in document:
            if token.i < seq_len:
                matrix[token.i][token.i] = 1
                # https://spacy.io/docs/api/token
                for child in token.children:
                    if child.i < seq_len:
                        matrix[token.i][child.i] = 1
                        matrix[child.i][token.i] = 1
        for i in range(max_len):
            matrix[i][i] = 1
        return matrix
    


    def pos_enc(self, sentence, aspect_index, dis):
        max_len = len(sentence)
        index_not_zero = np.count_nonzero(sentence)
        weight = np.zeros(max_len).astype('float32')
        for i in range(index_not_zero):
            if i < aspect_index[0] and  aspect_index[0]-i < dis:
               weight[i] = 1- (aspect_index[0]-i)*1.0/index_not_zero
            if i >= aspect_index[0] and i<=aspect_index[1]:
               weight[i] = 1
            if i> aspect_index[1] and i-aspect_index[1] <dis:
               weight[i] = 1- (i-aspect_index[1])*1.0/index_not_zero
        return weight



    def get_relation(self, inputs):
        all_case = [['1', '1'], ['1', '0'], ['1', '-1'],
                    ['0', '1'], ['0', '0'], ['0', '-1'],
                    ['-1', '1'], ['-1', '0'], ['-1', '-1']]
        all_relation = ['similar','unrelated','opposite',
                        'unrelated', 'similar', 'unrelated',
                        'opposite','unrelated','similar']
        for i in range(len(all_case)):
            if all_case[i] == inputs:
               res = all_relation[i]
        return res
        
    def setdata(self, filename):
        fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        
        all_sentence = []
        all_aspect = []
        all_aspect_index = []
        all_polarity = []
        for i in range(0, len(lines), 3):
            lines[i] = self.replace_symbol(lines[i])
            lines[i+1] = self.replace_symbol(lines[i+1])
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            sentence = (text_left + " " + aspect + " " + text_right).strip()
            all_sentence.append(sentence)
            all_aspect.append(self.aspect_mask([len(text_left.split()), len(text_left.split())+len(aspect.split())], self.config.max_sentence_len))
            all_aspect_index.append([len(text_left.split()), len(text_left.split())+len(aspect.split())])
            all_polarity.append(polarity)

        data = {'S':[],'A':[], 'L':[], 'G':[], 'P':[], 'N':[], 'R':[]}
        # S: sentence A: aspect  L: label G:dependency graph P: position encoding 
        # N: aspect number in one sentence  R: relations
        for i in range(len(all_sentence)):
            data['S'].append(self.text_padding(all_sentence[i], self.max_sentence_len))
            #start and end
            cur_asp = [all_aspect[i]]
            cur_relation = [0] # 0:similar
            cur_pos = [self.pos_enc(self.text_padding(all_sentence[i], self.max_sentence_len), all_aspect[i], self.config.dis)]
            for j in range(len(all_sentence)):
                if all_sentence[i] == all_sentence[j] and j != i and self.relation_set.index(self.get_relation([all_polarity[i],all_polarity[j]])) != 1:
                   cur_asp.append(all_aspect[j])
                   cur_relation.append(self.relation_set.index(self.get_relation([all_polarity[i],all_polarity[j]])))
                   cur_pos.append(self.pos_enc(self.text_padding(all_sentence[i], self.max_sentence_len), all_aspect_index[j], self.config.dis))
            data['A'].append(cur_asp)
            data['L'].append(self.label_set.index(all_polarity[i]))
            data['G'].append(self.dependency_adj_matrix(all_sentence[i], self.max_sentence_len))
            data['P'].append(cur_pos)
            data['N'].append(len(cur_asp))
            data['R'].append(cur_relation)

            if i%200 == 0:
               print ('cur line: ', i) 
        return data




    def nextBatch(self, traindata, testdata, is_training=True):
        nextSentenceBatch = []
        nextAspectBatch = []
        nextLabelBatch = []
        nextGraphBatch=[]
        nextPostionBatch = []
        nextNumBatch = []
        nextRelBatch = []
        if is_training:
            if (self.iter_num+1)*self.config.batch_size > len(traindata['S']):
                self.iter_num = 0
            if self.iter_num == 0:
               self.temp_order = list(range(len(traindata['S'])))
               np.random.shuffle(self.temp_order)
            temp_order = self.temp_order[self.iter_num*self.config.batch_size:(self.iter_num+1)*self.config.batch_size]
        else:
            if (self.iter_num+1)*self.config.batch_size > len(testdata['S']):
                self.iter_num = 0
            if self.iter_num == 0:
                self.temp_order = list(range(len(testdata['S'])))
#                np.random.shuffle(self.temp_order)
            temp_order = self.temp_order[self.iter_num*self.config.batch_size:(self.iter_num+1)*self.config.batch_size]
        
        sentence = []
        aspect = []
        label = []
        graph = []
        position = []
        number = []
        relation = []

        for it in temp_order:
            if is_training:
                sentence.append(traindata['S'][it])
                for item in traindata['A'][it]:
                    aspect.append(item)
                label.append(traindata['L'][it])
                graph.append(traindata['G'][it])
                for item in traindata['P'][it]:
                    position.append(item)
                number.append(traindata['N'][it])
                for item in traindata['R'][it]:
                    relation.append(item)
            else:
                sentence.append(testdata['S'][it])
                for item in testdata['A'][it]:
                    aspect.append(item)
                label.append(testdata['L'][it])
                graph.append(testdata['G'][it])
                for item in testdata['P'][it]:
                    position.append(item)
                number.append(testdata['N'][it])
                for item in testdata['R'][it]:
                    relation.append(item)
        self.iter_num += 1

        nextSentenceBatch = np.array(sentence)
        nextAspectBatch = np.array(aspect)
        nextLabelBatch = np.array(label)
        nextGraphBatch = np.array(graph)
        nextPositionBatch = np.array(position)
        nextNumBatch = np.array(number)
        nextRelBatch = np.array(relation)
        return nextSentenceBatch, nextAspectBatch, nextLabelBatch, nextGraphBatch, nextPositionBatch, nextNumBatch, nextRelBatch

if __name__ == '__main__':
    data = DataSet()
    print ('--------------')
    traindata = data.setdata(data.trainfile)
    testdata = data.setdata(data.testfile)

    pickle.dump(traindata, open('./data/'+data.domain+'/train.pkl', 'wb'))
    pickle.dump(testdata, open('./data/'+data.domain+'/test.pkl', 'wb'))

    print (len(testdata['S']))
    print (testdata['N'][:10])
    print (testdata['R'][:10])
