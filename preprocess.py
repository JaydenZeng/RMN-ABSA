# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import re
from Settings import Config

class Tokenizer:
    def __init__(self):
        self.config = Config()
        self.domain = self.config.domain

    def replace_symbol(self, text):
        text = text.replace('(', '-LRB- ')
        text = text.replace(')', ' -RRB-')
        text = text.replace("`", "'")
        return text
   
    def readtext(self, fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                lines[i] = self.replace_symbol(lines[i])
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                lines[i+1] = self.replace_symbol(lines[i+1])
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        return text

    def get_word_idx(self, text):
        word2idx = {}
        idx2word = {}
        idx = 0
        word2idx['<pad>'] = idx
        idx2word[idx] = '<pad>'
        idx += 1
        word2idx['<unk>'] = idx
        idx2word[idx] = '<unk>'
        idx += 1
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in word2idx:
                word2idx[word] = idx
                idx2word[idx] = word
                idx += 1
        return word2idx, idx2word


    
    def load_word_vec(self, path, word2idx):
        fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        word_vec = {}
        for line in fin:
            tokens = line.rstrip().split()
            if word2idx is None or tokens[0] in word2idx.keys():
               try:
                   word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
               except:
                   continue
        return word_vec


    def build_embedding_matrix(self, word2idx, embed_dim, pretrain_file):
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        word_vec = self.load_word_vec(pretrain_file, word2idx=word2idx)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
               embedding_matrix[i] = vec
        return embedding_matrix


if __name__ == '__main__':
   token = Tokenizer()
   text = token.readtext(['./data/'+token.domain+'/train.raw', './data/'+token.domain+'/test.raw'])
      
   word2idx, idx2word = token.get_word_idx(text)
   print (len(word2idx))
   pickle.dump(word2idx, open('./data/'+token.domain+'/word2idx.pkl', 'wb'))

   embedding_matrix = token.build_embedding_matrix(word2idx, 300, './data/glove.840B.300d.txt')
   print (np.shape(embedding_matrix))
   pickle.dump(embedding_matrix, open('./data/'+token.domain+'/glove_300.pkl', 'wb'))
#   print (embedding_matrix[1])
