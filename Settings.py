class Config(object):
    def __init__(self):
        self.max_sentence_len = 80
        self.keep_prob = 0.7
        self.hidden_size = 300
        self.batch_size = 32
        self.class_num = 3
        self.learning_rate = 0.001
        self.epoch_num = 5
        self.dis = 15
        self.r = 0.4
        self.domain = 'res14'  #twitter, lap14, res14, lap15, lap16, MAMS
