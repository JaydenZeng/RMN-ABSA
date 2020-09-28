import numpy as np
seg_num = 600
r = 0.30
with open('./all/train.raw', 'r+') as ftrain, open('./all/test.raw', 'r+') as ftest, \
    open('./res-30/train.raw', 'w+') as fw, open('./res-30/test.raw', 'w+') as fwt:
    train = ftrain.readlines()
    test = ftest.readlines()
    len_train = len(train)
    len_test  = len(test)
    print (len_train, len_test)
    index_train = range(0, len_train, seg_num)
    for ind in index_train:
        for i in range(ind, ind+int(seg_num*r)):
            fw.write(train[i])

    index_test = range(0, len_test, seg_num)
    for ind in index_test:
        for i in range(ind, ind+int(seg_num*r)):
            fwt.write(test[i])
