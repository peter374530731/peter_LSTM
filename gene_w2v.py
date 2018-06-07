import gensim,re
import numpy as np
#import genera_test
def gene_w2v(row = 0):
    pls = re.compile(r'\s')
    #genera_test.gene_test()
    model = gensim.models.Word2Vec.load("word_vecor/talkmodel_04_0524")
    with open('input/train_data_afc', 'r', encoding='UTF-8') as f:
        file = f.readlines()
        #for line in file:
            #print(line)
        s1 = 0
        s2 = 0
        s3 = 0
        s4 = 0
        arr_sen = np.reshape(np.zeros(128 * 20 * 200), (128, 20, 200))
        #for line in file:
        for i0 in range(128):
            l_p = pls.split(file[row*128+i0])
            #print(l_p)
            for l_q in l_p:
                try:
                    arr_sen[s2][s3] = model[l_q]
                except:
                    pass
                s3 += 1
            s3 = 0
            s2 += 1
            if s2 > 127:
                s2 = 0
                break


    return arr_sen
#for ii in range(30):
#print(gene_w2v(0))


