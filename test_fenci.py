import gensim,re,jieba

import numpy as np
def test_fenci():
    comt = ""
    #model = gensim.models.Word2Vec.load("62model")
    with open('test_fen', 'r', encoding='UTF-8') as f:
        file = f.readlines()
        print(file)
        for line in file:
            #l_p = pls.split(line)
            #if "报考知识库" in l_p[1]:
            seg_list = jieba.cut(line, cut_all=False)
            comt = comt + " ".join(seg_list)
    print(len(comt))
    with open('test_fen_res', 'w', encoding='UTF-8') as f1:
        f1.write(comt)


test_fenci()