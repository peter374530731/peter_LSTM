import jieba
comment = ""
with open('Train_data0606_1', 'r', encoding='GBK') as f:
    file = f.readlines()
    #print(file)
    for line in file:
        seg_list = jieba.cut(line, cut_all=False)
        comment =comment +" ".join(seg_list)
        #print(comment)
with open('Train_data_fc0606_1', 'w', encoding='UTF-8') as f1:
    f1.write(comment)
