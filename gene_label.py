import numpy as np

def gene_label(row=0):
    with open('input/train_label', 'r', encoding='UTF-8') as f:
        file = f.readlines()
        label_ls = []
        #while times<175*128:
        for times in range(128):
            if "报考知识库" in file[row*128+times]:
                label_ls.append([1,0,0,0,0,0])

            elif "新生入学" in file[row*128+times]:
                label_ls.append([0,1,0,0,0,0])

            elif "APP操作" in file[row*128+times]:
                label_ls.append([0,0,1,0,0,0])

            elif "售后" in file[row*128+times]:
                label_ls.append([0,0,0,1,0,0])

            elif "课程库" in file[row*128+times]:
                label_ls.append([0,0,0,0,1,0])
            else:
                label_ls.append([0, 0, 0, 0, 0,1])
            times += 1

        return np.reshape(label_ls, (128, 6))
