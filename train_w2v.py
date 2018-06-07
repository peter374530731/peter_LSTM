import jieba

from gensim.models import word2vec



# inp为输入语料
inp = 'train_data_fc'
# outp1 为输出模型
outp1 = 'wiki.zh.text.model'
# outp2为原始c版本word2vec的vector格式的模型
outp2 = 'wiki.zh.text.vector'
sentences = word2vec.Text8Corpus(inp)  # 加载语料
model = word2vec.Word2Vec(sentences, size=200)  # 默认window=5
print(model["老师"])
model.save("62model")