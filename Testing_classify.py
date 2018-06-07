import tensorflow as tf
import numpy as np
import os
import generate_w2v_test
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
batch_size = 128
n_inputs = 200  # MNIST data input (img shape: 28*28)
n_steps = 20  # time steps
n_hidden_units = 256  # neurons in hidden layer
n_classes = 5

tf.set_random_seed(1)

weights = {
    # shape (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units]),name="W_in"),
    # shape (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]),name="W_out")
}
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ]), name="B_in"),
    # shape (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]), name="B_out")
}

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])
def RNN(X, weights, biases):
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (128 batches * 28 steps, 28 inputs)

    X = tf.reshape(X, [-1, n_inputs])

    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    # 使用 basic LSTM Cell.
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)  # 初始化全零 state
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    # 把 outputs 变成 列表 [(batch, outputs)..] * steps
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # 选取最后一个 output
    return results

data = generate_w2v_test.gene_w2v()[5]
#print(data)
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('./model/')
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
with tf.Session() as sess:
    saver.restore(sess,ckpt.model_checkpoint_path)
    '''
    sess.run(weights)
    sess.run(biases)
    pred1 = RNN(x, weights, biases)
    res = sess.run(pred1, feed_dict={
        x: data
    })
    print(pred1)
'''
    W_weights = sess.run(weights)
    B_biases = sess.run(biases)
    print(W_weights["in"].shape)

    w_weights = {
        # shape (28, 128)
        'in': tf.constant(W_weights["in"]),
        # shape (128, 10)
        'out': tf.constant(W_weights["out"])
    }

    b_biases = {
        # shape (28, 128)
        'in': tf.constant(B_biases["in"]),
        # shape (128, 10)
        'out': tf.constant(B_biases["out"])
    }
    print(w_weights)

    pred = RNN(x, w_weights, b_biases)
    res = sess.run(pred, feed_dict={
        x: data
    })
    print(pred)
'''
    h1 = np.dot(data, W_weights["in"])+B_biases["in"]
    #print(h1)
    out = np.dot(h1[1], W_weights["out"])+B_biases["out"]
    print(out)
'''
