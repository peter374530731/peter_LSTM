import tensorflow as tf
import os
import gene_w2v
import gene_label
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.set_random_seed(1)   # set random seed

# hyperparameters
lr = 0.001                  # learning rate
Limit_step = 1000     # train step 上限
batch_size = 128
n_inputs = 200               # data input
n_steps = 20                # time steps
n_hidden_units = 256        # neurons in hidden layer
n_classes = 6              #  classes (0-9 digits)
num_batch = 195            #batch number for each epoch
# x y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 对 weights biases 初始值的定义
weights = {
    # shape (20, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units]),name="W_in"),
    # shape (128,5)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]),name="W_out")
}
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ]), name="B_in"),
    # shape (5, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]), name="B_out")
}

def RNN(X, weights, biases):
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (10000 batches * 30 steps, 20 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (10000 batches * 30 steps, 20 inputs) 换回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    # 使用 basic LSTM Cell.
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)  # 初始化全零 state
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    # 把 outputs 变成 列表 [(batch, outputs)..] * steps
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # 选取最后一个 output
    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)


correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#init = tf.global_variables_initializer()
saver = tf.train.Saver()
times = 0
epochs = 0
with tf.Session() as sess:
    #sess.run(init)
    model_file = tf.train.latest_checkpoint('./model/')
    saver.restore(sess, model_file)
    test_i = 0
    test_res = ""
    while test_i < num_batch:

        res = sess.run(pred, feed_dict={
            x: gene_w2v.gene_w2v(test_i)
        })

        for i in range(128):
            # print(res[i])
            if np.argmax(res[i]) == 0:
                test_res = test_res + "报考知识库" + "\n"
            elif np.argmax(res[i]) == 1:
                test_res = test_res + "新生入学" + "\n"
            elif np.argmax(res[i]) == 2:
                test_res = test_res + "APP操作" + "\n"
            elif np.argmax(res[i]) == 3:
                test_res = test_res + "售后" + "\n"
            elif np.argmax(res[i]) == 4:
                test_res = test_res + "课程库" + "\n"
            else:
                test_res = test_res + "other" + "\n"
        test_i += 1
        if test_i % 10 == 0:
            print(test_i)
    with open('output/pred_res_label', 'w', encoding='UTF-8') as f9:
        f9.write(test_res)
        print("Pred_finished")
