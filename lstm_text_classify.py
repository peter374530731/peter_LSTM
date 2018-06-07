import tensorflow as tf
import os
import gene_label
import gene_w2v
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.set_random_seed(1)   # set random seed

# hyperparameters
#==============================
lr = 0.001                  # learning rate
Limit_step = 1000     # train step 上限
batch_size = 128
n_inputs = 200               # data input
n_steps = 20                # time steps
n_hidden_units = 256        # neurons in hidden layer
n_classes = 6              #  classes (0-9 digits)
num_batch = 170             #batch number for each epoch
Train_acc = 0.98           #train acc target
#================================

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

init = tf.global_variables_initializer()
saver = tf.train.Saver()
times = 0
epochs = 0
with tf.Session() as sess:
    sess.run(init)
    step = 0
    acc = 0
    while step < Limit_step and acc < Train_acc:
        m_batch_xs = gene_w2v.gene_w2v(times)
        m_batch_ys = gene_label.gene_label(times)
        batch_xs = m_batch_xs
        batch_ys = m_batch_ys
        tf.reshape(batch_xs, [batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 10 == 0:
            acc=sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
            print("=====================================")
            print("step:", step)
            print("训练数据准确率ACC:", acc)

            print("训练残余误差lost:", sess.run(cost,feed_dict={
                x: batch_xs,
                y: batch_ys,
            }))
            for pred_time in range(5):
                x_pred_data = gene_w2v.gene_w2v(pred_time+170)
                y_pred_data = gene_label.gene_label(pred_time+170)
                pred_result = sess.run(accuracy, feed_dict={
                    x:x_pred_data,
                    y:y_pred_data
                })
                print("---------",pred_time,"组---------")
                print("测试数据准确率ACC（算法）:", pred_result)
                pred_data = sess.run(pred, feed_dict={
                    x:x_pred_data
                })
                pred_num = 0
                for ii in range(128):
                    if np.argmax(pred_data[ii])==np.argmax(y_pred_data[ii]):
                        pred_num += 1
                print("每组测试128条。正确数量：",pred_num,"正确率：",int(100*pred_num/128),"%" ,"\n")
        step += 1
        times += 1
        if times > num_batch-1:
            times = 0
            epochs += 1
            print("epoch:", epochs)
    save_path = saver.save(sess, "model/model.ckpt")
    print(save_path)
