import numpy as np
import tensorflow as tf
import random as rand
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt

'''超参数'''
seq_max_len = 9          # 记忆的步长，反向传播的长度
batch_size = 5            # 多少组数据
n_classes = 2             # 数据标签的种类
state_size = 50           # RNN 隐藏层 cell的个数
learning_rate = 0.01       # 学习率
in_classes = 7            #输入数据的种类数
training_iters = 1000
display_step = 10

'''数据准备'''
X_ =[[1, 2, 3, 5, 0, 0, 0, 0, 0],
    [1, 7, 0, 0, 0, 0, 0, 0, 0],
    [1, 2, 3, 4, 2, 6, 0, 0, 0],
    [1, 2, 3, 4, 2, 6, 0, 0, 0],
    [1, 2, 3, 4, 2, 3, 4, 2, 6],
    [1, 2, 3, 4, 2, 3, 5, 0, 0]]

X =[[1, 2, 3, 4, 5, 6, 7, 0, 0],
    [1, 7, 0, 0, 0, 0, 0, 0, 0],
    [1, 2, 3, 4, 0, 0, 0, 0, 0],
    [1, 2, 3, 4, 2, 6, 0, 0, 0],
    [1, 2, 3, 4, 5, 0, 0, 0, 0],
    [1, 2, 3, 4, 2, 3, 5, 1, 2]]

Y = [0, 1, 0, 1, 0, 1]

X_test = [[1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 3, 0, 0, 0, 0, 0, 0, 0],
          [1, 2, 3, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 0, 0],
          [1, 2, 3, 4, 5, 0, 0, 0, 0],
          [1, 3, 4, 5, 6, 7, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 0, 0]]

Y_test = [0, 1, 0, 0, 0, 0, 0]


'''batch'''
def get_seq_length(seq):
    seqlen = 0
    for item in seq:
        if item != 0:
            seqlen += 1
            continue
        break

    return seqlen

def batch(input_d, labels, batch_size = 10):
    batch_input = []
    batch_label = []
    batch_seqlist = []
    length = len(input_d)
    lines = range(length)
    picked = rand.sample(lines, batch_size)
    for line in picked:
        batch_input.append(input_d[line])
        batch_label.append(labels[line])
        batch_seqlist.append(get_seq_length(input_d[line]))

    return batch_input, batch_label, batch_seqlist


'''place holder'''
input = tf.placeholder(tf.int32, [None, seq_max_len])
output = tf.placeholder(tf.int32, [None, n_classes])
seqlen = tf.placeholder(tf.int32, [None])

W2 = tf.Variable(tf.truncated_normal([state_size, n_classes], stddev=0.1), name="W2")  # 权重初始化为截断的正态分布
b2 = tf.Variable(tf.zeros([n_classes]), name="b2")

'''RNN输入'''
with tf.variable_scope('input'):
    x_one_hot = tf.one_hot(input, in_classes)
    rnn_inputs = tf.unstack(x_one_hot, axis=1)



'''定义RNN cell
with tf.variable_scope('rnn_cell'):
    W1 = tf.get_variable(tf.truncated_normal([n_classes + in_classes, state_size], stddev=0.1), name="W1")  # 权重初始化为截断的正态分布
    b1 = tf.get_variable(tf.zeros([state_size]), name="b1")
'''

def dynamicRNN(x, seqlen, weights, biases):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(state_size)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)
    print(outputs)
    outputs = tf.stack(outputs)
    print(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])
    print(outputs)
    batch_size = tf.shape(outputs)[0]
    print(batch_size)
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)#?
    outputs = tf.gather(tf.reshape(outputs, [-1, state_size]), index)
    print(outputs)
    #return tf.matmul(outputs, weights) + biases
    return tf.nn.softmax(tf.matmul(outputs, weights) + biases)  # 输出层的计算方法，激活函数为Sigmoid


pred = dynamicRNN(rnn_inputs, seqlen, W2, b2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = output))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(output,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y, batch_seqlen = batch(X, Y, batch_size)
        batch_y_one_hot = tf.one_hot(batch_y, n_classes, 0, 1)
        batch_y = tf.unstack(batch_y_one_hot, axis=0)
        a = tf.convert_to_tensor(batch_x)
        batch_y = sess.run(batch_y)
        c = tf.convert_to_tensor(batch_seqlen)
        sess.run(optimizer, feed_dict={input: batch_x, output: batch_y, seqlen: batch_seqlen})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={input: batch_x, output: batch_y, seqlen: batch_seqlen})
            loss = sess.run(cost, feed_dict={input: batch_x, output: batch_y, seqlen: batch_seqlen})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1

    print(sess.run(pred, feed_dict={input: X_test, seqlen: [1, 2, 3, 4, 5, 6, 7]}))