import tensorflow as tf
from nn import NN


class RNN(NN):
    n_inputs = 28  # MNIST data input (img shape: 28*28)
    n_steps = 28  # time steps
    n_hidden = 50  # neurons in hidden layer

    def __init__(self, leaning_rate=1e-2, max_iterators=200, batch_size=200):
        NN.__init__(self, leaning_rate, max_iterators, batch_size)

    def inference(self):
        x = self.x

        x = tf.reshape(x, [-1, self.n_inputs])

        W1 = self.weight_variables([self.n_inputs, self.n_hidden])
        b1 = self.biases_variables([self.n_hidden])
        x = tf.matmul(x, W1) + b1

        x = tf.reshape(x, [-1, self.n_steps, self.n_hidden])

        cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state, time_major=False)

        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

        W2 = self.weight_variables([self.n_hidden, 10])
        b2 = self.biases_variables([10])
        x = tf.matmul(outputs[-1], W2) + b2

        return x
