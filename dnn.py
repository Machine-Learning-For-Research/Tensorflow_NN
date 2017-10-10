import tensorflow as tf
from nn import NN


class DNN(NN):
    def __init__(self, leaning_rate=1e-2, max_iterators=200, batch_size=200):
        NN.__init__(self, leaning_rate, max_iterators, batch_size)

    def inference(self):
        x = self.x

        W1 = self.weight_variables([784, 50])
        b1 = self.biases_variables([50])
        x = tf.matmul(x, W1) + b1
        x = tf.nn.relu(x)

        W2 = self.weight_variables([50, 10])
        b2 = self.biases_variables([10])
        x = tf.matmul(x, W2) + b2

        return x
