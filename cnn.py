# coding=utf-8
import tensorflow as tf
from nn import NN


# 进行步长为1, 边距为0的卷积
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 进行2x2的池化
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


class CNN(NN):
    def __init__(self, leaning_rate=1e-4, max_iterators=200, batch_size=200):
        NN.__init__(self, leaning_rate, max_iterators, batch_size)

    def inference(self):
        # 为了接下来的矩阵运算, 需要对x进行reshape操作
        x = tf.reshape(self.x, [-1, 28, 28, 1])

        # 第一层卷积, 5x5的patch, 输入通道为1, 输出通道为32, 抽取出32个特征值
        W_conv1 = self.weight_variables([5, 5, 1, 32])
        b_conv1 = self.biases_variables([32])
        # 运算结果进行ReLU激活
        x = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        # 2x2的池化操作
        x = max_pool(x)

        # 第二层卷积, 5x5的patch, 输入通道为32, 输出通道为64, 抽取出64个特征值
        W_conv2 = self.weight_variables([5, 5, 32, 64])
        b_conv2 = self.biases_variables([64])
        # 运算结果进行ReLU激活
        x = tf.nn.relu(conv2d(x, W_conv2) + b_conv2)
        # 2x2的池化操作
        x = max_pool(x)

        # 第一层密集连接层
        # 此时的流信息shape是[-1, 7, 7, 64]的, 需要reshape为[-1, 7x7x64], 以进行后续全连接传播
        x = tf.reshape(x, [-1, 7 * 7 * 64])
        W_fc1 = self.weight_variables([7 * 7 * 64, 1024])
        b_fc1 = self.biases_variables([1024])
        x = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

        # 第二层密集连接层
        W_fc2 = self.weight_variables([1024, 10])
        b_fc2 = self.biases_variables([10])
        x = tf.matmul(x, W_fc2) + b_fc2

        return x
