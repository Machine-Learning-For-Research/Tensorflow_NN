# Tensorflow_NN
Tensorflow实现DNN、CNN、RNN三种神经网络

#### DNN

```python
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
```

### CNN

```python
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
```

#### RNN

```python
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
```

