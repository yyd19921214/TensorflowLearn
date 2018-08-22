import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 利用Logistic Regression来识别手写数字
# 训练速度较快，思路简单
# 获取MNIST数据
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Variables
batch_size = 100
total_steps = 5000
steps_per_test = 100

# 建立模型
x = tf.placeholder(tf.float32, [None, 784])
y_label = tf.placeholder(tf.float32, [None, 10])
# 初始化参数的权重值
w = tf.Variable(tf.zeros([784, 10]))
# 初始化截距
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, w) + b)

# 计算误差
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(y), reduction_indices=[1]))
# 训练模型使得误差尽可能的小
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 计算准确率
correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_label, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 开始训练模型
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 迭代计算5000次
    for step in range(total_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={x: batch_x, y_label: batch_y})
        # 每迭代计算100次，验证在测试集上的结果
        if step % steps_per_test == 0:
            print(step, sess.run(accuracy, feed_dict={x: mnist.test.images, y_label: mnist.test.labels}))
