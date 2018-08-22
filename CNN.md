# 利用CNN进行手写数字识别

------
我们使用 MNIST 数据集训练一个可以识别数据的深度学习模型来帮助识别手写数字

------
## MNIST数据集
MNIST 是一个入门级计算机视觉数据集，包含了很多手写数字图片，如图所示：
![mnist-pic](https://germey.gitbooks.io/ai/assets/2017-10-25-14-18-39.png)

数据集中包含了图片和对应的标注，在 TensorFlow中提供了这个数据集，我们可以用如下方法进行导入：
```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
print(mnist)
```
输出结果如下
```python
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
Datasets(train=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x101707ef0>, validation=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x1016ae4a8>, test=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x1016f9358>)
```
在这里程序会首先下载 MNIST 数据集，然后解压并保存到刚刚制定好的 MNIST_data 文件夹中，然后输出数据集对象。
数据集中包含了 55000 行的训练数据集（mnist.train）、5000 行验证集（mnist.validation）和 10000 行的测试数据集（mnist.test），文件如下所示：
![mnist-datashow](https://germey.gitbooks.io/ai/assets/2017-10-25-14-26-54.jpg)
正如前面提到的一样，每一个 MNIST 数据单元有两部分组成：一张包含手写数字的图片和一个对应的标签。我们把这些图片设为 xs，把这些标签设为 ys。训练数据集和测试数据集都包含 xs 和 ys，比如训练数据集的图片是 mnist.train.images ，训练数据集的标签是 mnist.train.labels，每张图片是 28 x 28 像素，即 784 个像素点，我们可以把它展开形成一个向量，即长度为 784 的向量。

所以训练集我们可以转化为 [55000, 784] 的向量，第一维就是训练集中包含的图片个数，第二维是图片的像素点表示的向量。

------
## Softmax
Softmax 可以看成是一个激励（activation）函数或者链接（link）函数，把我们定义的线性函数的输出转换成我们想要的格式，也就是关于 10 个数字类的概率分布。因此，给定一张图片，它对于每一个数字的吻合度可以被 Softmax 函数转换成为一个概率值。Softmax 函数可以定义为：
![softmax-1](https://germey.gitbooks.io/ai/assets/2017-10-25-15-09-37.jpg)
比如判断一张图片中的动物是什么，可能的结果有三种，猫、狗、鸡，假如我们可以经过计算得出它们分别的得分为 3.2、5.1、-1.7，Softmax 的过程首先会对各个值进行次幂计算，分别为 24.5、164.0、0.18，然后计算各个次幂结果占总次幂结果的比重，这样就可以得到 0.13、0.87、0.00 这三个数值，所以这样我们就可以实现差别的放缩，即好的更好、差的更差。

如果要进一步求损失值可以进一步求对数然后取负值，这样 Softmax 后的值如果值越接近 1，那么得到的值越小，即损失越小，如果越远离 1，那么得到的值越大。

------
## 权重初始化
在卷积神经网络中，我们需要在初始化的时候权重加入少量噪声来打破对称性和避免零梯度，偏置项直接使用一个较小的正数来避免节点输出恒为零的问题。

所以权重我们可以使用截尾正态分布函数 truncated_normal() 来生成初始化张量，我们可以给它指定均值或标准差，均值默认是 0， 标准差默认是 1，例如我们生成一个 [10] 的张量，代码如下
```python
import tensorflow as tf
initial = tf.truncated_normal([10], stddev=0.1)
with tf.Session() as sess:
    print(sess.run(initial))
```
另外 constant() 方法是用于生成常量的方法，例如生成一个 [10] 的常量张量，代码如下
```python
import tensorflow as tf
initial = tf.constant(0.2, shape=[10])
with tf.Session() as sess:
    print(sess.run(initial))
```
这里我们可以将这两个方法封装成一个函数来尝试
```python
def weight(shape, stddev=0.1, mean=0):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initial)
 
def bias(shape, value):
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial)
```
-------
## 卷积
这次我们需要使用卷积神经网络来处理图片，所以这里的核心部分就是卷积和池化了，首先我们来了解一下卷积和池化。卷积常用的方法为 conv2d() ，它的 API 如下：
```tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)```

这个方法是 TensorFlow 实现卷积常用的方法，也是搭建卷积神经网络的核心方法，参数介绍如下：

* input，指需要做卷积的输入图像，它要求是一个 Tensor，具有 [batch_size, in_height, in_width, in_channels] 这样的 shape，具体含义是 [batch_size 的图片数量, 图片高度, 图片宽度, 输入图像通道数]，注意这是一个 4 维的 Tensor，要求类型为 float32 和 float64 其中之一。
* filter，相当于 CNN 中的卷积核，它要求是一个 Tensor，具有 [filter_height, filter_width, in_channels, out_channels] 这样的shape，具体含义是 [卷积核的高度，卷积核的宽度，输入图像通道数，输出通道数（即卷积核个数）]，要求类型与参数 input 相同，有一个地方需要注意，第三维 in_channels，就是参数 input 的第四维。
* strides，卷积时在图像每一维的步长，这是一个一维的向量，长度 4，具有 [stride_batch_size, stride_in_height, stride_in_width, stride_in_channels] 这样的 shape，第一个元素代表在一个样本的特征图上移动，第二三个元素代表在特征图上的高、宽上移动，第四个元素代表在通道上移动。
* padding，string 类型的量，只能是 SAME、VALID 其中之一，这个值决定了不同的卷积方式。
use_cudnn_on_gpu，布尔类型，是否使用 cudnn 加速，默认为true。
返回的结果是 [batch_size, out_height, out_width, out_channels] 维度的结果。
我们这里拿一张 3×3 的图片，单通道（通道为1）的图片，拿一个 1×1 的卷积核进行卷积：
```python
input = tf.Variable(tf.random_normal([1, 3, 3, 1]))
filter = tf.Variable(tf.random_normal([1, 1, 1, 1]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
print(op.shape)
```
结果如下：
```(1, 3, 3, 1)```
将图片扩大为 7×7，卷积核仍然使用 3×3：
```
input = tf.Variable(tf.random_normal([1, 7, 7, 1]))
filter = tf.Variable(tf.random_normal([3, 3, 1, 1]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
print(op.shape)
```
结果如下：
```(1, 5, 5, 1)```
最后我们用一个例子来感受一下：
```
import tensorflow as tf
input = tf.Variable(tf.random_normal([2, 4, 4, 5]))
filter = tf.Variable(tf.random_normal([2, 2, 5, 2]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
print(op.shape)
print(sess.run(op))
```
这里 input、filter 通过指定 shape 的方式调用 random_normal() 方法进行随机初始化，input 的维度为 [2, 4, 4, 5]，即 batch_size 为 2，图片是 4×4，输入通道数为 5，卷积核大小为 2×2，输入通道 5，输出通道 2，步长为 1，padding 方式选用 VALID，最后输出得到输出的 shape 和结果。
```
(2, 3, 3, 2)
[[[[  2.05039382  -8.82934952]
   [ -9.77668381   3.63882256]
   [ -4.46390772  -5.91670704]]

  [[  8.41201782  -6.72245312]
   [ -1.47592044  13.03628349]
   [  5.44015312   2.46059227]]

  [[ -3.18967772   1.24733043]
   [-10.1108532   -6.44734669]
   [  1.99426246   2.91549349]]]


 [[[ -1.66685319   0.32011557]
   [ -5.66163826  -0.37670898]
   [ -0.74658942   1.31723833]]

  [[ -5.85412216  -0.29930949]
   [ -0.75974303  -1.84006214]
   [ -2.05475235   4.9572196 ]]

  [[ -4.09344864   1.39405775]
   [ -1.28887582  -2.82365012]
   [  4.87360907  10.8071022 ]]]]
```
------

## 池化
池化层往往在卷积层后面，通过池化来降低卷积层输出的特征向量，同时改善结果。
在这里介绍一个常用的最大值池化 max_pool() 方法，其 API 如下：
```tf.nn.max_pool(value, ksize, strides, padding, name=None)```
是CNN当中的最大值池化操作，其实用法和卷积很类似。

参数介绍如下：

* value，需要池化的输入，一般池化层接在卷积层后面，所以输入通常是 feature map，依然是 [batch_size, height, width, channels] 这样的shape。
* ksize，池化窗口的大小，取一个四维向量，一般是 [batch_size, height, width, channels]，因为我们不想在 batch 和 channels 上做池化，所以这两个维度设为了1。
* strides，和卷积类似，窗口在每一个维度上滑动的步长，一般也是 [stride_batch_size, stride_height, stride_width, stride_channels]。
* padding，和卷积类似，可以取 VALID、SAME，返回一个 Tensor，类型不变，shape 仍然是 [batch_size, height, width, channels] 这种形式。
在这里输入为 [3, 7, 7, 2]，池化窗口设置为 [1, 2, 2, 1]，步长为 [1, 1, 1, 1]，padding 模式设置为 VALID。
```
input = tf.Variable(tf.random_normal([3, 7, 7, 2]))
op = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
print(op.shape)
```
## 卷积和池化
所以了解了以上卷积和池化方法的用法，我们可以定义如下两个工具方法,这两个方法分别实现了卷积和池化，并设置了默认步长和核大小。
```
def conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.conv2d(input, filter, strides=strides, padding=padding)
def max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
    return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding)
```
------

## 模型的实现
- *初始化参数*

首先我们需要初始化一些数据，包括输入的 x 和对一个的标注 y_label：
```
x = tf.placeholder(tf.float32, shape=[None, 784])
y_label = tf.placeholder(tf.float32, shape=[None, 10])
```
- *第一层卷积*

现在我们可以开始实现第一层了。它由一个卷积接一个 max pooling 完成。卷积在每个 5×5 的 patch 中算出 32 个特征。卷积的权重张量形状是 [5, 5, 1, 32]，前两个维度是 patch的大小，接着是输入的通道数目，最后是输出的通道数目，而对于每一个输出通道都有一个对应的偏置量，我们首先初始化 w 和 b
```python
w_conv1 = weight([5, 5, 1, 32])
b_conv1 = bias([32])
```
为了用这一层，我们把 x 变成一个四维向量，其第 2、3 维对应图片的宽、高，最后一维代表图片的颜色通道数，因为是灰度图所以这里的通道数为 1，如果是彩色图，则为 3。
随后我们需要对图片做 reshape 操作，将其
```x_reshapex  = tf.reshape(x, [-1, 28, 28, 1])```
我们把 x_reshape 和权值向量进行卷积，加上偏置项，然后应用 ReLU 激活函数，最后进行 max pooling
```python
h_conv1 = tf.nn.relu(conv2d(x_reshape, w_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1)
```
- *第二层卷积*

现在我们已经实现了一层卷积，为了构建一个更深的网络，我们再继续增加一层卷积，将通道数变成 64，所以这里的初始化权重和偏置为：
```python
w_conv2 = weight([5, 5, 32, 64])
b_conv2 = bias([64])
```
- *全连接层与Dropout*

现在，图片尺寸减小到7×7，我们再加入一个有 1024 个神经元的全连接层，用于处理整个图片。我们把池化层输出的张量 reshape 成一些向量，乘上权重矩阵，加上偏置，然后对其使用 ReLU。
```python
w_fc1 = weight([7 * 7 * 64, 1024])
b_fc1 = bias([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
```
为了减少过拟合，我们在输出层之前加入 dropout。我们用一个 placeholder 来代表一个神经元的输出在 dropout 中保持不变的概率。这样我们可以在训练过程中启用 dropout，在测试过程中关闭 dropout。 TensorFlow 的 tf.nn.dropout 操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的 scale，所以用 dropout 的时候可以不用考虑 scale。
```
keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=keep_prob)
```

- *输出层*

最后，我们添加一个 Softmax 输出层，这里我们需要将 1024 维转为 10 维，所以需要声明一个 [1024, 10] 的权重和 [10] 的偏置
```python
w_fc2 = weight([1024, 10])
b_fc1 = bias([10])
y = tf.nn.softmax(tf.matmul(h_fc1_dropout, w_fc2) + b_fc1)
```
-------

## 训练和评估模型
为了进行训练和评估，我们使用与之前简单的单层 Softmax 神经网络模型几乎相同的一套代码，只是我们会用更加复杂的 Adam 优化器来做梯度最速下降，在 feed_dict 中加入额外的参数 keep_prob 来控制 dropout 比例，然后每 100次 迭代输出一次日志：
```python
# Loss
cross_entropy = -tf.reduce_sum(y_label * tf.log(y))
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
 
# Prediction
correct_prediction = tf.equal(tf.argmax(y_label, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
# Train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(total_steps + 1):
        batch = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={x: batch[0], y_label: batch[1], keep_prob: dropout_keep_prob})
        # Train accuracy
        if step % steps_per_test == 0:
            print('Training Accuracy', step,
                  sess.run(accuracy, feed_dict={x: batch[0], y_label: batch[1], keep_prob: 1}))
 
# Final Test
    print('Test Accuracy', sess.run(accuracy, feed_dict={x: mnist.test.images, y_label:       mnist.test.labels, keep_prob: 1}))
```

------

## 运行
以上代码，在最终测试集上的准确率大概是99.2%。
运行结果：
```python
Training Accuracy 0 0.05
Training Accuracy 100 0.7
Training Accuracy 200 0.85
Training Accuracy 300 0.9
Training Accuracy 400 0.93
Training Accuracy 500 0.91
Training Accuracy 600 0.94
Training Accuracy 700 0.95
Training Accuracy 800 0.95
Training Accuracy 900 0.95
Training Accuracy 1000 0.97
Training Accuracy 1100 0.95
Training Accuracy 1200 0.96
Training Accuracy 1300 0.99
Training Accuracy 1400 0.98
Training Accuracy 1500 0.95
Training Accuracy 1600 0.97
Training Accuracy 1700 1.0
Training Accuracy 1800 0.95
Training Accuracy 1900 0.95
Training Accuracy 2000 0.95
Training Accuracy 2100 0.96
Training Accuracy 2200 0.96
Training Accuracy 2300 0.98
Training Accuracy 2400 0.97
Training Accuracy 2500 0.96
Training Accuracy 2600 0.99
Training Accuracy 2700 0.96
Training Accuracy 2800 0.98
Training Accuracy 2900 0.95
Training Accuracy 3000 0.99
```

## 代码链接
[利用卷积神经网络进行手写数字识别](https://github.com/yyd19921214/TensorflowLearn/blob/master/CNN.py)