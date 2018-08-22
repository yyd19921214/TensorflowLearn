# 利用Logistic Regression进行手写数字识别

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

## 实现回归模型

首先导入 TensorFlow，命令如下：
```python
import tensorflow as tf
```
接下来我们指定一个输入，在这里输入即为样本数据，如果是训练集那么则是 55000 x 784 的矩阵，如果是验证集则为 5000 x 784 的矩阵，如果是测试集则是 10000 x 784 的矩阵，所以它的行数是不确定的，但是列数是确定的。

所以可以先声明一个 placeholder 对象：

```python
x = tf.placeholder(tf.float32, [None, 784])
```
这里第一个参数指定了矩阵中每个数据的类型，第二个参数指定了数据的维度。

接下来我们需要构建第一层网络，表达式如下：

![softmax-2](https://germey.gitbooks.io/ai/assets/2017-10-25-15-27-25.jpg)

这里实际上是对输入的 x 乘以 w 权重，然后加上一个偏置项作为输出，而这两个变量实际是在训练的过程中动态调优的，所以我们需要指定它们的类型为 Variable，代码如下：
```python
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
```
接下来需要实现的就是上图所述的公式了，我们再进一步调用 Softmax 进行计算，得到 y：
```yy = tf.nn.softmax(tf.matmul(x, w) + b)```

通过上面几行代码我们就已经把模型构建完毕了，结构非常简单。

------

## 损失函数
为了训练我们的模型，我们首先需要定义一个指标来评估这个模型是好的。其实，在机器学习，我们通常定义指标来表示一个模型是坏的，这个指标称为成本（cost）或损失（loss），然后尽量最小化这个指标。但是这两种方式是相同的。

一个非常常见的，非常漂亮的成本函数是“交叉熵”（cross-entropy）。交叉熵产生于信息论里面的信息压缩编码技术，但是它后来演变成为从博弈论到机器学习等其他领域里的重要技术手段。它的定义如下：
![loss-1](https://germey.gitbooks.io/ai/assets/2017-10-25-15-45-09.jpg)
y 是我们预测的概率分布, y_label 是实际的分布，比较粗糙的理解是，交叉熵是用来衡量我们的预测用于描述真相的低效性。
我们可以首先定义 y_label，它的表达式是：
```y_label = tf.placeholder(tf.float32, [None, 10])```
接下来我们需要计算它们的交叉熵，代码如下：
```cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(y), reduction_indices=[1]))```

首先用 reduce_sum() 方法针对每一个维度进行求和，reduction_indices 是指定沿哪些维度进行求和。
然后调用 reduce_mean() 则求平均值，将一个向量中的所有元素求算平均值。
这样我们最后只需要优化这个交叉熵就好了。
所以这样我们再定义一个优化方法：

```train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)```

这里使用了 GradientDescentOptimizer，在这里，我们要求 TensorFlow 用梯度下降算法（gradient descent algorithm）以 0.5 的学习速率最小化交叉熵。梯度下降算法（gradient descent algorithm）是一个简单的学习过程，TensorFlow 只需将每个变量一点点地往使成本不断降低的方向移动即可。

------

## 运行模型
定义好了以上内容之后，相当于我们已经构建好了一个计算图，即设置好了模型，我们把它放到 Session 里面运行即可：
```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(total_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={x: batch_x, y_label: batch_y})
```
该循环的每个步骤中，我们都会随机抓取训练数据中的 batch_size个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行 train。
这里需要一些变量的定义：

```
batch_size = 100
total_steps = 5000
```

------

## 测试模型
首先让我们找出那些预测正确的标签。tf.argmax() 是一个非常有用的函数，它能给出某个 Tensor 对象在某一维上的其数据最大值所在的索引值。由于标签向量是由 0,1 组成，因此最大值 1 所在的索引位置就是类别标签，比如 tf.argmax(y, 1) 返回的是模型对于任一输入 x 预测到的标签值，而 tf.argmax(y_label, 1) 代表正确的标签，我们可以用 tf.equal() 方法来检测我们的预测是否真实标签匹配（索引位置一样表示匹配）。

```correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_label, axis=1))```

这行代码会给我们一组布尔值。为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。例如，[True, False, True, True] 会变成 [1, 0, 1, 1] ，取平均值后得到 0.75。
```accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))```
最后，我们计算所学习到的模型在测试数据集上面的正确率，定义如下：

```python
steps_per_test = 100
if step % steps_per_test == 0:
    print(step, sess.run(accuracy, feed_dict={x: mnist.test.images, y_label: mnist.test.labels}))
```
这个最终结果值应该大约是92%。

这样我们就通过完成了训练和测试阶段，实现了一个基本的训练模型，后面我们会继续优化模型来达到更好的效果。
运行结果如下：
```
0 0.453
100 0.8915
200 0.9026
300 0.9081
400 0.9109
500 0.9108
600 0.9175
700 0.9137
800 0.9158
900 0.9176
1000 0.9167
1100 0.9186
1200 0.9206
1300 0.9161
1400 0.9218
1500 0.9179
1600 0.916
1700 0.9196
1800 0.9222
1900 0.921
2000 0.9223
2100 0.9214
2200 0.9191
2300 0.9228
2400 0.9228
2500 0.9218
2600 0.9197
2700 0.9225
2800 0.9238
2900 0.9219
3000 0.9224
3100 0.9184
3200 0.9253
3300 0.9216
3400 0.9218
3500 0.9212
3600 0.9225
3700 0.9224
3800 0.9225
3900 0.9226
4000 0.9201
4100 0.9138
4200 0.9184
4300 0.9222
4400 0.92
4500 0.924
4600 0.9234
4700 0.9219
4800 0.923
4900 0.9254
5000 0.9218
```

------

## 本文代码地址

### [利用Logistic Regression进行手写数字识别](https://github.com/yyd19921214/TensorflowLearn/blob/master/LR.py)


