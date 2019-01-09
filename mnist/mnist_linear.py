# 引入一个input_data.py 里面包含下载数据和处理数据
import input_data
import tensorflow as tf

# 会在自动生成的MNIST_data中读取数据
# mnist.train.images[60000,784] 是训练图片， mnist.train.labels[60000，10]是训练标签
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 定义占位符，方便后续数据规模的改变，而不是定死
y_ = tf.placeholder(tf.float32, [None, 10], name="y_")
with tf.name_scope('hypothesis'):
    # 定义占位符，方便后续数据规模的改变，而不是定死
    x = tf.placeholder(tf.float32, [None, 784], name="x")
    # Variable 用于计算输入值，也可以用来在计算中被修改
    W = tf.Variable(tf.zeros([784, 10]), name='W')
    b = tf.Variable(tf.zeros([10]), name='b')
    res = tf.add(tf.matmul(x,W, name='multiply'), b, name='add')
    # softmax回归,给不同的对象分配概率。res 中是一个[None,10]的矩阵。softmax可以看成是一个激励函数,对每一个数字转换成一个概率值
    y = tf.nn.softmax(res, name='y')




with tf.name_scope('cost_function'):
    loss_value = -tf.reduce_sum(y_*tf.log(y), name='loss')

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss_value)

with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        wcoeff, bias = sess.run([W, b])

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
