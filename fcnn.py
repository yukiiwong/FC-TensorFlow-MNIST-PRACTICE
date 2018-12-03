import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
from PIL import Image
import numpy as np
import cv2

#导入mnist数据,'MNIST_data':判断'MNIST_data'文件夹中是否有数据，没有则下载；one_hot:特征提取的一种编码形式
mnist = input_data.read_data_sets('./MNIST_data', one_hot = True)

#占位符
x = tf.placeholder(dtype=tf.float32, shape=(None, 28*28*1), name='x')#input_layer
y = tf.placeholder(dtype=tf.float32, shape=(None,10), name='y')#label

batch_size = 200 #分批次训练，每次的训练数据量

#建立神经网络中的层结构
def add_layer(input_data, input_num, output_num, activation_function = None):
    #output = input_data * weight + bias
    #tf.random_normal:用于从服从指定正太分布的数值中取出指定个数的值
    w = tf.Variable(initial_value = tf.random_normal(shape = [input_num, output_num]))#初始化系数w
    b = tf.Variable(initial_value = tf.random_normal(shape = [1, output_num]))#初始化bias
    output = tf.add(tf.matmul(input_data, w), b) #等同于 y = wx + b
    if activation_function:
        output = activation_function(output)
    return output

#建立神经网络全连接,两个隐藏层，一个输出层
def build_nn(data):
    hidden_layer1 = add_layer(data, 784, 100, activation_function=tf.nn.sigmoid)
    hidden_layer2 = add_layer(hidden_layer1, 100, 50, activation_function=tf.nn.sigmoid)
    output_layer = add_layer(hidden_layer2, 50, 10)
    return output_layer

#训练神经网络
def train_nn(data):
    #out of NN
    output = build_nn(data)
    #loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(loss)
    #保存训练好的模型
    saver = tf.train.Saver()

    liter_num = 1500#训练次数

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if not os.path.exists('checkpoint'):
            for i in range(liter_num):
                #减轻内存负担，分批次进行训练
                epoch_cost = 0
                for _ in range(int(mnist.train.num_examples / batch_size)):
                    x_data, y_data = mnist.train.next_batch(batch_size)
                    cost, _ = sess.run([loss,optimizer], feed_dict={x: x_data, y: y_data})
                    epoch_cost += cost
                print('Epoch', i, ':', epoch_cost)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(output, 1)),tf.float32))
            acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
            print(acc)
            saver.save(sess, './mnist.ckpt')
        else:
            saver.restore(sess, './mnist.ckpt')
            predict('1.jpg',sess,output)

'''
#创建十个文件夹查看数据集的图片格式
def reconstruct_image():
    for i in range(10):
        if not os.path.exists('./SAMPLE'.format(i)):
            os.mkdir('./SAMPLE'.format(i))
        if not os.path.exists('./SAMPLE/{}'.format(i)):
            os.mkdir('./SAMPLE/{}'.format(i))
    #根据label分发为28*28规格的图片
    batch_size = 1
    for i in range(int(mnist.train.num_examples / batch_size)):
        #x_data = [[784]], y_data = [[10]]
        x_data, y_data = mnist.train.next_batch(batch_size)
        img = Image.fromarray(np.reshape(np.array(x_data[0] * 255, dtype = 'uint8'), newshape = (28,28)))#将图片的格式转化成Image能用的格式
        dir = np.argmax(y_data[0])
        img.save('./SAMPLE/{}/{}.bmp'.format(dir, i))
'''
#输入手写图片
def read_data(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #图片需要保证正方形，规格为28*28；归一化；灰度图
    process_image = cv2.resize(image, dsize=(28,28))
    process_image= cv2.dilate(process_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    process_image = cv2.erode(process_image, kernel, iterations=2)
    cv2.imshow("process_image", process_image)
    process_image = np.resize(process_image, new_shape=(1,784))
    process_image = process_image / 255.0
    return image,process_image

#输出预测
def predict(image_path, sess, output):
    image, processed_image = read_data(image_path)
    result = sess.run(output, feed_dict = {x: processed_image})
    result = np.argmax(result, 1)
    print("The prediction is", result)
    cv2.putText(image,"The prediction is {}".format(result),(20,20),cv2.FONT_HERSHEY_COMPLEX,1, (0,255,255))
    cv2.imshow("image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
train_nn(x)
#reconstruct_image()