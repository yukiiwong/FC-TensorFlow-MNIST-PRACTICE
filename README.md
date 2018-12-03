# FC-TensorFlow-MNIST-PRACTICE
Use Tensorflow to establish a fully connected neural network to identify my own handwritten numbers.
The result isn't ideal. And it has a very pool recognition of 6.

What's more, there is a commented function(reconstruct_imge()) in the program.
```python
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
```
You can run this function to unzip the souce MNIST data, and convert the numbers to a 28*28 .bmp file.It can
help you to impove the accuracy.


reference: Peter_Chan in Bilibili, TensorFlow系列教程(2)——手写数字的识别
