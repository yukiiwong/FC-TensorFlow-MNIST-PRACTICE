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


![0](https://github.com/yukiiwong/FC-TensorFlow-MNIST-PRACTICE/blob/master/SAMPLE/0/10.bmp)![6](https://github.com/yukiiwong/FC-TensorFlow-MNIST-PRACTICE/blob/master/SAMPLE/1/12.bmp)![6](https://github.com/yukiiwong/FC-TensorFlow-MNIST-PRACTICE/blob/master/SAMPLE/6/3.bmp)

I put part of the data source in the SAMPLE folder. You will find that they are all white on black.

I used the drawing in Windows to draw the picture below. In order to improve the recognition, I painted the background of the picture as black.
And the image is expanded and etched by the Opencv library in Python.
![3](https://github.com/yukiiwong/FC-TensorFlow-MNIST-PRACTICE/blob/master/1.jpg)

### The files checkpoint, mnist.ckpt.data-00000-of-00001, mnist.ckpt.index and mnist.ckpt.meta are a FC nerual network that I have trained for 3000 times. And its accuracy is 0.953. The same situation is that also has a very pool recognition of number 6.Put these files in the same path to run the fcnn.py file. You can save nearly 45 minutes.

reference: Peter_Chan in Bilibili, TensorFlow系列教程(2)——手写数字的识别
