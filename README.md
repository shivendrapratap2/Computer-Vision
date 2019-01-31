# Computer-Vision, Implementation of CNN

# 1_Layer_CNN
this file cosist of 1 layer of each convolutional layer and pooling layer. 

 >> Convolution layer, including:
  * Zero Padding
  * Convolve window
  * Convolution forward
  
 >>Pooling layer, including:
  * Pooling forward
  * Create mask
  * Distribute value
  
# Application of CNN
this python program uses tensorflow module of python to design a simple neural network of few conv and pooling layers  which would tell what numerical numbers (0 - 6) a hand sign is showing.

 >> The model :
 * creates placeholders
 * initializes parameters
 * forward propagates
    Implements a three-layer ConvNet in Tensorflow:
       CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
 * computes the cost
 * creates an optimizer

Above neural network is shallow network of few layers, but gives a very fine understanding of basic building blocks of a neural network.
and many neural networks like this combines up to makes a deep neural network.


# Happy_house

This happy_house is a neural network designed using keras (a high-level neural networks API (programming framework), written in Python and capable of running on top of several lower-level frameworks including TensorFlow and CNTK.) to decide that person in a image is happy or not. images would be taken by a camera attached on front door of house.

Details of the "Happy" dataset:
 * Images are of shape (64,64,3)
 * Training: 600 pictures
 * Test: 150 pictures
 
 To train and test a model, there are four steps in Keras:

 * Create the model by calling the function (Model(inputs = X_input, outputs = X, name='HappyModel')) //from keras.models import Model
 * Compile the model by calling model.compile(optimizer = "...", loss = "...", metrics = ["accuracy"])
 * Train the model on train data by calling model.fit(x = ..., y = ..., epochs = ..., batch_size = ...)
 * Test the model on test data by calling model.evaluate(x = ..., y = ...)

# Resnets

Resnets are neural network of skip connection Designed to solve problem of very deep neural networks.

During training, you might therefore see the magnitude (or norm) of the gradient for the earlier layers descrease to zero very rapidly as training proceeds. To solve this problem of very deep neural networks resnets were invented.

<p align="center"> <img src="shivendrapratap2/Computer-Vision/Readme1.png"/> </p>

 >>Here're some other functions we used in the code below:
 * Conv2D
 * BatchNorm
 * Zero padding
 * Max pooling
 * Fully conected layer
 * Addition








Yet to be updated and upload of necessary file is pending. 
