# Computer-Vision, Implementation of CNN

## Understanding CNNs

### 1_Layer_CNN
this file cosist of 1 layer of each convolutional layer and pooling layer. 

 >> Convolution layer, including:
  * Zero Padding
  * Convolve window
  * Convolution forward
  
 >>Pooling layer, including:
  * Pooling forward
  * Create mask
  * Distribute value
  
### Application of CNN
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

### Resnets

Resnets are neural network of skip connection Designed to solve problem of very deep neural networks.

During training, you might therefore see the magnitude (or norm) of the gradient for the earlier layers descrease to zero very rapidly as training proceeds. To solve this problem of very deep neural networks resnets were invented.

![Readme1](https://user-images.githubusercontent.com/35829508/54498254-08b4d180-492b-11e9-8206-129a5ca7d075.PNG)

 >>Here're some other functions we used in the code below:
 * Conv2D
 * BatchNorm
 * Zero padding
 * Max pooling
 * Fully conected layer
 * Addition
 
![Readme2](https://user-images.githubusercontent.com/35829508/54498271-36017f80-492b-11e9-8440-825c9bfbd78b.PNG)

## imgLabeller

image labeller is a tool consist of a user interface and a tensorflow object detection api in background.
user dont have to draw bounding boxes manually, instead User can select whatever model, labels and threshold
for detection and save the results into a xml file.

![Readme1](https://github.com/shivendrapratap2/Computer-Vision/blob/master/ImageLabeller/imgLabeller.PNG)


## Face detection and recognition

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

## car detection with YOLOv2 

YOLO (You Only Look Once) is a popular algoritm because it achieves high accuracy while also being able to run in real-time. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

Yolo Algorithm for object localisation and detection divides the input image in the grids of 19 X 19. then applies convolution to each
grid and outputs the probability of detection of an object along with bounding box for that grid.

 Basic Building block for YOLO:
 * yolo_filter_boxes (filters those boxes which have very less chance of detection)
 * intersection over union (IOU, ratio of intersection of two bounding boxes with their area of union)
 * nonmax supression (to supress those bounding boxes which are trying to detect same object)
