
# How to convert the weights for Keras framework

1> Keep yad2k.py, keras_yolo.py, yolo_utils.py, keras_darknet19.py, utils.py in one directory (because these are dependent on one another)

along with your downloaded yolo.weights and yolov2.cfg file.

2> If you are unable to download yolov2.cfg just create an empty .cfg file in your directory and copy and pase the code from other cfg file.

3> Now run the command on your terminal (using python3)

"python yad2k.py yolov2.cfg yolo.weights whatever_your_file_name.h5"
  
this would create your .h5 of weights and you can load it in your model.
