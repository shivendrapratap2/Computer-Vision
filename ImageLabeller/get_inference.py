import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util


def load_model(MODEL_FOLDER):
    
    # Grab path to current working directory
    CWD_PATH = os.getcwd()    
    MODEL_FOLDER = MODEL_FOLDER
    
    # Path to frozen detection graph .pb file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_FOLDER, "frozen_inference_graph.pb")
    
    
    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
        sess = tf.Session(graph=detection_graph)
    
    return detection_graph, sess


def Predict(img, detection_graph, sess, MODEL_FOLDER, labels2show, threshold= 0.7):
    
    # Grab path to current working directory
    CWD_PATH = os.getcwd()
    
    # Number of classes the object detector can identify
    NUM_CLASSES = 90
    
    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_FOLDER, 'mscoco_complete_label_map.pbtxt') 
    
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    # color scheeme for different classes
    color_map = {'person': 'DeepSkyBlue', 'dog': 'IndianRed', 'cat':'yellow', 'chair': 'Cyan', 'bottle': 'Orange'}
    
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    
    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    
    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    
    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    # threshold of detection
    thresh= threshold
    
    items = []
    coordinates = []
    #if you want to resize to tune inference
    #img = cv2.resize(img_org, (300,300))
    img_expanded = np.expand_dims(img, axis=0)
    #print(img_expanded.shape)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: img_expanded})
    
    objects = []
    for index, value in enumerate(classes[0]):
        object_dict = {}
        if scores[0, index] > thresh:
            object_dict[(category_index.get(value)).get('name')] = scores[0, index]
            objects.append(object_dict)
            #print (objects)
            
    #Get all the detected class labels in one list
    for y in objects:
        for keys in y.keys():
            m = list(y.keys())[0]
            items.append(m)
             
    #Get co ordinates of the detected classes
    coordinates = vis_util.return_coordinates(
                img,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=10,
                min_score_thresh=thresh)

    new_items= []
    new_coordinates= []
    display_str_list= []
    for i, item in enumerate(items):
        if item.lower() in labels2show:
            new_items.append(item.lower())
            new_coordinates.append(coordinates[i][:-1])
            display_str_list.append([item+' : '+str(int(coordinates[i][-1]))+'%'])
                   
    for i,box in enumerate(new_coordinates):
        
        ymin, ymax, xmin, xmax  = box[0], box[1], box[2], box[3]
        display_str = display_str_list[i]
        color = color_map[new_items[i]]      
        vis_util.draw_bounding_box_on_image_array(img,
                                                ymin,
                                                xmin,
                                                ymax,
                                                xmax,
                                                color=color,
                                                thickness=4,
                                                display_str_list=display_str,
                                                use_normalized_coordinates=False)
        
    return  new_coordinates,new_items,img