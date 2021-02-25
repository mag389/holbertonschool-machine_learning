#!/usr/bin/env python3
""" first yolo algorithm class """
import tensorflow.keras as K


class Yolo():
    """ the yolo class for obj detection with yolo algo """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ initialize yolo object
            model_path: path to where darknet keras model is stored
            classes_path: path where list of class nams used for darknet model
                listed in order of index
            class_t: is a float representing box score threshold for init
            nms_t: float of IOU threshold for non-max suppression
            anchors: np.ndarray of shape (uotputs, anchor_boxes, 2) contains
                anchor boxes.
                outputs: number of outputs (predictions) made by darknet model
                anchor_boxes: number of boxes used for each pred
                2:[anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(model_path)
        self.class_names = []
        with open(classes_path, 'r') as classes:
            for line in classes:
                self.class_names.append(line.strip())
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
