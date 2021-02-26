#!/usr/bin/env python3
""" first yolo algorithm class """
import tensorflow.keras as K
import numpy as np


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

    def process_outputs(self, outputs, image_size):
        """ processes the outputs
            outputs: list of np.ndarrays of predictions of model
              outputs shape:(grid_height, grid_width, anchor_boxes, 4 +
                             1 + classes)
                grid height/width: size of output grid
                anchor_boxes: number of anchor boxes used
                4 + 1: (t_x, t_y, t_w, t_h) + box_confidence
                classes: class probs for all classes
            image_size: np.ndarray of images original size [h, w]
            Returns: tuple of (boxes, box_confidence, box_class_probs)
              boxes: list of arrays of processes boundary boxes for each output
              box_confidence: list of arrays (grid_h, widht, anch_boxes, 1)
              box_class_probs: list arrays shape(grid_h, w, anch_box, classes)
        """
        boxes = []
        box_confidence = []
        box_class_probs = []
        i_h = image_size[0]
        i_w = image_size[1]
        for i in range(len(outputs)):
            output = outputs[i]
            g_h, g_w, a_b, c = output.shape
            classes = c - 5

            box = np.zeros((g_h, g_w, a_b, 4))

            # calculate the sigmoid box conf
            box_conf = 1 / (1 + np.exp(-1 * outputs[i][:, :, :, 4:5]))
            box_confidence.append(box_conf)

            # calculate sigmoid box class probabilities
            class_prob = 1 / (1 + np.exp(-1 * outputs[i][:, :, :, 5:]))
            box_class_probs.append(class_prob)
            for row in range(g_h):
                for col in range(g_w):
                    # tx, ty, tw, th = output[row][col][:, :4]
                    tx = output[row, col, :, 0]
                    ty = output[row, col, :, 1]
                    tw = output[row, col, :, 2]
                    th = output[row, col, :, 3]

                    pw = self.anchors[i, :, 0]
                    ph = self.anchors[i, :, 1]

                    bx = 1 / (1 + np.exp(-1 * tx)) + col
                    by = 1 / (1 + np.exp(-1 * ty)) + row
                    bw = np.exp(tw) * pw
                    bh = np.exp(th) * ph

                    # normalize to be able to fit
                    bx /= g_w
                    by /= g_h
                    bw /= int(self.model.input.shape[1])
                    bh /= int(self.model.input.shape[2])
                    # fit to our data
                    x1 = (bx - bw / 2) * i_w
                    x2 = (bx + bw / 2) * i_w
                    y1 = (by - bh / 2) * i_h
                    y2 = (by + bh / 2) * i_h
                    # print(y2.shape)
                    # print(box.shape)
                    box[row, col, :, :] = np.array([x1, y1, x2, y2]).T
            boxes.append(box)
            """
            # calculate new boundary boxes
            # bx = sigmoid(t_x) + c_x
            # by = sigmoid(t_y) + c_y
            # bw = p_w * exp(t_w)
            # bh = p_h * exp(t_h)
            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]

            bx = 1 / (1 + np.exp(-1 * t_x)
            by = 1 / (1 + np.exp(-1 * t_y)
            bw = np.exp(t_w)
            bh = np.exp(t_h)
            boxXY = 1 / (1 + np.exp(-1 * output[..., 0:2]))
            boxWH = tf.math.exp(output[..., 2:4])
            # has shape (gh, gw, ab, 2)
            # outputanch = self.anchors.shape[0]
            # newanch = self.anchors.reshape(1, 1, ouptutanch, a_b, 2)
            boxWH = boxWH * self.anchors[i].reshape(1, 1, a_b, 2)
            # c_x, c_y top left corners of grid box

            # pw, ph: ancor dimensions for box
            # pw = self.anchors[i, :, 0]
            # ph = self.anchors[i, :, 1]
            # bw = bw * pw
            # bh = bh * ph
            """
        return (boxes, box_confidence, box_class_probs)
