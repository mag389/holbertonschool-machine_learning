#!/usr/bin/env python3
""" first yolo algorithm class """
import tensorflow.keras as K
import tensorflow as tf
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

    def filter_boxes(self, boxes, box_confidence, box_class_probs):
        """ filter the output boxes
            args are all list of arrays
            boxes: np.ndarray(g_h, g_w, a_b, 4): processed boundary boxes
            box_confidences:(g_h, g_W, a_b, 1): processed confidence for the
                boxes for each  outpput
            box_class_probs:(g_h, g_w, a_b, classes): processed box class
                probabilties for each output
            Returns: tuple of (filtered_boxes, box_classes, box_scores):
              filtered_boxes: np.array shape (?, 4) of all filtered bound boxes
              box_classes: array (?, ) contains class number that each box in
                  filtered_boxes represents
              box_Scores: array shape (?) containing box scores for each box in
                  filtered_boxes
        """
        f_boxes = []
        b_classes = []
        b_scores = []
        for i in range(len(boxes)):
            boxscore = box_confidence[i] * box_class_probs[i]
            maxes = np.amax(boxscore, axis=3)
            keep = np.argwhere(maxes[:, :, :] >= self.class_t)

            for kept in keep:
                f_boxes.append(boxes[i][kept[0], kept[1], kept[2]])
                b_classes.append(np.argmax(boxscore[kept[0],
                                                    kept[1], kept[2]]))
                b_scores.append(maxes[kept[0], kept[1], kept[2]])
        """ muchj easier in tf 2.x

            box_class = tf.argmax(boxscore, axis=-1)
            box_score = tf.math.reduce_max(boxscore, axis=-1)
            mask = boxscore >= self.class_t

            boxes = tf.compat.v1.boolean_mask(boxes, mask)
            scores = tf.compaat.v1.boolean_mask(boxscore, mask)
            classes = tf.compat.v1.boolean_mask(box_class, mask)

            f_boxes.append(boxes)
            b_classes.append(classes)
            b_scores.append(scores)
        """
        filtered_boxes = np.array(f_boxes)
        box_classes = np.array(b_classes)
        box_scores = np.array(b_scores)
        return (filtered_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """ performs non-max suppression on filtered boxes
            args are ouput of filter function
            loop through classes and for each class only keep boxes that are
              sufficiently unique i.e. IoU
            output is the same arrays and shapes, but with fewer elements.
              boxes are now sorted by class
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []
        for i in range(len(self.class_names)):
            idx = np.where(box_classes == i)
            boxes = filtered_boxes[idx]
            classes = box_classes[idx]
            scores = box_scores[idx]

            # with boxes of single class perform nms
            iou = self.nms_t
            scorecpy = np.copy(scores)
            # for k in range(len(boxes)):
            while np.amax(scorecpy) != -1:
                k = np.argmax(scorecpy)
                scorecpy[k] = -1
                scoretmp = scores[k]
                # scores[k] = -1
                classtmp = classes[k]
                # print(classtmp)
                classes[k] = -1
                discard = False
                box1 = boxes[k]
                bx1, by1, bx2, by2 = box1
                barea = (bx1 - bx2) * (by1 - by2)
                for j in range(len(boxes)):
                    if scores[j] == -1:
                        continue
                    if scorecpy[j] == -1 or classes[j] == -1:
                        continue
                    box2 = boxes[j]
                    cx1, cy1, cx2, cy2 = box2
                    carea = (cx1 - cx2) * (cy1 - cy2)
                    if cx1 > bx2 or cx2 < bx1 or cy1 > by2 or cy2 < by1:
                        continue
                    ox1 = np.maximum(bx1, cx1)
                    ox2 = np.minimum(bx2, cx2)
                    oy1 = np.maximum(by1, cy1)
                    oy2 = np.minimum(by2, cy2)
                    # safe check but no change
                    # if ox2 - ox1 <= 0 or oy2 - oy1 <= 0:
                    #     continue
                    overlap = (ox1 - ox2) * (oy1 - oy2)
                    union = barea + carea - overlap
                    frac = overlap / union
                    if scores[k] == 0.6397598:
                        print(overlap, union, frac, scores[j])
                    if frac > iou:
                        if scores[j] > scoretmp:
                            discard = True
                            # scores[k] = -1
                            scorecpy[k] = -1
                            classes[k] = -1
                            # break
                        else:
                            scorecpy[j] = -1
                            scores[j] = -1
                            classes[j] = -1
                if not discard:
                    box_predictions.append(boxes[k])
                    predicted_box_classes.append(classtmp)
                    predicted_box_scores.append(scoretmp)
            box_p = np.array(box_predictions)
            p_box_c = np.array(predicted_box_classes)
            p_box_s = np.array(predicted_box_scores)
        return (box_p, p_box_c, p_box_s)
