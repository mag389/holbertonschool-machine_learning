#!/usr/bin/env python3
""" confusion matrix creation """
import numpy as np


def create_confusion_matrix(labels, logits):
    """ creates a confusion matrix
        labels: np.ndarray (m, classes) with correct labels in one-hot form
            m is number of data points
            classes is the number of classes
        logits: np.ndarray (m, classes) with predictions in one-hot form
        Returns: a confusino np.ndarray(classes, classes)
    """
    # print(labels.shape)
    classes = labels.shape[1]
    cm = np.zeros((classes, classes))
    # print(cm)
    for i in range(labels.shape[0]):
        x = np.where(labels[i] == 1)[0][0]
        y = np.where(logits[i] == 1)[0][0]
        #  print("into loop {} {}".format(x, y))
        # print(x)
        # print(x[0])
        # print(x[0][0])
        cm[x][y] = cm[x][y] + 1
        # print("stops att assignment")
    return cm
