#!/usr/bin/env python3
""" moving average """
import numpy as np


def moving_average(data, beta):
    """ calcs weighted moving avg of data set
        data: list of data to calculate from
        beta: weights used for moving avg
        Return: list containing moving avg of data
    """
    m_avg = [0] * len(data)
    last = 0
    for i in range(0, len(data)):
        last = beta * last + (1 - beta) * data[i]
        m_avg[i] = last / (1 - beta**(i + 1))
    return m_avg
