#!/usr/bin/env python3
""" policy gradient function file """
import numpy as np


def policy(state, weight):
    """ computes to policy with a weight of a matrix
        does via the softmax of the observations * the weights
        returns a policy which is probabilities of weights
    """
    unweighted = state @ weight
    # print(unweighted)
    ret = np.exp(unweighted - np.max(unweighted))
    # ret = np.exp(unweighted)
    # either works just changes when normalization occurs
    return ret / np.sum(ret)


def policy_gradient(state, weight):
    """ computes the monte-carlo policy gradient based on matrix and weights
        state: matrix representing current observation of environment
        weight: matrix of random weight
        return: action and gradient ( in that order)
    """
    pol = policy(state, weight)
    print("created policy")
    action = np.random.choice(pol[0].shape[0], p=pol[0])
    print("selected action")
    s = pol.reshape(-1, 1)
    print("made s")
    dsoftmax = (np.diagflat(s) - np.dot(s, s.T))[action, :]
    print("made dsoftmax")
    dlog = dsoftmax / pol[0, action]
    print("made dlog")
    grad = state.T.dot(dlog[None, :])
    print("made grad")
    return action, grad
