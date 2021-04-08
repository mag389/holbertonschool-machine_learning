#!/usr/bin/env python3
""" viterbi algo """
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """ calculates most likely sequence of hidden states for hmm using viterbi
        Observation: np arr (T,) with index of observation(s)
          T: number of observations
        Emission: np arr (N, M) of emmission probabilities(spec observe|state)
          N: number of hidden states
          M: number of possible observations
          Emission[i, j] = prob observing J given state i
        Transition: np arr (N, N) transition probs
          Transition[i, j] = prob of transition form i to j
        Initial: np arr (N, 1) of prob of starting in specific hidden state
        Returns: Path, P or None, None on failure
          Path: list of length T containing most likely sequence os states
          P probability of obtaining Path
    """
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    T = Observation.shape[0]
    N, M = Emission.shape
    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None
    if Initial.shape[0] != N:
        return None, None

    F = np.zeros((N, T))
    bt = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    b = Observation
    a = Transition
    E = Emission
    for i in range(1, T):
        last = F[:, i - 1] * Transition.T
        cur = Emission[np.newaxis, :, Observation[i]].T
        # F[:, i] = np.max(last * cur, axis=1)
        F[:, i] = np.max((F[:, i - 1] * a.T) * E[np.newaxis, :, b[i]].T, 1)
        bt[:, i] = np.argmax(last, axis=1)

    path = [0] * T
    path[-1] = np.argmax(F[:, T - 1])
    for i in reversed(range(1, T)):
        path[i - 1] = int(bt[path[i], i])
        # path[i - 1] = np.argmax(F[:, i])
    return path, np.max(F[:, -1])
