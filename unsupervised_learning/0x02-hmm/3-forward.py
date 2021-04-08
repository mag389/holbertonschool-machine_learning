#!/usr/bin/env python3
""" forward algo """
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """ performs forward algo for HMM
        Observation: np arr (T,) with index of observation(s)
          T: number of observations
        Emission: np arr (N, M) of emmission probabilities(spec observe|state)
          N: number of hidden states
          M: number of possible observations
          Emission[i, j] = prob observing J given state i
        Transition: np arr (N, N) transition probs
          Transition[i, j] = prob of transition form i to j
        Initial: np arr (N, 1) of prob of starting in specific hidden state
        Returns: P, F or None, None on failure
          P: likelihood of the observations given the model
          F: np arr (N, T) containing forward path probs
            F[i, j] is prob of being in hidden state i at time j given history
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
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for i in range(1, T):
        last = F[:, i - 1]
        cur = Transition * Emission[:, Observation[i]]
        F[:, i] = last @ (cur)
    P = np.sum(F[:, -1])
    return (P, F)
