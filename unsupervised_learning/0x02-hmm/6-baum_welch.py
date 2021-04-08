#!/usr/bin/env python3
""" perform baum welch algo """
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """ performs baum-welch for a hidden markov model
        Observation: np arr (T,) with index of observation(s)
          T: number of observations
        Emission: np arr (N, M) of emmission probabilities(spec observe|state)
          N: number of hidden states
          M: number of possible observations
          Emission[i, j] = prob observing J given state i
        Transition: np arr (N, N) transition probs
          Transition[i, j] = prob of transition form i to j
        Initial: np arr (N, 1) of prob of starting in specific hidden state
        iterations: number of iterations to perform E-M
        Returns:Transiton, Emission, or None, None
    """
    if type(Observations) is not np.ndarray or len(Observations.shape) != 1:
        return None, None
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    T = Observations.shape[0]
    N, M = Emission.shape
    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None
    if Initial.shape[0] != N:
        return None, None
    if type(iterations) is not int or iterations < 0:
        return None, None
    A = Transition
    O = Observations
    B = Emission
    pi = Initial
    # start test method
    for n in range(iterations):
        forw, alpha = forward(O, B, A, pi)
        back, beta = backward(O, B, A, pi)
        # print(alpha.shape)
        alpha, beta = alpha.T, beta.T
        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            np.dot(alpha[t, :].T, A)
            np.dot(alpha[t, :].T, A) * B[:, O[t + 1]].T
            denom = np.dot(np.dot(alpha[t, :].T, A) * B[:, O[t + 1]].T, beta[t + 1, :])
            for i in range(N):
                num = alpha[t, i] * A[i, :] * B[:, O[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = num / denom
        gamma = np.sum(xi, axis=1)
        A = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
        K = B.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            B[:, l] = np.sum(gamma[:, O == l], axis=1)
        B = np.divide(B, denominator.reshape((-1, 1)))
    return A, B
    # end test
    A = Transition
    O = Observations
    B = Emission
    pi = Initial
    gamma = np.ones((N, T))
    theta = np.zeros((N, N, T))
    for i in range(iterations):
        forw, alpha = forward(O, B, A, pi)
        back, beta = backward(O, B, A, pi)
        F, B = alpha, beta
        P = alpha * beta
        P = P / np.sum(P, 0)
        # print(P)
        # for j in range(T):
        #     gamma[:, j] = alpha[j] * beta[j] / (alpha[j] @ beta[j])
        old_A = np.copy(A)
        old_O = np.copy(B)
        A = np.ones((N, N))
        B = np.ones((N, T))
        # get transition probs at each time step
        for a_ind in range(N):
            for b_ind in range(N):
                for t_ind in range(T - 1):
                    theta[a_ind, b_ind, t_ind] = \
                    F[a_ind, t_ind] * \
                    B[b_ind, t_ind + 1] * \
                    old_A[a_ind, b_ind] * \
                    old_O[b_ind, O[t_ind]]
        # for new A mat and O mat
        for a_ind in range(N):
            for b_ind in range(N):
                A[a_ind, b_ind] = np.sum(theta[a_ind, b_ind, :])/ \
                                  np.sum(P[a_ind,:])
        A = A / np.sum(A, 1)
        for a_ind in range(N):
            for o_ind in range(T):
                right_obs_ind = np.array(np.where(O == o_ind)) + 1
                O[a_ind, o_ind] = np.sum(P[a_ind, right_obs_ind])/ \
                                  np.sum(P[a_ind, 1:])
        O = O / np.sum(O, 1)
        if np.linalg.norm(old_A - A) < .00001:
            if  np.linalg.norm(old_O - O) < .00001:
                break
    return A, O
    print(gamma)
    gamma1 = alpha * beta / np.sum(alpha * beta, 0)
    print(gamma1)
    print(gamma == gamma1)
    return gamma

    

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


def backward(Observation, Emission, Transition, Initial):
    """ performs backward algo for HMM
        Observation: np arr (T,) with index of observation(s)
          T: number of observations
        Emission: np arr (N, M) of emmission probabilities(spec observe|state)
          N: number of hidden states
          M: number of possible observations
          Emission[i, j] = prob observing J given state i
        Transition: np arr (N, N) transition probs
          Transition[i, j] = prob of transition form i to j
        Initial: np arr (N, 1) of prob of starting in specific hidden state
        Returns: P, B or None, None on failure
          P: likelihood of the observations given the model
          B: np arr (N, T) containing backward path probs
            F[i, j] is prob of generating future observations from hidden state
              i at time j
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

    B = np.ones((N, T))
    for i in reversed(range(0, T - 1)):
        later = B[:, i + 1] * Transition
        cure = Emission[:, Observation[i + 1]]
        B[:, i] = later @ cure
    P = np.sum(B[:, 0] * Initial.T * Emission[:, Observation[0]])
    return P, B
