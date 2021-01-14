#!/usr/bin/env python3
""" updates a variable with grad desc with momemtum """


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ update var with grad descent momentum
        alpha: learning rate
        beta1: momentum weight
            alpha and beta1 are the hyperparameters
        var: numpy.ndarray with variable to be updated
        grad: numpy ndarray containing gradient of var (dw)
        v: previous first moment of var (dw_prev)
        Return: updated variable and new moment respectively
    """
    dvar = beta1 * v + (1 - beta1) * grad
    var = var - alpha * dvar
    return var, dvar
