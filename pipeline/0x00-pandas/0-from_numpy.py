#!/usr/bin/env python3
""" creates pd dataframe from numpy """
import pandas as pd


def from_numpy(array):
    """ creates pd.DataFrame from np.ndarray
        array: np arr from which to create datafram
        columns of pd.DataFrame labeled in alphabetical and capitalized
        Returns: newly created pd.DataFrame
    """
    alphabet = list(map(chr, range(65, 91)))
    return pd.DataFrame(array, columns=alphabet[0:array.shape[1]])
