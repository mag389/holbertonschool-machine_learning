#!/usr/bin/env python3
""" positional encoding for transformer """
import numpy as np


def positional_encoding(max_seq_len, dm):
    """ calculates positional encoding for a transformer
        max_seq_lem: int representing max seq len
        dm: model depth
        Returns: np arr (max_seq_len, dm) of positional encoding vectors
    """
    pe = np.ones((max_seq_len, dm))
    # loops
    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            den = np.power(10000, ((2 * (i//2))/dm))
            pe[pos, i] = np.sin(pos / den)
            den = np.power(10000, (2 * ((i + 1)//2))/dm)
            pe[pos, i + 1] = np.cos(pos / den)
    return pe

    pe *= np.arange(max_seq_len)
    return pe
    pos = np.arange(max_seq_len)
    den = np.power(10000, (np.arange(dm) // 2) * 2 / np.float32(dm))
    pe = pos[:, np.newaxis] * den[np.newaxis, :]
    # pe *= pos * den.T

    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])

    return pe
