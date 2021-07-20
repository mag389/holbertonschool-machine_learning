#!/usr/bin/env python3
""" create dataframe from file """
import pandas as pd


def from_file(filename, delimiter):
    """ loads data from a file
        filename: file to load from
        delimiter: column separator
        Returns: pd.DataFrame
    """
    return pd.read_csv(filename, sep=delimiter)
