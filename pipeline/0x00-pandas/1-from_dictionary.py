#!/usr/bin/env python3
""" create dataframe from dict """
import pandas as pd


dictio = {}
dictio['First'] = [0.0, 0.5, 1.0, 1.5]
dictio['Second'] = ['one', 'two', 'three', 'four']

indices = ['A', 'B', 'C', 'D']

# df = pd.DataFrame(dictio, index=indices)
# i couldn't determine if sorting was needed, but i left it in
df = pd.DataFrame(dictio, index=indices, columns=sorted(dictio.keys()))
