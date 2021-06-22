#!/usr/bin/env python3
""" loads premade frozen lake environment """
import gym
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """ loads premade env from openai gym
        desc: either none or a list of lists of custom description of map
        map_name: either none or a string containing name of premade map
          (if both desc and map name are none will load randomly
        is_slippery: bool for if ice is slippery
        returns: the env
    """
    
    env = FrozenLakeEnv(desc, map_name, is_slippery)
    return env
