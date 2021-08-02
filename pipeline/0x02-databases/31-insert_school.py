#!/usr/bin/env python3
""" inserts into collection """
import pymongo


def insert_school(mongo_collection, **kwargs):
    """ inserts new doc into collection
        mongo_collection: pymongo collection object
        Returns: new _id
    """
    return mongo_collection.insert(kwargs)
