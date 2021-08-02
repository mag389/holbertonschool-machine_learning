#!/usr/bin/env python3
""" returns schools with topic """
import pymongo


def schools_by_topic(mongo_collection, topic):
    """ mongo_collection: pymongo collection object
        topic: string to search
    """
    return mongo_collection.find({"topics": topic})
