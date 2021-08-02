#!/usr/bin/env python3
""" updates topics of school document """
import pymongo


def update_topics(mongo_collection, name, topics):
    """ mongo_collection: pymongo collection object
        name: string school name tp update
        topics: list of topics approached in the school
    """
    mongo_collection.update_many({"name": name},
                                 {'$set':
                                 {"name": name, "topics": topics}})
