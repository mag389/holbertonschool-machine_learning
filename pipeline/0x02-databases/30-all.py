#!/usr/bin/env python3
""" list documents in collection with python """
import pymongo


def list_all(mongo_collection):
    """ lists all documents from specified colleciton """
    return list(mongo_collection.find())
