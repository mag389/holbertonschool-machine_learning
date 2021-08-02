#!/usr/bin/env python3
""" script for displaying some logged info """
# import pymongo
from pymongo import MongoClient


if __name__ == '__main__':
    client = MongoClient()
    db = client.logs.nginx
    print("{} logs".format(db.count()))
    print("Methods:")
    get = db.count({"method": "GET"})
    # get = db.find({"method": "GET"})
    post = db.find({"method": "POST"})
    put = db.find({"method": "PUT"})
    patch = db.find({"method": "PATCH"})
    delete = db.find({"method": "DELETE"})
    print("\tmethod GET: {}".format(get))
    print("\tmethod POST: {}".format(post.count()))
    print("\tmethod PUT: {}".format(put.count()))
    print("\tmethod PATCH: {}".format(patch.count()))
    print("\tmethod DELETE: {}".format(delete.count()))

    status = db.find({'method': 'GET', 'path': '/status'})
    print("{} status check".format(status.count()))
