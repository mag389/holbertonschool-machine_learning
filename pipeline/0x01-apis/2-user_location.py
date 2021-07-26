#!/usr/bin/env python3
""" prints location of user """
import requests
import sys
import time


if __name__ == '__main__':
    url = sys.argv[1]
    res = requests.get(url)
    if res.status_code == 403:
        rate_lim = res.headers['X-Ratelimit-Reset']
        res_time = int(reset) - int(time.time())
        print("Reset in {} min".format(res_time))
        exit()
    acc = res.json()
    loc = acc.get('location')
    if loc is not None:
        print(loc)
    else:
        print(("Not found"))
    exit()
    """
    elif res.status_code == 200:
        loc = res.json()['location']
        if loc is not None:
            print(loc)
        else:
            print("Not found")
    else:
        print("Not found")
    """
