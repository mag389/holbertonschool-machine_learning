#!/usr/bin/env python3
""" prints location of user """
import requests
import sys
import time


if __name__ == '__main__':
    url = sys.argv[1]
    res = requests.get(url)
    if res.status_code == 200:
        print(res.json()['location'])
    elif res.status_code == 403:
        rate_lim = res.headers['X-Ratelimit-Reset']
        res_time = int(reset) - int(time.time())
        print("reset in {} min".format(res_time))
    else:
        print("Not found")
