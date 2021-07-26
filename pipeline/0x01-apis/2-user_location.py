#!/usr/bin/env python3
""" prints location of user """
import requests
import sys
import time


if __name__ == "__main__":
    url = sys.argv[1]
    r = requests.get(url)
    if r.status_code == 403:
        reset_time = int(r.headers.get('X-Ratelimit-Reset'))
        now = time.time()
        minutes = reset_time - now
        minutes = round(minutes / 60)
        print("Reset in {} min".format(minutes))
        exit()
    user = r.json()
    location = user.get('location')
    if location:
        print(user.get('location'))
    else:
        print("Not found")
