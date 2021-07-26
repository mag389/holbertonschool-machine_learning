#!/usr/bin/env python3
""" frequency of rocket launches byt spacex """
import requests
import time


if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/launches'
    res = requests.get(url)
    resjs = res.json()
    names_dict = {}
    count_dict = {}
    url_rocket = 'https://api.spacexdata.com/v4/rockets/'
    for launch in resjs:
        rocket = launch.get('rocket')
        if rocket not in names_dict.keys():
            url_name = url_rocket + rocket
            name = requests.get(url_name).json()['name']
            names_dict[rocket] = name
        if names_dict[rocket] not in count_dict.keys():
            count_dict[names_dict[rocket]] = 1
        else:
            count_dict[names_dict[rocket]] += 1
    rolist = sorted(count_dict.items(), key=lambda kv: kv[1], reverse=True)
    for item in rolist:
        print("{}: {}".format(item[0], item[1]))
    exit()
    for k, v in sorted(count_dict.items(), key=lambda kv: (kv[1], kv[0])):
        print("{}: {}".format(k, v))
    print(names_dict.items())
    print(count_dict.items())
    for k, v in names_dict.items():
        print("{}: {}".format(v, count_dict[v]))
    for k, v in count_dict.items():
        print
