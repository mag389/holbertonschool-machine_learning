#!/usr/bin/env python3
""" displays upcoming launch """
import requests
import time
import sys


if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/launches/upcoming'
    res = requests.get(url)
    resjs = res.json()
    soonest = None
    soonest_date = None
    for launch in resjs:
        if soonest is None:
            soonest = launch
            soonest_date = launch['date_unix']
            continue
        if launch['date_unix'] < soonest_date:
            soonest = launch
            soonest_date = launch['date_unix']
    """ launch name, date, rocket_name, launchpad_name, launchpad_locality """
    print(soonest['name'], end='')
    print(" ({})".format(soonest['date_local']), end='')

    url_rocket = 'https://api.spacexdata.com/v4/rockets/'
    rocket_name = requests.get(url_rocket + soonest['rocket']).json()['name']
    print(" {}".format(rocket_name), end='')

    url_launchpad = 'https://api.spacexdata.com/v4/launchpads/'
    launchpad = url_launchpad + soonest['launchpad']
    launchpad = requests.get(launchpad).json()
    print(" - {}".format(launchpad['name']), end='')
    print(" ({})".format(launchpad['locality']))
