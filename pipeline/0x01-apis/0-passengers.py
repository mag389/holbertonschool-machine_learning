#!/usr/bin/env python3
""" list ships than can hold passengers """
import requests


def availableShips(passengerCount):
    """ returns list of hsips that can hold given number of passengers
        if not ships returns empty list
    """
    ships = []
    url = 'https://swapi-api.hbtn.io/api/starships'
    still_left = True
    while still_left is True:
        res = requests.get(url).json()
        for ship in res['results']:
            passen = ship['passengers']
            passen = passen.replace(',', '')
            if passen == "n/a" or passen == "unknown":
                continue
            if int(passen) >= passengerCount:
                ships.append(ship['name'])
        url = res['next']
        if url is None:
            still_left = False
    return ships
