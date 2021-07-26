#!/usr/bin/env python3
""" sentient species """
import requests


def sentientPlanets():
    """ returns list of names of home planets of all sentient species """
    planets = []
    url = 'https://swapi-api.hbtn.io/api/species/'
    while url:
        res = requests.get(url).json()
        for species in res['results']:
            hw = None
            if species['designation'] == "sentient":
                hw = species['homeworld']
            elif species['classification'] == "sentient":
                hw = species['homeworld']
            if hw is not None:
                planet = requests.get(hw).json()
                if planet is None or planet['name'] == "unknown":
                    continue
                planets.append(planet['name'])
        url = res['next']
    return planets
