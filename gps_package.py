import pandas as pd
from itertools import combinations, permutations

class community(object):
    def __init__(self, user_ids=[], locations=[]):

class user(object):
    def __init__(self, id=None, name=None, trips=[], locations=[], community_members=None):
        self.uid = id
        self.name = name
        if trips:
            self.trips=trips
            start_points = [trip[0] for trip in trips]
            end_points = [trip[1] for trip in trips]
            self.locations = list(set(start_points + end_points))
        elif locations:
            self.locations = locations
            self.trips = permutations(locations,2)
        else:
            self.trips = []
            self.locations = []

        if community_members:
            self.neighbors = community_members

    def add_trip(self, trip):
        self.trips.append(trip)

    def add_location(self, location):
        self.locations.append(location)

    def add_community(self, member):
        if isinstance(member, list):
            self.community_members = self.community_members + member
        else:
            self.community_members.append(member)

    def set_id(self, uid):
        self.uid = uid



    def __repr__(self):
        return f'ID: {self.uid}; Num_trips = {len(self.trips)}'