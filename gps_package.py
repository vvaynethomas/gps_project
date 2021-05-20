import pandas as pd
import numpy as np
from itertools import combinations, permutations
import geopy
import osmnx as ox
from osmnx import distance
import string

# to be replaced by lookups with Nominatim
# place = 'Shanghai, China'
# G = ox.graph_from_place(place, network_type='drive', simplify=True, retain_all=False)
# nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
print('loading data')
full_data = pd.read_excel('shanghai_gps.xlsx', skiprows=6, header=0)
full_data.columns = ['_'.join(name.lower()
                              .translate(str.maketrans(' ', ' ', string.punctuation)).split()) for name in
                     full_data.columns]


def extract_user(df, id):
    filtered_df = df[df.uid == id]
    new_user = User(id=id)
    for index, row in filtered_df.iterrows():
        print(type(row))
        new_user.add_trips(Trip(row))
    return new_user


def extract_users(dataframe):
    users = []
    for id in dataframe.uid.unique():
        new_user = extract_user(dataframe, id)
        users.append(new_user)
    return users


def extract_trips(dataframe):
    trips = dataframe.apply(lambda x: Trip(full=x), axis=1)
    return trips


class Visit(object):
    def __init__(self, uid=np.nan, timestamp=np.nan, latitude=np.nan, longitude=np.nan, nn_id=np.nan):
        self.uid = uid
        self.timestamp = timestamp
        self.latitude = latitude
        self.longitude = longitude
        self.location = (latitude, longitude)
        # self.nn_id = nn_id
        # global G
        # if not nn_id or nn_id == np.nan:
        #     # place = lookup_city(self.latitude,self.longitude)
        #     self.nn_id = ox.distance.nearest_nodes(G, self.longitude, self.latitude)

    def __add__(self, other):
        if type(other) == type(self):
            return Trip(uid=self.uid
                        , start_time=self.timestamp
                        , end_time=other.timestamp
                        , start_loc=self.location
                        , end_loc=other.location)
        else:
            print('can only form trips between visits')


class Trip(object):
    def __init__(self, full=None, uid=0, start_time=None, end_time=None, start_loc=None, end_loc=None, trip_id=0):
        self.visits = []
        if full is not None:
            if isinstance(full, list):
                if isinstance(full[0], int):
                    self.uid = full[0]
                    self.start_time = full[1]
                    self.end_time = full[2]
                    if isinstance(full[3], tuple) and isinstance(full[4], tuple):
                        self.start_loc = full[3]
                        self.start_lat = self.start_loc[0]
                        self.start_lng = self.start_loc[1]
                        self.visits.append(Visit(uid=self.uid, timestamp=self.start_time, latitude=self.start_lat,
                                                 longitude=self.start_lng))
                        self.end_loc = full[4]
                        self.end_lat = self.end_loc[0]
                        self.end_lng = self.end_loc[1]
                        self.visits.append(
                            Visit(uid=self.uid, timestamp=self.end_time, latitude=self.end_lat, longitude=self.end_lng))
                elif isinstance(full[0], Visit):
                    v1 = full[0]
                    v2 = full[1]
                    self.visits = [v1, v2]
                    self.uid = v1.uid
                    self.start_time = v1.timestamp
                    self.end_time = v2.timestamp
                    self.start_loc = v1.location
                    self.end_loc = v2.location
                    self.start_lat = v1.latitude
                    self.start_lng = v1.longitude
                    self.end_lat = v2.latitude
                    self.end_lng = v2.longitude

            elif isinstance(full, pd.Series):
                print('extracting trip from series')
                self.uid = full.uid
                self.start_time = full.start_time_in_seconds
                self.end_time = full.end_time_in_seconds
                self.start_lng = full.start_longitude
                self.start_lat = full.start_latitude
                self.end_lng = full.end_longitude
                self.end_lat = full.end_latitude
                self.start_loc = (self.start_lat, self.start_lng)
                self.end_loc = (self.end_lat, self.end_lng)
                self.visits.append(
                    Visit(uid=self.uid, timestamp=self.start_time, latitude=self.start_lat, longitude=self.start_lng))
                self.visits.append(
                    Visit(uid=self.uid, timestamp=self.end_time, latitude=self.end_lat, longitude=self.end_lng))


        else:
            if uid:
                self.uid = uid
            else:
                self.uid = np.nan
            if start_time:
                self.start_time = start_time
            else:
                self.start_time = np.nan
            if end_time:
                self.end_time = end_time
            else:
                self.end_time = np.nan
            if start_loc:
                self.start_loc = start_loc
                self.start_lat = start_loc[0]
                self.start_lng = start_loc[1]
            else:
                self.start_loc, self.start_lat, self.start_lng = np.nan, np.nan, np.nan
            if end_loc:
                self.end_loc = end_loc
                self.end_lat = end_loc[0]
                self.end_lng = end_loc[1]
            else:
                self.end_loc, self.end_lat, self.end_lng = np.nan, np.nan, np.nan
        if (isinstance(self.start_time, int) and isinstance(self.end_time, int)):
            self.duration = self.end_time - self.start_time
        else:
            self.duration = np.nan
        self.tid = trip_id

    def __repr__(self):
        return print(self.as_list())

    def __str__(self):
        return str(self.as_list())

    def as_list(self):
        return [self.uid, self.start_time, self.end_time, self.start_lng, self.start_lat, self.end_lng, self.end_lat,
                self.tid]

    def as_row(self):
        return {
            'uid': self.uid
            , 'start_time_in_seconds': self.start_time
            , 'end_time_in_seconds': self.end_time
            , 'start_longitude': self.start_lng
            , 'start_latitude': self.start_lat
            , 'end_longitude': self.end_lng
            , 'end_latitude': self.end_lat
            , 'tid': self.tid
        }


class Community(object):  # for collaborative filtering
    def __init__(self, users=None, data=None):
        global full_data
        if users:
            print(users)
            if isinstance(users[0], User):
                self.users = users
            elif isinstance(users[0], int):
                self.users = [extract_user(full_data,i) for i in users]

        elif data:
            self.users = extract_users(data)
        self.trips = [trip for trips in [user.trips for user in self.users] for trip in trips]
        self.visits = [visit for visits in [trip.visits for trip in self.trips] for visit in visits]
        self.trips_df = pd.DataFrame(data=[trip.as_row() for trip in self.trips], columns=['uid'
            , 'start_time_in_seconds'
            , 'end_time_in_seconds'
            , 'start_longitude'
            , 'start_latitude'
            , 'end_longitude'
            , 'end_latitude'
            , 'tid'])
        if users:
            if isinstance(users,list):
                if isinstance(users[0],int):
                    self.users = [extract_user(full_data,uid) for uid in users]
                    uids=users
                elif isinstance(users[0],User):
                    self.users = users
                    uids = [user.uid if user.uid else np.nan for user in users]
            if isinstance(users,int):
                self.users = [extract_user(full_data,users)]

        elif data:
            uids = data.uid.unique()

        else:
            self.users = []
        self.index = 0
        if uids:
            self.uids = uids
        else:
            self.uids = list(set([i.uid for i in self.users]))


class User(object):
    def __init__(self, id=0, trips=None, locations=None, visits=None, connected_users=None):
        self.uid = id
        self.visits = []
        self.trips = []
        self.locations = []
        if trips:
            self.trips = self.trips + trips
            start_points = [trip.start_loc for trip in trips]
            end_points = [trip.end_loc for trip in trips]
            self.locations = self.locations + start_points + end_points
            self.visits = self.visits + [trip.visits for trip in trips]
        if locations:
            self.locations = list(set(self.locations + locations))
            self.visits = self.visits + [Visit(uid=self.uid, latitude=loca[0], longitude=loca[1]) for loca in locations]
            if len(trips) == 0:
                self.trips = []
        if visits:
            self.visits = self.visits + visits
            self.locations = self.locations + [visit.location for visit in visits]
        elif locations:
            self.locations = locations
            self.trips = permutations(locations, 2)
        if visits:
            self.visits = self.visits + visits

        else:
            self.trips = []
            self.visits = []
            self.locations = []

        if connected_users:
            if isinstance(connected_users[0], int):
                self.connections = [extract_user(full_data,id) for id in connected_users]
            elif isinstance(connected_users[0], User):
                self.connections = connected_users
        else:
            print('no connections')
            self.connections = []

    def add_trips(self, trips_to_add):
        if isinstance(trips_to_add, list):
            if isinstance(trips_to_add[0], Trip):
                for trip in trips_to_add:
                    prev_tids = [t.tid for t in self.trips]
                    if trip.tid in prev_tids:
                        trip.tid = max(prev_tids) + 1
                    trip.uid = self.uid
                    self.trips.append(trip)
                    self.locations.append(trip.start_loc)
                    self.locations.append(trip.end_loc)
            else:
                trip = Trip(full=trips_to_add)
                if trip.tid in [t.tid for t in self.trips]:
                    trip.tid = max([t.tid for t in self.trips]) + 1
                self.trips.append(trip)
                self.visits = self.visits.append(trip.visits)
        elif isinstance(trips_to_add, Trip):
            prev_tids = [t.tid for t in self.trips]
            if trips_to_add.tid in prev_tids:
                trips_to_add.tid = max(prev_tids) + 1
            trips_to_add.uid = self.uid
            self.trips.append(trips_to_add)
            self.locations.append(trips_to_add.start_loc)
            self.locations.append(trips_to_add.end_loc)

    def add_locations(self, location):
        if isinstance(location, tuple):
            self.locations.append(location)
        if isinstance(location, list):
            if isinstance(location[0], tuple):
                self.locations = self.locations + location
            elif isinstance(location[0], int) and len(location) == 2:
                self.locations.append(tuple(location))

    def add_connections(self, member):
        if isinstance(member, list):
            self.connections = self.connections + member
        else:
            self.connections.append(member)

    def set_id(self, uid):
        self.uid = uid

    def get_trips_df(self):
        trip_df = pd.DataFrame([trip.as_row() for trip in self.trips])
        trip_df['uid'] = self.uid
        return trip_df

    def get_visits_df(self):
        visit_df = pd.DataFrame({'uid': self.uid
                                    , "timestamp": [visit.timestamp for visit in self.visits]
                                    , 'latitude': [visit.latitude for visit in self.visits]
                                    , 'longitude': [visit.longitude for visit in self.visits]
                                    , "nearest_node": [visit.nn_id for visit in self.visits]})

    def to_community(self):
        if self.connections and len(self.connections)>0:
            all_users = self.connections + list(self.uid)
        else:
            all_users = [self.uid]
        print(all_users)
        return Community(users=all_users)

    def __repr__(self):
        return f'ID: {self.uid}; Num_trips = {len(self.trips)}'


def main():
    print('extracting users')
    # all_users = extract_users(full_data)
    test_uid = 146
    test_user = extract_user(full_data,test_uid)
    print(test_user.uid)
    # print(f'extracted {len(all_users)} from data')
    # all_trips = Community(users=all_users).trips_df
    all_trips = test_user.to_community().trips_df
    print(all_trips.head())
if __name__ == "__main__":
    main()
