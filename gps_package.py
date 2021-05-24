import datetime
import pickle
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from itertools import permutations
from geopy import distance as geodist
import osmnx as ox
from osmnx import distance
import string
import math
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.covariance import EllipticEnvelope
from sklearn.mixture import BayesianGaussianMixture
from sklearn.svm import OneClassSVM


# to be replaced by lookups with Nominatim

def load_location(place='Shanghai, China', pickle_graph=True, pickle_file=None):
    if pickle_file:
        pickle_path = pickle_file
    else:
        pickle_path = '_'.join([word.strip() for word in place.split(',')]) + '_graph.p'
    global G
    try:
        print(f'attempting to load pickle from {pickle_path}')
        G = pickle.load(open(pickle_path, 'rb'))
    except:
        print(f'generating graph for place')
        G = ox.graph_from_place(place, network_type='drive', simplify=True, retain_all=False)
    # global nodes, edges
    # nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    print(f'loaded graph for {place}')
    print('loading data')
    global full_data
    full_data = pd.read_excel('shanghai_gps.xlsx', skiprows=6, header=0)
    full_data.columns = ['_'.join(name.lower()
                                  .translate(str.maketrans(' ', ' ', string.punctuation)).split()) for name in
                         full_data.columns]
    if pickle_graph:
        pickle.dump(G, open(pickle_path, 'wb'))
    return full_data, G


def calculate_great_circle_distance(start_latitude, start_longitude, end_latitude, end_longitude):
    pickup = [start_latitude, start_longitude]
    dropoff = [end_latitude, end_longitude]
    dist = geodist.great_circle(pickup, dropoff).km
    return dist


def calculate_manhattan_distance(start_latitude, start_longitude, end_latitude, end_longitude):
    pickup = [start_latitude, start_longitude]
    dropoff_a = [start_latitude, end_longitude]
    dropoff_b = [end_latitude, start_longitude]
    distance_a = geodist.great_circle(pickup, dropoff_a).km
    distance_b = geodist.great_circle(pickup, dropoff_b).km
    return distance_a + distance_b


def calculate_bearing(start_latitude, start_longitude, end_latitude, end_longitude):
    d_lon = end_longitude - start_longitude
    y = math.sin(d_lon) * math.cos(end_latitude)
    x = math.cos(start_latitude) * math.sin(end_latitude) - math.sin(start_latitude) * math.cos(
        end_latitude) * math.cos(d_lon)
    bearing = math.atan2(y, x)
    if bearing < 0:
        bearing += 2 * math.pi
    return bearing


def extract_user(trip_df, user_id):
    filtered_df = trip_df[trip_df.uid == user_id]
    # print(f'adding {len(filtered_df.index)} trips for user {user_id}')
    new_user = User(id=user_id)
    trips = extract_trips(filtered_df)
    new_user.add_trips(trips)
    return new_user


def extract_users(dataframe):
    users = []
    uids = dataframe.uid.unique()
    for i, uid in enumerate(uids):
        new_user = extract_user(dataframe, uid)
        users.append(new_user)
        print(f'added user #{i}; id {uid}')
    print(f'extracted {len(users)} users')
    return users


def extract_trips(dataframe):
    trips = dataframe.apply(lambda x: Trip(full=x), axis=1).tolist()
    return trips


class Visit(object):
    def __init__(self, uid=None, timestamp=None, latitude=None, longitude=None, nn_id=None):
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

    def __repr__(self):
        return str(self.as_row())

    def __add__(self, other):
        if type(other) == type(self):
            return Trip(uid=self.uid
                        , start_time=self.timestamp
                        , end_time=other.timestamp
                        , start_loc=self.location
                        , end_loc=other.location)
        else:
            print('can only form trips between visits')

    def get_nearest_node(self, graph=None):
        if graph:
            local_graph = graph
        else:
            global G
            local_graph = G
        return ox.distance.nearest_nodes(local_graph, self.longitude, self.latitude)

    def as_row(self):
        return {
            'uid': self.uid
            , 'timestamp': self.timestamp
            , 'longitude': self.longitude
            , 'latitude': self.latitude
        }


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
                # print('extracting trip from series')
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
                # print(self.visits)
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
        self.bearing = calculate_bearing(self.start_lat, self.start_lng, self.end_lat, self.end_lng)
        self.mh_dist = calculate_manhattan_distance(self.start_lat, self.start_lng, self.end_lat, self.end_lng)
        self.gc_dist = calculate_great_circle_distance(self.start_lat, self.start_lng, self.end_lat, self.end_lng)

    def __repr__(self):
        return str(self.as_row())

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
            , 'duration_in_seconds': self.duration
            , 'bearing': self.bearing
            , 'manhattan_distance': self.mh_dist
            , 'great_circle_distance': self.gc_dist
            , 'tid': self.tid
        }


class Community(object):  # to hold multiple users' trips and visits
    def __init__(self, users=None, data=None, G=None):
        global full_data
        if users:
            if isinstance(users, list):
                if isinstance(users[0], User):
                    self.users = users
                elif isinstance(users[0], int):
                    self.users = [extract_user(full_data, i) for i in users]
                    # print(self.users[0])
            else:
                print("users must be given as a list of User objects or user IDs")
        elif isinstance(data, pd.DataFrame):
            self.users = extract_users(data)
        else:
            self.users = extract_users(full_data)

        self.uids = list(set([user.uid for user in self.users]))
        self.trips = [trip for trips in [user.trips for user in self.users] for trip in trips]
        self.visits = [visit for visits in [trip.visits for trip in self.trips] for visit in visits]

    def get_trips_df(self):
        return pd.concat([user.get_trips_df() for user in self.users])

    def get_visits_df(self):
        return pd.concat([user.get_visits_df() for user in self.users])

    def extract_users(self, attribute='uid', ids=None, range=None):
        if attribute == 'uid':
            if isinstance(ids, int):
                return [user for user in self.users if user.uid == ids]
            if isinstance(ids, list):
                return [user for user in self.users if user.uid in ids]

    def add_visit(self, user_id=None, timestamp=None, latitude=None, longitude=None):
        if user_id in self.uids:
            user = [user for user in self.users if user.uid == user_id][0]
            new_visit = Visit(uid=user_id, timestamp=timestamp, latitude=latitude, longitude=longitude)
            user.add_visit(new_visit)

    def transform_time(self, X=None, data_type='trips', inplace=False):
        if data_type == 'trips':
            X = self.get_trips_df().copy()
            X['start_datetime'] = pd.to_datetime(X['start_time_in_seconds'], unit='s')
            X['end_datetime'] = pd.to_datetime(X['end_time_in_seconds'], unit='s')
            X['start_month'] = X['start_datetime'].dt.month
            X['start_day'] = X['start_datetime'].dt.day
            X['start_day_of_week'] = X['start_datetime'].dt.dayofweek
            X['start_hour_of_day'] = X['start_datetime'].dt.hour
            X['start_minute_of_hour'] = X['start_datetime'].dt.minute
            X['end_month'] = X['end_datetime'].dt.month
            X['end_day'] = X['end_datetime'].dt.day
            X['end_day_of_week'] = X['end_datetime'].dt.dayofweek
            X['end_hour_of_day'] = X['end_datetime'].dt.hour
            X['end_minute_of_hour'] = X['end_datetime'].dt.minute
            if inplace:
                X.drop(['start_time_in_seconds', 'end_time_in_seconds'], axis=1, inplace=True)
                self.train = X
        elif data_type == 'visits':
            X = self.get_visits_df.copy()
            X['timestamp_dt'] = pd.to_datetime(X['timestamp'], unit='s')
            X['month'] = X['timestamp_dt'].dt.month
            X['day'] = X['timestamp_dt'].dt.day
            X['day_of_week'] = X['timestamp_dt'].dt.dayofweek
            X['hour_of_day'] = X['timestamp_dt'].dt.hour
            X['minute_of_hour'] = X['timestamp_dt'].dt.minute
            if inplace:
                X.drop(['timestamp'], axis=1, inplace=True)
                self.train = X
        return X


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
                self.connections = [extract_user(full_data, id) for id in connected_users]
            elif isinstance(connected_users[0], User):
                self.connections = connected_users
        else:
            # print('no connections')
            self.connections = []

    def add_trips(self, trips_to_add):
        # print(type(trips_to_add))
        if isinstance(trips_to_add, list):
            # print('trips in list')
            if isinstance(trips_to_add[0], Trip):
                # print('trips in list of Trip objects')
                for trip in trips_to_add:
                    prev_tids = [t.tid for t in self.trips]
                    if trip.tid in prev_tids:
                        trip.tid = int(max(prev_tids) + 1)
                    trip.uid = self.uid
                    self.trips.append(trip)
                    for visit in trip.visits:
                        self.visits.append(visit)
                    self.locations.append(trip.start_loc)
                    self.locations.append(trip.end_loc)
            else:
                # print('trip as list of attributes')
                trip = Trip(full=trips_to_add)
                if trip.tid in [t.tid for t in self.trips]:
                    trip.tid = int(max([t.tid for t in self.trips]) + 1)
                self.trips.append(trip)
                for visit in trip.visits:
                    self.visits.append(visit)
        elif isinstance(trips_to_add, Trip):
            # print('single Trip object passed')
            prev_tids = [t.tid for t in self.trips]
            if trips_to_add.tid in prev_tids:
                trips_to_add.tid = int(max(prev_tids) + 1)
            trips_to_add.uid = self.uid
            self.trips.append(trips_to_add)
            for visit in trips_to_add.visits:
                self.visits.append(visit)
            self.locations.append(trips_to_add.start_loc)
            self.locations.append(trips_to_add.end_loc)

    def add_visit(self, visit=None, location=None, latitude=None, longitude=None, timestamp=None):
        new_visit = None
        if visit and isinstance(visit, Visit):
            new_visit = visit
        else:
            if not timestamp:
                dt_time = datetime.datetime.now()
                timestamp = 10000 * dt_time.year + 100 * dt_time.month + dt_time.day
            if isinstance(location, tuple):
                self.locations.append(location)
                new_visit = Visit(self.uid, latitude=location[0], longitude=location[1], timestamp=timestamp)
            elif isinstance(location, list):
                if isinstance(location[0], tuple):
                    self.locations = self.locations + location
                    new_visit = Visit(self.uid, latitude=location[0][0], longitude=location[0][1], timestamp=timestamp)
                elif isinstance(location[0], int) and len(location) == 2:
                    self.locations.append(tuple(location))
                    new_visit = Visit(self.uid, latitude=location[0], longitude=location[1], timestamp=timestamp)
            elif latitude and longitude:
                new_visit = Visit(self.uid, timestamp=timestamp, latitude=latitude, longitude=longitude)
        prev_vids = [visit.vid for visit in self.visits]
        new_visit.vid = max(prev_vids) + 1
        self.visits.append(new_visit)

    def add_connections(self, member):
        if isinstance(member, list):
            self.connections = self.connections + member
            # print(self.connections)
        else:
            self.connections.append(member)

    def get_trips_df(self):
        trip_df = pd.DataFrame([trip.as_row() for trip in self.trips])
        trip_df['uid'] = self.uid
        return trip_df

    def get_visits_df(self):
        # visit_df = pd.DataFrame([visit.as_row() for visit in self.visits])
        global G
        visit_df = pd.DataFrame({'uid': self.uid
                                    , "timestamp": [visit.timestamp for visit in self.visits]
                                    , 'latitude': [visit.latitude for visit in self.visits]
                                    , 'longitude': [visit.longitude for visit in self.visits]
                                    , "nearest_node": [visit.get_nearest_node(graph=G) for visit in self.visits]})
        # visit_df['uid'] = self.uid
        visit_df['vid'] = visit_df.sort_values(by=['uid', 'timestamp']).groupby('uid', sort=False).cumcount() + 1
        return visit_df

    def to_community(self):
        if self.connections and len(self.connections) > 0:
            print(len(self.connections))
            all_users = self.connections + [self.uid]
            print(all_users)
        else:
            all_users = [self.uid]
        print(all_users)
        return Community(users=all_users)

    def __repr__(self):
        return f'ID: {self.uid}; Num_trips = {len(self.trips)}'


class gps_anomaly_detector(object):
    def __init__(self, data=None, test_data=None, kind='all'):
        if isinstance(data, pd.DataFrame):
            self.train = data
        else:
            self.train = None
        if isinstance(test_data, pd.DataFrame):
            self.test = test_data
        self.valid = None
        self.scaler = None
        self.model = None
        self.models = None
        self.kind = kind
        self.type = None
        if data and isinstance(data, Community):
            self.trip_data = data.get_trips_df()
            self.visit_data = data.get_visits_df()
        elif data and isinstance(data, pd.DataFrame):
            community = Community(data=data)
            self.trip_data = community.get_trips_df()
            self.visit_data = community.get_visits_df()
        else:
            self.trip_data = None
            self.visit_data = None
        if not self.train:
            if self.kind == 'trip' and isinstance(self.trip_data, pd.DataFrame):
                self.train = self.trip_data
            elif self.kind == 'visit' and isinstance(self.visit_data, pd.DataFrame):
                self.train = self.visit_data
            elif self.kind == 'all' and isinstance(self.visit_data, pd.DataFrame) and isinstance(self.trip_data,
                                                                                                 pd.DataFrame):
                self.train = [self.trip_data, self.visit_data]
            else:
                self.train = None

    def split(self, test_size=0.3, valid_size=0):
        if self.kind == 'trip':
            train, df_test = train_test_split(self.trip_data, test_size=test_size)
            if valid_size > 0:
                df_train, df_valid = train_test_split(train, test_size=valid_size)
                self.valid = df_valid
            else:
                df_train = train
            self.train = df_train
        elif self.kind == 'visit':
            train, df_test = train_test_split(self.visit_data, test_size=test_size)
            if valid_size > 0:
                df_train, df_valid = train_test_split(train, test_size=valid_size)
                self.valid = df_valid
            else:
                df_train = train
            self.train = df_train
        else:
            trips_train, df_trips_test = train_test_split(self.trip_data, test_size=test_size)
            visits_train, df_visits_test = train_test_split(self.visit_data, test_size=test_size)
            self.test = [df_trips_test, df_visits_test]
            if valid_size > 0:
                df_trips_train, df_trips_valid = train_test_split(trips_train, test_size=valid_size)
                df_visits_train, df_visits_valid = train_test_split(visits_train, test_size=valid_size)
                self.valid = [df_trips_valid, df_visits_valid]
            else:
                df_trips_train, df_visits_train = trips_train, visits_train
            self.train = [df_trips_train, df_visits_train]

    def standardize(self, X=None, columns=None, kind='visits', pca=True):
        if not X:
            if isinstance(self.train, list):
                X1 = self.train[0].copy()
                X2 = self.train[1].copy()
                if columns:
                    X1_cols, X2_cols = [], []
                    for column in columns:
                        if column in X1.columns:
                            X1_cols.append(column)
                        elif column in X2.columns:
                            X2_cols.append(column)


            else:
                X = self.train
        if columns:
            X_normalized = X.copy()
            X_normalized[columns] = pd.DataFrame(normalize(X[columns]))
            if not self.scaler:
                self.scaler = StandardScaler()
                self.scaler.fit(X_normalized[columns])
        else:
            X_normalized = pd.DataFrame(normalize(X))
            if not self.scaler:
                self.scaler = StandardScaler()
                self.scaler.fit(X_normalized)
        X_standardized = pd.DataFrame(self.scaler.transform(X_normalized))
        if pca:
            X_pc = PCA(n_components=2).fit_transform(X_standardized)
            self.X_standardized = pd.DataFrame(X_pc, columns=['P1', 'P2'])
            return self.X_standardized
        else:
            return X_standardized

    def fit(self, train=None, test=None, model_type='ensemble'):
        self.type = model_type
        if not train:
            train = self.train
        if not isinstance(train, list):
            train = [train]
        models = {}
        for i in range(len(train)):
            if model_type.lower() in ['ensemble', 'if', 'isolationforest']:
                if_model = IsolationForest()
                if_model.fit(train[i])
                if model_type == 'ensemble':
                    models[f'isolation_forest{i}'] = if_model
                else:
                    self.model = if_model
            if model_type.lower() in ['ensemble', 'ec', 'empiricalcovariance']:
                ec_model = EllipticEnvelope(support_fraction=1., contamination=0.002)
                ec_model.fit(train[i])
                if model_type == 'ensemble':
                    models[f'elliptic_envelope{i}'] = ec_model
                else:
                    self.model = ec_model
            if model_type.lower() in ['ensemble', 'bgm', 'bayesiangaussian']:
                bgm_model = BayesianGaussianMixture(n_components=len(self.train.uid.unique()))
                bgm_model.fit(train[i])
                if model_type == 'ensemble':
                    models[f'bayesian_gaussian{i}'] = bgm_model
                else:
                    self.model = bgm_model
            if model_type.lower() in ['ensemble', 'ocs', 'oneclasssvm']:
                ocs_model = OneClassSVM(nu=0.25, gamma=0.35)
                ocs_model.fit(train[i])
                if model_type == 'ensemble':
                    models[f'one_class_svm{i}'] = ocs_model
                else:
                    self.model = ocs_model
            if model_type.lower() == 'ensemble':
                self.model = models

    def predict(self, X=None, kind='visit'):
        if not X:
            X = self.train
        if self.type == 'ensemble':
            model_preds = {}
            for name, model in self.model.items():
                model_preds[name] = model.predict(X)
                # y_pred_test = model.predict(X_test)
                # n_error_train = y_pred_train[y_pred_train == -1].size
                # n_error_test = y_pred_test[y_pred_test == -1].size
            pred_df = pd.DataFrame(model_preds)
            results_df = pred_df.copy()
            results_df['max'] = pred_df.max(axis=1)
            results_df['min'] = pred_df.min(axis=1)
            results_df['avg'] = pred_df.mean(axis=1)
            # results_df['mode'] = pred_df.apply(lambda x: x.mode())
            return results_df

        elif len(self.model) == 1:
            return self.model.predict(X)


def main():
    global G, full_data
    full_data, G = load_location(pickle_graph=False) 
    glob_comm = Community(users=extract_users(full_data))
    # G, nodes, edges = pickle.load(open('shanghai_graph.p', 'rb'))
    # print('extracting users')
    # all_users = extract_users(full_data)
    # test_uid = 146
    # test_user = extract_user(full_data, test_uid)
    # print(test_user.trips)
    # print(test_user.visits)
    # filtered_df = full_data[full_data.uid == test_uid]
    # trips = extract_trips(filtered_df)
    # print(type(trips[0]))
    # test_user.add_trips(trips)
    # print(len(test_user.trips))
    # test_user.add_connections(member=[180, 211, 196])
    # print(test_user.uid)
    # print(test_user.connections)
    # print(f'extracted {len(all_users)} from data')
    # full_community = pickle.load(open("global_community.p", "rb"))
    # all_trips = Community(users=all_users).trips_df
    # test_comm = test_user.to_community()
    # all_trips = test_comm.get_trips_df()
    # print(all_trips.head())
    # all_trips = full_community.trips_df
    # print(len(glob_comm.visits))
    # all_visits = test_user.get_visits_df()
    # print(all_visits.head())
    # print(all_visits.info())
    # print(all_trips.sample(n=10, random_state=3))
    # print(all_visits.sample(n=10, random_state=3))

    ad = gps_anomaly_detector(glob_comm, kind='visit')
    print(ad.train.describe())
    ad.fit()
    pickle.dump(ad, open("full_comm_ad.p", "wb"))
    results = ad.predict()
    print(results.describe())

if __name__ == "__main__":
    main()
