
import os
import src.consts as consts
import dateutil.parser as parser
import pandas as pd
from src.util_event import build_disease_instance
import numpy as np
from src.util_event import read_events_from_df
from src.st_clustering.st_dbscan import ST_DBSCAN
from collections import Counter

def retrieve_cascades_info_as_dataframe_from_df(df_events, res_col_name, eventId2eventDate, eventId2geonamesId):
    # TODO in the same cluster, multiple events with the same location can exist. How to deal with it ?
    clustering_dict = {}
    geonamesId_dict = {} # we want the cluster to have unique ADM1 zones
    for index, row in df_events.iterrows():
        id = str(row[consts.COL_ID])
        gn_id = eventId2geonamesId[row[consts.COL_ID]]
        c_id = str(row[res_col_name])
        if c_id != "":
            # we want the cluster to have unique ADM1 zones
            if (c_id not in geonamesId_dict) or (gn_id not in geonamesId_dict[c_id]):
                if c_id not in clustering_dict:
                    clustering_dict[c_id] = []
                if c_id not in geonamesId_dict:
                    geonamesId_dict[c_id] = []
                clustering_dict[c_id].append(id)
                geonamesId_dict[c_id].append(gn_id)

    clusters_list = []
    size_list = []
    for key in clustering_dict.keys():
        event_ids_nonordered = np.array(clustering_dict[key])
        event_dates = [eventId2eventDate[int(id)] for id in event_ids_nonordered]
        event_ids_ordered = event_ids_nonordered[np.argsort(event_dates)]
        event_ids_ordered_str = [str(eid) for eid in event_ids_ordered]
        size_list.append(len(event_ids_ordered_str))
        clusters_list.append(",".join(event_ids_ordered_str))
    df_cascades = pd.DataFrame({'cascade': clusters_list, 'size': size_list})

    return df_cascades

# https://github.com/eren-ck/st_dbscan/blob/master/demo/demo.ipynb
# eps1: max spatial distance, from the range [0,1]
# eps2: max temporal distance
def retrieve_cascades_with_st_clustering(df_events, out_cascades_filepath=None, eps1=0.1, eps2=24*10):
    # init
    RES_COL_NAME = "st-clustering_eps1=" + str(eps1) + "_eps2=" + str(eps2)
    df_events[RES_COL_NAME] = ""
    print(df_events.columns)


    #events = read_events_from_df(df_events)

    # normalize the data
    df_events['lng'] = (df_events['lng'] - df_events['lng'].min()) / (
                df_events['lng'].max() - df_events['lng'].min())
    df_events['lat'] = (df_events['lat'] - df_events['lat'].min()) / (
                df_events['lat'].max() - df_events['lat'].min())
    # transform to numpy array
    data = df_events.loc[:, ['timestamp_in_hour', 'lng', 'lat']].values
    #
    # The input data format is: ['temporal_index', 'x', 'y', < optional attributes >]
    st_dbscan = ST_DBSCAN(eps1=eps1, eps2=eps2, min_samples=5)
    st_dbscan.fit(data)
    counter = Counter(st_dbscan.labels)
    #print(counter)
    # -1: outlier
    labels_str = [str(l) if l != -1 else "" for l in st_dbscan.labels]
    df_events[RES_COL_NAME] = labels_str
    df_events = df_events[df_events[RES_COL_NAME] != ""]

    # -----
    eventId2eventDate = dict(zip(df_events['id'], df_events['published_at']))
    eventId2geonamesId = dict(zip(df_events['id'], df_events['geonames_id']))
    df_cascades = retrieve_cascades_info_as_dataframe_from_df(df_events, RES_COL_NAME, eventId2eventDate, eventId2geonamesId)
    # -----

    if out_cascades_filepath is not None:
        df_cascades.to_csv(out_cascades_filepath, sep=";", index=False)

    # df_events.reset_index(drop=True, inplace=True)
    return df_events, df_cascades



def retrieve_single_cascade_from_df(df_events, out_cascades_filepath=None):
    df_events.sort_values(by=['ADM1_geonameid', "timestamp_in_hour"], inplace=True)
    df_events_grouped = df_events.groupby(['ADM1_geonameid']).agg(
                                    {
                                      "timestamp_in_hour": 'first',
                                      'id' : 'first'
    }).reset_index()
    df_events_grouped.sort_values(by=["timestamp_in_hour"], inplace=True)
    print(df_events_grouped)
    cascade_data = df_events_grouped["id"].astype(str).tolist()
    print(cascade_data)
    cascade_str = ','.join(cascade_data)

    df_cascades = pd.DataFrame({'cascade': [cascade_str], 'size': [len(cascade_data)]})
    if out_cascades_filepath is not None:
        df_cascades.to_csv(out_cascades_filepath, sep=";", index=False)
    return df_cascades



def retrieve_flyway_cascades(df_events, out_cascades_filepath=None):
    # ==================================================================
    # CASCADES BY FLYWYAY
    # ==================================================================
    in_map_folder = consts.IN_MAP_SHAPEFILE_FOLDER
    world_map_csv_filepath = os.path.join(in_map_folder, "naturalearth_adm1_with_fixed_geometries_and_flyway.csv")
    map_info = pd.read_csv(world_map_csv_filepath, sep=";", keep_default_na=False)
    map_info["flyway_info"] = map_info["flyway_info"].apply(lambda x: eval(x) if x != '' else [])
    geonamesId2flyway = dict(zip(map_info["gn_id"], map_info["flyway_info"]))
    df_events["flyway_info"] = df_events["ADM1_geonameid"].apply(
        lambda x: geonamesId2flyway[x] if x in geonamesId2flyway else [])
    print(df_events["flyway_info"])

    flyway_list = ['atlantic americas', 'black sea mediterranean', 'central asia',
                   'east africa - west asia', 'east asian - australasian', 'east atlantic',
                   'mississippi americas', 'pacific americas']
    result_list = []
    for flyway_info in flyway_list:
        print(flyway_info)
        df_events["is_same_flyway"] = df_events["flyway_info"].apply(
            lambda x: 1 if flyway_info in x else 0)
        df_events_by_flyway = df_events[df_events["is_same_flyway"] == 1]
        print(df_events_by_flyway.shape)
        del df_events["is_same_flyway"]
        if df_events_by_flyway.shape[0] > 0:
            df_by_flyway = retrieve_single_cascade_from_df(df_events_by_flyway, out_cascades_filepath=None)
            result_list.append(df_by_flyway)
    df_all = pd.concat(result_list)
    if out_cascades_filepath is not None:
        df_all.to_csv(out_cascades_filepath, sep=";", index=False)
    return df_all


# ======================================
# MAIN FUNCTION
# ======================================

def perform_cascade_preprocessing_default(df_events, out_cascades_filepath, min_cascade_size=5, force=False):
    df_events.sort_values(by=['ADM1_geonameid', "timestamp_in_hour"], inplace=True)
    df_events = df_events.groupby(['ADM1_geonameid']).agg(
        {
            "timestamp_in_hour": 'first',
            'id': 'first',
            'lng': 'first',
            'lat': 'first',
            'published_at': 'first',
            'geonames_id': 'first',
        }).reset_index()
    df_events.sort_values(by=["timestamp_in_hour"], inplace=True)

    if df_events.shape[0]>2:
        df_cascades_by_flyway = retrieve_flyway_cascades(df_events, out_cascades_filepath=None)
        df_events_by_serotype, df_cascades_by_st_clustering = retrieve_cascades_with_st_clustering(df_events, out_cascades_filepath=None, eps1=0.1, eps2=24*10)
        df_all = pd.concat([df_cascades_by_flyway, df_cascades_by_st_clustering])
        df_all = df_all[df_all["size"] >= min_cascade_size]
        df_all.to_csv(out_cascades_filepath, sep=";", index=False)


def perform_cascade_preprocessing_by_serotype(df_events, out_cascades_filepath, min_cascade_size=5, force=False):
    df_events["disease_cluster"] = df_events["disease_cluster"].apply(lambda x: eval(x))

    clusters = set()
    for cl in df_events["disease_cluster"].tolist():
        for c in cl:
            clusters.add(c)

    nb_clusters = len(clusters)
    df_all_list = []
    for cluster_id in range(nb_clusters):
        print("--- cluster id:", cluster_id)
        df_events["disease_cluster_"+str(cluster_id)] = df_events["disease_cluster"].apply(lambda x: 1 if cluster_id in x else 0)
        df_events_by_serotype = df_events[df_events["disease_cluster_"+str(cluster_id)] == 1].copy(deep=True)

        df_events_by_serotype.sort_values(by=['ADM1_geonameid', "timestamp_in_hour"], inplace=True)
        df_events_by_serotype = df_events_by_serotype.groupby(['ADM1_geonameid']).agg(
            {
                "timestamp_in_hour": 'first',
                'id': 'first',
                'lng': 'first',
                'lat': 'first',
                'published_at': 'first',
                'geonames_id': 'first',
                "disease_cluster_" + str(cluster_id): 'first'
            }).reset_index()
        df_events_by_serotype.sort_values(by=["timestamp_in_hour"], inplace=True)

        if df_events_by_serotype.shape[0]>2:
            df_cascades_by_flyway = retrieve_flyway_cascades(df_events_by_serotype, out_cascades_filepath=None)
            df_events_by_serotype, df_cascades_by_st_clustering = retrieve_cascades_with_st_clustering(df_events_by_serotype, out_cascades_filepath=None, eps1=0.1, eps2=24*10)
            df_all = pd.concat([df_cascades_by_flyway, df_cascades_by_st_clustering])
            df_all = df_all[df_all["size"] >= min_cascade_size]
            df_all_list.append(df_all)

        del df_events_by_serotype["disease_cluster_" + str(cluster_id)]
        del df_events["disease_cluster_" + str(cluster_id)]

    df_all = pd.concat(df_all_list)
    df_all.to_csv(out_cascades_filepath, sep=";", index=False)

