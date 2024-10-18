

import numpy as np
import pandas as pd
from src.util_event import read_events_from_df
import src.consts as consts
import os
import dateutil.parser as parser

def create_cascades_nodes_matrix(events_filepath, cascades_filepath, df_world_map_info, out_folder):


    df_events = pd.read_csv(events_filepath, sep=";", keep_default_na=False)
    df_events[consts.COL_PUBLISHED_TIME] = df_events[consts.COL_PUBLISHED_TIME].apply(lambda x: parser.parse(x))
    df_events["hierarchy_data"] = df_events["hierarchy_data"].apply(lambda x: eval(x))
    df_events["timestamp"] = df_events["timestamp"].apply(lambda x: x/24)
    df_events["timestamp"] = df_events["timestamp"].apply(lambda x: x/6) # to be from the range [0,180]
    eventId2timestamp = dict(zip(df_events["id"],df_events["timestamp"]))


    # world_map_csv_filepath = os.path.join(in_map_folder, "world", "ne_10m_admin_1_states_provinces",
    #                                       "naturalearth_adm1.csv")
    # df_world_map_info = pd.read_csv(world_map_csv_filepath, usecols=["gn_id", "name", "admin", "lon", "lat"], sep=";",
    #                                 keep_default_na=False)
    # # df_world_map_info_nz = df_world_map_info[df_world_map_info["gn_id"] != -1]
    df_world_map_info_nz = df_world_map_info
    N = df_world_map_info_nz.shape[0]
    id_list = df_world_map_info_nz["gn_id"].to_numpy().flatten()
    #id2geonameid = dict(zip(range(N), id_list))
    geonameid2id = dict(zip(id_list, range(N)))
    df_events["nodeIdx"] = df_events["ADM1_geonameid"].apply(lambda x: geonameid2id[x])
    eventId2nodeIdx = dict(zip(df_events["id"],df_events["nodeIdx"]))
    #nodeIdx2eventId = dict(zip(df_events["nodeIdx"],df_events["id"]))

    df_cascades = pd.read_csv(cascades_filepath, sep=";", keep_default_na=False)
    cascade_list = df_cascades["cascade"].to_list()
    nb_cascades = df_cascades.shape[0]

    T = np.full((nb_cascades, N), np.inf)

    C = len(cascade_list)
    for i in range(C):
        event_ids_str = cascade_list[i]
        event_ids = [int(id) for id in event_ids_str.split(",")]
        for eid in event_ids:
            T[i,eventId2nodeIdx[eid]] = eventId2timestamp[eid]
        min_t = np.min(T[i:])
        T[i, :] -= min_t # at least one node should have an infection time at t=0
        #a = T[i, :]
        #print(i, np.max(a[a != np.inf]))


    #print(T)
    cascades_time_savefile = os.path.join(out_folder, "cascades_time.npy")
    np.save(cascades_time_savefile, T)



if __name__ == '__main__':
    print('Starting')
    in_folder = os.path.join(consts.OUT_FOLDER, "preprocessing")
    preprocessed_events_filepath = os.path.join(in_folder,
                                                "processed_empres-i_events_updated.csv")  # only 2021 data
    cascades_filepath = os.path.join(in_folder, "cascades", "cascades.csv")

    in_map_folder = consts.IN_MAP_SHAPEFILE_FOLDER
    world_map_csv_filepath = os.path.join(in_map_folder, "world", "ne_10m_admin_1_states_provinces",
                                          "naturalearth_adm1.csv")
    df_world_map_info = pd.read_csv(world_map_csv_filepath, usecols=["gn_id", "name", "admin", "lon", "lat"], sep=";",
                                    keep_default_na=False)

    out_folder = os.path.join(consts.OUT_FOLDER, "preprocessing", "FIM")

    try:
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
    except OSError as err:
        print(err)

    create_cascades_nodes_matrix(preprocessed_events_filepath, cascades_filepath, df_world_map_info, out_folder)
