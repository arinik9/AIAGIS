
import pandas as pd
import numpy as np
import os
import csv
import src.consts as consts

from src.util_gis import haversine


# ======================================
# spatial preprocessing
# 1) convert all locations at country or ADM1 level & discard all entries with are only at country level
# 2) read the shapefile and remove all locations, whose id is not in the map, i.e. failed geonameids
#
# Remark: Note that we do not perform any preprocessing to remove duplicated locations ids, because
# it is handled in the part "Construct data tensors from nonzero cascades". Remember the assumption
# in this method is that if a place is infected, it can be reinfected again later.
# ======================================
def retrieve_ADM1_geoname_ids(df_events):
    df_events["ADM1_geonameid"] = df_events["hierarchy_data"].apply(lambda x: x[1]) # get the id at ADM1 level
    return df_events


# we use geographic zones at ADM1 level, so if there is any entry which is only at country level, discard it
def discard_country_level_geoname_ids(df_events):
    df_events["hier_level"] = df_events["hierarchy_data"].apply(lambda x: len(x))
    df_events = df_events[df_events["hier_level"] > 1]
    del df_events["hier_level"]
    return df_events

def discard_failed_ADM1_geoname_ids(df_events, df_world_map_info):
    geonames_id_list_in_world_map = df_world_map_info["gn_id"].to_list()
    print(df_events)
    print(geonames_id_list_in_world_map[:5])
    print(df_events["ADM1_geonameid"].to_list()[:5])
    print("before", df_events.shape)
    df_events_with_existing_ids = df_events[df_events["ADM1_geonameid"].isin(geonames_id_list_in_world_map)]
    print("after", df_events_with_existing_ids.shape)
    return df_events_with_existing_ids


# ======================================
# create the distance matrix
# - calculate the distance matrix between each pair of ADM1 entities (retrieve it from the map)
# ======================================
def create_distance_matrix_from_map(df_world_map_info, dist_matrix_filepath):
    id_list = df_world_map_info["gn_id"].to_numpy().flatten()
    print(id_list[:10])
    N = len(id_list)

    D = np.full(shape=(N,N), fill_value=np.nan)
    centroid_list = list(zip(df_world_map_info["lon"], df_world_map_info["lat"]))
    print(len(centroid_list))
    for i in range(N): #for index, row in df_map.iterrows():
        lon1 = centroid_list[i][0]
        lat1 = centroid_list[i][1]
        D[i, i] = 0.0
        for j in range(N):
            if i<j:
               lon2 = centroid_list[j][0]
               lat2 = centroid_list[j][1]
               dist = haversine(lon1, lat1, lon2, lat2)
               D[i,j] = dist
               D[j,i] = dist

    D2 = np.trunc(1000 * D) / 1000
    df = pd.DataFrame(D2)
    df.index = id_list
    df.columns = id_list
    df.to_csv(dist_matrix_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    return df

def create_distance_matrix_from_events(df_events, df_dist_matrix_from_map, out_dist_matrix_filepath):
    id_list = df_events["id"].to_numpy().flatten()
    N = len(id_list)

    D = np.full(shape=(N,N), fill_value=np.nan)
    for i, row1 in df_events.iterrows():
        print(i, "/", N)
        gn_id1 = row1["ADM1_geonameid"]
        D[i, i] = 0.0
        for j, row2 in df_events.iterrows():
           if i<j:
               gn_id2 = row2["ADM1_geonameid"]
               D[i, j] = df_dist_matrix_from_map.loc[gn_id1, gn_id2]
               D[j, i] = D[i, j]

    df = pd.DataFrame(D)
    df.index = id_list
    df.columns = id_list
    df.to_csv(out_dist_matrix_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)



def check_duplicates_in_map(df_world_map_info):
    id_list = df_world_map_info["gn_id"].to_numpy().flatten()
    print(id_list[:10])
    N = len(id_list)
    print("N:", N)
    res = np.unique(id_list, return_counts=True)
    print("N (unique):", res)
    # freqs = res[1]
    # idx = np.where(freqs == 2)
    # print(res[0][idx])
    # id2geonameid = dict(zip(range(N), id_list))
    # geonameid2id = dict(zip(id_list, range(N)))
    # geonameid2name = dict(zip(id_list, df_map_nz["name"]))
    # geonameid2country = dict(zip(id_list, df_map_nz["admin"]))
    # geonameid2lon = dict(zip(id_list, df_map_nz["lon"]))
    # geonameid2lat = dict(zip(id_list, df_map_nz["lat"]))
    # # print(id2geoid)
    # # print("---")
    # # print(geoid2id)
    # # print(np.where(id_list==1844176))
    # # print(geonameid2id['1844176'])




# ======================================
# MAIN FUNCTION
# ======================================
def perform_spatial_preprocessing(df_events, in_map_folder, output_dist_matrix_filepath, force=False):
    # read the world map
    world_map_csv_filepath = os.path.join(in_map_folder, "world", "ne_10m_admin_1_states_provinces", "naturalearth_adm1.csv")
    df_world_map_info = pd.read_csv(world_map_csv_filepath, usecols=["gn_id", "name", "admin", "lon", "lat"], sep=";",
                         keep_default_na=False)
    #df_world_map_info_nz = df_world_map_info[df_world_map_info["gn_id"] != -1]

    # TODO: remove some entries, if needed
    check_duplicates_in_map(df_world_map_info)

    if (not os.path.exists(output_dist_matrix_filepath)) or force:
        # prepare distance matrix and write into the file
        create_distance_matrix_from_map(df_world_map_info, output_dist_matrix_filepath)

    # prepare spatial entities
    df_events = discard_country_level_geoname_ids(df_events)
    print(df_events.shape)
    df_events = retrieve_ADM1_geoname_ids(df_events)
    print(df_events.shape)
    df_events = discard_failed_ADM1_geoname_ids(df_events, df_world_map_info)
    df_events.reset_index(drop=True, inplace=True)
    print(df_events.shape)
    return df_events



if __name__ == '__main__':
    print('Starting')
    in_map_folder = consts.IN_MAP_SHAPEFILE_FOLDER
    world_map_csv_filepath = os.path.join(in_map_folder, "world", "ne_10m_admin_1_states_provinces", "naturalearth_adm1.csv")
    df_world_map_info = pd.read_csv(world_map_csv_filepath, usecols=["gn_id", "name", "admin", "lon", "lat"], sep=";",
                         keep_default_na=False)
    #df_world_map_info_nz = df_world_map_info[df_world_map_info["gn_id"] != -1]
    df_world_map_info_nz = df_world_map_info

    output_preprocessing_folder = os.path.join(consts.OUT_FOLDER, "preprocessing")
    events_filepath = os.path.join(output_preprocessing_folder, "processed_empres-i_events_updated.csv") # only 2021 data
    df_events_prep_upd = pd.read_csv(events_filepath, sep=";", keep_default_na=False)

    output_dist_matrix_filepath = os.path.join(output_preprocessing_folder, "spatial_dist_matrix_from_map.csv")
    df_dist_matrix_from_map= create_distance_matrix_from_map(df_world_map_info_nz, output_dist_matrix_filepath)

    output_dist_matrix_filepath = os.path.join(output_preprocessing_folder, "spatial_dist_matrix_from_events.csv")
    create_distance_matrix_from_events(df_events_prep_upd, df_dist_matrix_from_map, output_dist_matrix_filepath)