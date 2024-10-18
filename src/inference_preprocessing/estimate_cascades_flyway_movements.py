import os
import pandas as pd
import geopandas as gpd
import numpy as np
import src.consts as consts
import dateutil.parser as parser
import datetime
from src.postprocessing.complete_genome_data_from_existing_data import identify_k_neighbors_indices
from src.util_event import build_disease_instance
from src.inference.MultiC.multic_s_NEW import construct_digraph_from_inferred_edgelist_single_layer_phase


def create_cascade_graph(df_world_map_info, cascade_str, eventId2geonamesId, out_graphml_filepath):
    geonamesId2lng = dict(zip(df_world_map_info['gn_id'], df_world_map_info['lon']))
    geonamesId2lat = dict(zip(df_world_map_info['gn_id'], df_world_map_info['lat']))

    cascade_list = [int(x) for x in cascade_str.split(",")]

    source_id_list = []
    target_id_list = []
    for i in range(len(cascade_list)-1):
        source_id_list.append(cascade_list[i])
        target_id_list.append(cascade_list[i+1])

    df_edgelist = pd.DataFrame({"source_eventId": source_id_list, "target_eventId": target_id_list})
    df_edgelist["weight_alpha"] = 1
    df_edgelist["source_geonameId"] = df_edgelist["source_eventId"].apply(lambda x: eventId2geonamesId[x])
    df_edgelist["target_geonameId"] = df_edgelist["target_eventId"].apply(lambda x: eventId2geonamesId[x])

    df_edgelist["source_lat"] = df_edgelist["source_geonameId"].apply(lambda x: geonamesId2lat[x])
    df_edgelist["source_lon"] = df_edgelist["source_geonameId"].apply(lambda x: geonamesId2lng[x])
    df_edgelist["target_lat"] = df_edgelist["target_geonameId"].apply(lambda x: geonamesId2lat[x])
    df_edgelist["target_lon"] = df_edgelist["target_geonameId"].apply(lambda x: geonamesId2lng[x])

    construct_digraph_from_inferred_edgelist_single_layer_phase(out_graphml_filepath, df_edgelist)




def find_nearby_events(df_events, event_id, df_spatial_dist_matrix, df_temporal_dist_matrix, dist_tresh, temp_tresh):
    event_id_list = df_events["id"].tolist()
    event_id_list = [str(i) for i in event_id_list]
    #print(event_id)
    spatial_dist_row = df_spatial_dist_matrix.loc[event_id, event_id_list].to_numpy()
    s_dist_values, s_dist_indices = identify_k_neighbors_indices(spatial_dist_row, tresh=dist_tresh, k=None)
    #print("spatial")
    #print(s_dist_values, s_dist_indices)
    temp_dist_row = df_temporal_dist_matrix.loc[event_id, event_id_list].to_numpy()
    temp_dist_sub = temp_dist_row[s_dist_indices]
    t_dist_values, t_dist_indices = identify_k_neighbors_indices(temp_dist_sub, tresh=temp_tresh, k=None)
    #print("temporal")
    #print(t_dist_values, t_dist_indices)
    overall_indices = s_dist_indices[t_dist_indices]
    overall_s_values = spatial_dist_row[overall_indices]
    overall_t_values = temp_dist_row[overall_indices]
    #print(overall_s_values, "!!", overall_t_values)
    df_events_nearby = df_events.iloc[overall_indices, :]
    #print("nearby events")
    #print(df_events_nearby.loc[:,["loc_name", "loc_country_code", "published_at"]])
    return(df_events_nearby)

# direction: either "north" or "south"
# spring_starting_date
# winter_starting_date
# incubation_period
# infection_period
# TODO: handle the hemisphere information
def trace_movement_path(df_events, starting_event_id, path_size, end_date, starting_direction,
                        df_spatial_dist_matrix, df_temporal_dist_matrix,
                        migration_returning_back_date, infection_rate,
                        init_spatial_dist_nearby=500, init_temp_dist_nearby=5, max_iter=50,
                        is_flyway_movement=False
                        ):
    df_events_copy = df_events.copy(deep=True)
    event_id = starting_event_id
    gn_id = df_events_copy.loc[df_events_copy["id"] == event_id, "ADM1_geonameid"].tolist()[0]
    lat = df_events_copy.loc[df_events_copy["id"] == event_id, "lat"].tolist()[0]
    p_date = df_events_copy.loc[df_events_copy["id"] == event_id, consts.COL_PUBLISHED_TIME] + datetime.timedelta(days=1)
    starting_date = p_date.tolist()[0]
    starting_flyways = df_events_copy.loc[df_events_copy["id"] == event_id, "flyway_info"].tolist()[0]
    direction = starting_direction

    path = [event_id]

    # direction = None
    # if starting_date < migration_returning_back_date:
    #     direction = "north"
    # if (starting_date >= spring_starting_date) and (starting_date < winter_starting_date):
    #     direction = "south"
    # elif (starting_date >= winter_starting_date):
    #     direction = "north"


    # event_info = df_events[df_events["id"] == event_id]
    if is_flyway_movement:
        source_flyways = starting_flyways
        if len(source_flyways) > 1:
            print("error for flyways")
            # sdf()
        source_flyway = source_flyways[0]
        df_events_copy["is_same_flyway"] = df_events_copy["flyway_info"].apply(
            lambda x: 1 if source_flyway in x else 0)
        df_events_copy = df_events_copy[df_events_copy["is_same_flyway"] == 1]
        #print("current size:", df_events_copy.shape[0], "/", df_events.shape[0])

    i = 0
    found_place = True
    change_migration_direction = False
    while len(path)<path_size and i<max_iter and starting_date<end_date and found_place:
        #time.sleep(1)
        #print("--- iter no", i)
        #print(df_events_copy.loc[df_events_copy["id"] == event_id, ["loc_name", "loc_country_code", "published_at"]])
        df_events_copy = df_events_copy[df_events_copy[consts.COL_PUBLISHED_TIME] >= starting_date]
        df_events_copy = df_events_copy[df_events_copy["ADM1_geonameid"] != gn_id]
        #print("shape", df_events.shape)
        #print(df_events.loc[df_events["id"] == event_id, ["loc_name", "loc_country_code", "published_at"]])

        if is_flyway_movement:
            if not change_migration_direction and starting_date < migration_returning_back_date:
                direction = starting_direction
            elif not change_migration_direction and (starting_date >= migration_returning_back_date):
                if direction == "south":
                    direction = "north"
                else:
                    direction = "south"
                change_migration_direction = True
            #print("direction:", direction)

        spatial_dist_nearby = init_spatial_dist_nearby
        temp_dist_nearby = init_temp_dist_nearby
        #print(spatial_dist_nearby, temp_dist_nearby)
        found_place = False
        for j in range(10): # // we try 10 times to find a nearby geo zone >> each time of failure, we increase spatial and temp distances
            nearby_events = find_nearby_events(df_events_copy, event_id,
                                               df_spatial_dist_matrix, df_temporal_dist_matrix,
                                               spatial_dist_nearby, temp_dist_nearby)
            #print(lat, "--", nearby_events["lat"].tolist())
            if is_flyway_movement:
                if direction == "north": # north direction
                    nearby_events = nearby_events[nearby_events["lat"] >= lat]
                else:
                    nearby_events = nearby_events[nearby_events["lat"] <= lat]
                #if change_direction:
                #    print(nearby_events.loc[:,["loc_name", "loc_country_code", "lat", "published_at"]])
            #print("nb nearby events", nearby_events.shape[0])
            if nearby_events.shape[0]>0 and np.random.rand(1)<infection_rate:
                indx = np.random.choice(list(nearby_events.index), 1)[0]
                event_id = nearby_events.loc[indx, "id"]
                #print("new event id", event_id)
                event_date = nearby_events.loc[indx, consts.COL_PUBLISHED_TIME]
                gn_id = nearby_events.loc[indx, "ADM1_geonameid"]
                lat = nearby_events.loc[indx, "lat"]
                path.append(event_id)
                # update df_events_copy
                starting_date = event_date + datetime.timedelta(days=1)
                #print(starting_date)
                #print(gn_id, event_id, lat)
                found_place = True
                #print("found")
                break
            spatial_dist_nearby += 250  # in km
            temp_dist_nearby += 5  # in days
        i += 1

        # ------- TODO
        # if starting_date < spring_starting_date:
        #     direction = "north"
        # if (starting_date >= spring_starting_date) and (starting_date < winter_starting_date):
        #     direction = "south"
        # elif (starting_date >= winter_starting_date):
        #     direction = "north"
        # ------- TODO

        # TODO maybe verify if the current date is close to spring or winter ?
        if is_flyway_movement:
            if not found_place:
                found_place = True
                if not change_migration_direction:
                    change_migration_direction = True
                    if direction == "south":
                        direction = "north"
                    else:
                        direction = "south"

    print("found_place:", found_place, " - i:", i, "out of", max_iter, " - path_size:", len(path), "out of", path_size)
    #print(path)
    return path


def trace_all_movement_paths_by_serotype(df_events_by_serotype, df_starting_events,
                             df_spatial_dist_matrix, df_temporal_dist_matrix,
                             migration_returning_back_date, final_end_date, random_seed=0):
    # take only the events located in the north hemisphere
    #df_starting_events_in_north_hemisphere = df_starting_events[df_starting_events["lat"] > 20.0]
    #df_starting_events_in_south_hemisphere = df_starting_events[df_starting_events["lat"] <= 20.0]

    path_list = []
    np.random.seed(random_seed)
    print("There are", df_starting_events.shape[0], "distinct starting locations")
    for i, row in df_starting_events.iterrows():
        event_id = row["id"]
        init_spatial_dist_nearby = 500
        init_temp_dist_nearby = 1

        # nb_paths = np.random.randint(10, 30, 1)[0]
        nb_paths = 10
        print("----- starting event id:", event_id, "with nb_paths:", nb_paths)
        # path_list = []
        # end_date = spring_starting_date
        #starting_direction = "south"
        for j in range(nb_paths):  # j.th path
            for infection_rate in [0.2,0.4]:
                for is_flyway_movement in [False]: #[False, True]
                    for end_date in [final_end_date]: # [migration_returning_back_date, final_end_date]
                        for starting_direction in [None]: # ["north", "south"]:
                            path_size = path_size_list = np.random.randint(20, 100, nb_paths)[0]  # from [20,100] and 50 is excluded

                            # since the starting date is in the wintering period, we want to trace a movement up to the beginning of the spring period
                            path = trace_movement_path(df_events_by_serotype, event_id, path_size, end_date, starting_direction,
                                                       df_spatial_dist_matrix, df_temporal_dist_matrix,
                                                       migration_returning_back_date, infection_rate,
                                                       init_spatial_dist_nearby, init_temp_dist_nearby,
                                                       50, is_flyway_movement)
                            print("path_size", path_size)
                            print(path, "with size:", len(path))
                            if path not in path_list:
                                path_list.append(path)

        print("total size of all paths:", len(path_list))
    return path_list


def trace_all_movement_paths_with_serotype_info(df_events, nb_clusters,
                             df_spatial_dist_matrix, df_temporal_dist_matrix,
                             migration_returning_back_date, final_end_date, min_cascade_size, random_seed=0):

    for cluster_id in range(nb_clusters):
        df_events["disease_cluster_"+str(cluster_id)] = df_events["disease_cluster"].apply(lambda x: 1 if cluster_id in x else 0)
        df_events_by_serotype = df_events[df_events["disease_cluster_"+str(cluster_id)] == 1].copy(deep=True)
        del df_events_by_serotype["disease_cluster_"+str(cluster_id)]
        del df_events["disease_cluster_" + str(cluster_id)]

        serotype = None
        for i in df_events_by_serotype["disease"].tolist():
            serotype = build_disease_instance(i).get_disease_data()["serotype"]
            if serotype != "unknown serotype":
                break

        if serotype in ["h5n1", "h5n5", "h5n8"]:

            nb_first_days=15
            df_starting_events = retrieve_starting_events(df_events_by_serotype, nb_first_days)

            # TODO: spring and wintering date depends on where we are in the hemisphere
            path_list = trace_all_movement_paths_by_serotype(df_events_by_serotype, df_starting_events,
                                         df_spatial_dist_matrix, df_temporal_dist_matrix,
                                         migration_returning_back_date, final_end_date, random_seed)

            out_cascades_filepath = os.path.join(output_cascades_folder, "cascades_from_flyway_movements_disease="+serotype+".csv")
            path_str_list = []
            size_list = []
            for path in path_list:
                path_str = [str(i) for i in path]
                size_list.append(len(path_str))
                path_str = ",".join(path_str)
                path_str_list.append(path_str)
            df_cascades = pd.DataFrame({'cascade': path_str_list, 'size': size_list})
            df_cascades = df_cascades[df_cascades["size"] >= min_cascade_size]
            df_cascades.to_csv(out_cascades_filepath, sep=";", index=False)



# definition of the starting interval: [first, first date + nb_first_days]
def retrieve_starting_events(df_events, nb_first_days=15):
    df_events.sort_values(by=[consts.COL_PUBLISHED_TIME], ascending=True, inplace=True)
    first_date = df_events.iloc[0][consts.COL_PUBLISHED_TIME]
    init_dates = [first_date + datetime.timedelta(days=i) for i in range(nb_first_days)]
    df_starting_events = df_events[df_events[consts.COL_PUBLISHED_TIME].isin(init_dates)]
    df_starting_events = df_starting_events[~df_starting_events.duplicated(subset=["ADM1_geonameid"])].copy()
    return df_starting_events


if __name__ == '__main__':
    print('Starting')
    MIN_CASCADE_SIZE = 10
    random_seed = 0
    migration_returning_back_date = parser.parse('2021-07-01') # summer starting date for north hemispehere, winter staring date for south hemisphere
    final_end_date = parser.parse('2022-01-01')


    output_preprocessing_folder = os.path.join(consts.OUT_FOLDER, "preprocessing")
    output_cascades_folder = os.path.join(output_preprocessing_folder, "cascades")
    try:
        if not os.path.exists(output_cascades_folder):
          os.makedirs(output_cascades_folder)
    except OSError as err:
        print(err)

    events_filepath = os.path.join(output_preprocessing_folder, "processed_empres-i_events_updated_with_flyway.csv")  # only 2021 data
    df_events_prep_upd = pd.read_csv(events_filepath, sep=";", keep_default_na=False)
    df_events_prep_upd[consts.COL_PUBLISHED_TIME] = df_events_prep_upd[consts.COL_PUBLISHED_TIME].apply(lambda x: parser.parse(x))
    df_events_prep_upd["disease_cluster"] = df_events_prep_upd["disease_cluster"].apply(lambda x: eval(x))
    df_events_prep_upd["flyway_info"] = df_events_prep_upd["flyway_info"].apply(lambda x: eval(x))

    eventId2geonamesId = dict(zip(df_events_prep_upd["id"], df_events_prep_upd["ADM1_geonameid"]))

    in_map_folder = consts.IN_MAP_SHAPEFILE_FOLDER
    adm1_map_shape_filepath = os.path.join(in_map_folder, "world", "ne_10m_admin_1_states_provinces", "naturalearth_adm1_with_fixed_geometries.shp")
    map_data = gpd.read_file(adm1_map_shape_filepath, encoding="utf-8")
    map_data = map_data.to_crs(4326)

    world_map_csv_filepath = os.path.join(in_map_folder, "world", "ne_10m_admin_1_states_provinces", "naturalearth_adm1.csv")
    df_world_map_info = pd.read_csv(world_map_csv_filepath, usecols=["gn_id", "name", "admin", "lon", "lat"], sep=";",
                         keep_default_na=False)
    #df_world_map_info_nz = df_world_map_info[df_world_map_info["gn_id"] != -1]
    #df_world_map_info_nz = df_world_map_info

    spatial_dist_matrix_filepath = os.path.join(output_preprocessing_folder, "spatial_dist_matrix_from_events.csv")
    temporal_dist_matrix_filepath = os.path.join(output_preprocessing_folder, "temporal_dist_matrix.csv")
    df_spatial_dist_matrix = pd.read_csv(spatial_dist_matrix_filepath, sep=";", keep_default_na=False, index_col=0)
    df_temporal_dist_matrix = pd.read_csv(temporal_dist_matrix_filepath, sep=";", keep_default_na=False, index_col=0)

    # --------------------------
    clusters = set()
    for cl in df_events_prep_upd["disease_cluster"].tolist():
        for c in cl:
            clusters.add(c)
    nb_clusters = len(clusters)
    # --------------------------

    trace_all_movement_paths_with_serotype_info(df_events_prep_upd, nb_clusters, df_spatial_dist_matrix, df_temporal_dist_matrix,
                                                    migration_returning_back_date, final_end_date, MIN_CASCADE_SIZE, random_seed)

    # # ------

    # serotype = "h5n1"
    # cascades_filepath = os.path.join(output_cascades_folder, "cascades_from_flyway_movements_disease=" + serotype + ".csv")
    # df_cascades  = pd.read_csv(cascades_filepath, sep=";", keep_default_na=False)
    # cascade_idx = 0
    # cascade_str = df_cascades.loc[0,"cascade"]
    # cascade_name = "cascade"+str(cascade_idx)+"_disease="+serotype
    #
    # out_cascade_graphml_filepath = os.path.join(output_cascades_folder, cascade_name+".graphml")
    # create_cascade_graph(df_world_map_info, cascade_str, eventId2geonamesId, out_cascade_graphml_filepath)
    #
    # output_cascade_plot_filepath = os.path.join(output_cascades_folder, cascade_name + ".png")
    # plot_graph_on_map2(out_cascade_graphml_filepath, map_data, output_cascade_plot_filepath)