
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import src.consts as consts
import dateutil.parser as parser



def add_flyway_info_into_events_and_map_data(df_events, world_map_shape_filepath, bird_flyways_shape_filepath,
                                             out_map_with_flyway_shape_filepath, out_map_with_flyway_csv_filepath):

    map_data = gpd.read_file(world_map_shape_filepath, encoding="utf-8")
    map_data = map_data.to_crs("epsg:3857")
    print(map_data.shape)

    flyway_data = gpd.read_file(bird_flyways_shape_filepath, encoding="utf-8")
    flyway_data = flyway_data.to_crs("epsg:3857")

    new_data = gpd.overlay(map_data, flyway_data, how='intersection')
    #new_data.to_file(driver='ESRI Shapefile', filename=os.path.join(output_preprocessing_folder, 'temp.shp'), encoding="utf-8")

    # this new geodataframe can gave more rows than the original geodataframe, because some zones can be in multuple flyways
    #print(new_data.shape)
    new_data = new_data[new_data["gn_id"] != -1]
    print(new_data.shape)
    # new_data["id"] : flyway id
    # new_data["name_2"] : flyway name
    geonameId2flyways = {}
    for i, row in new_data.iterrows():
        gn_id = int(row["gn_id"])
        if gn_id not in geonameId2flyways:
            geonameId2flyways[gn_id] = []
        geonameId2flyways[gn_id].append(row["name_2"])

    #print(geonameId2flyways)
    df_events["flyway_info"] = df_events["ADM1_geonameid"].apply(lambda x: str(geonameId2flyways[int(x)]) if int(x) in geonameId2flyways else "")

    map_data["flyway_info"] = map_data["gn_id"].apply(lambda x: str(geonameId2flyways[int(x)]) if int(x) in geonameId2flyways else "")
    #map_data.rename({'lng': 'lon'}, axis=1, inplace=True)
    map_data.to_file(driver='ESRI Shapefile', filename=out_map_with_flyway_shape_filepath, encoding="utf-8")

    df = pd.DataFrame(map_data.drop(columns='geometry'))
    #df.rename({'lng': 'lon'}, axis=1, inplace=True)
    df.to_csv(out_map_with_flyway_csv_filepath, sep=";", index=False)

    return df_events

# if __name__ == '__main__':
#     print('Starting')
#     output_preprocessing_folder = os.path.join(consts.OUT_FOLDER, "preprocessing")
#
#     events_filepath = os.path.join(output_preprocessing_folder, "processed_empres-i_events.csv")  # only 2021 data
#     df_events_prep_upd = pd.read_csv(events_filepath, sep=";", keep_default_na=False)
#     df_events_prep_upd[consts.COL_PUBLISHED_TIME] = df_events_prep_upd[consts.COL_PUBLISHED_TIME].apply(lambda x: parser.parse(x))
#
#     in_map_folder = consts.IN_MAP_SHAPEFILE_FOLDER
#     adm1_map_shape_filepath = os.path.join(in_map_folder, "world", "ne_10m_admin_1_states_provinces", "naturalearth_adm1_with_fixed_geometries.shp")
#     map_data = gpd.read_file(adm1_map_shape_filepath, encoding="utf-8")
#     map_data = map_data.to_crs("epsg:3857")
#     print(map_data.shape)
#
#     in_bird_folder = os.path.join(consts.IN_EXT_DATA_FOLDER, "bird_flyways")
#     bird_flyways_shape_filepath = os.path.join(in_bird_folder, "bird_flyways.shp")
#     flyway_data = gpd.read_file(bird_flyways_shape_filepath, encoding="utf-8")
#     flyway_data = flyway_data.to_crs("epsg:3857")
#
#     new_data = gpd.overlay(map_data, flyway_data, how='intersection')
#     #new_data.to_file(driver='ESRI Shapefile', filename=os.path.join(output_preprocessing_folder, 'temp.shp'), encoding="utf-8")
#
#     # this new geodataframe can gave more rows than the original geodataframe, because some zones can be in multuple flyways
#     #print(new_data.shape)
#     new_data = new_data[new_data["gn_id"] != -1]
#     print(new_data.shape)
#     # new_data["id"] : flyway id
#     # new_data["name_2"] : flyway name
#     geonameId2flyways = {}
#     for i, row in new_data.iterrows():
#         gn_id = int(row["gn_id"])
#         if gn_id not in geonameId2flyways:
#             geonameId2flyways[gn_id] = []
#         geonameId2flyways[gn_id].append(row["name_2"])
#
#     print(geonameId2flyways)
#     df_events_prep_upd["flyway_info"] = df_events_prep_upd["ADM1_geonameid"].apply(lambda x: str(geonameId2flyways[int(x)]) if int(x) in geonameId2flyways else "")
#     out_filepath = os.path.join(output_preprocessing_folder, "processed_empres-i_events_updated_with_flyway.csv")
#     df_events_prep_upd.to_csv(out_filepath, sep=";", index=False)
#
#
#     map_data["flyway_info"] = map_data["gn_id"].apply(lambda x: str(geonameId2flyways[int(x)]) if int(x) in geonameId2flyways else "")
#     out_filepath = os.path.join(in_map_folder, "naturalearth_adm1_with_fixed_geometries_and_flyway.shp")
#     map_data.to_file(driver='ESRI Shapefile', filename=out_filepath, encoding="utf-8")
#
#     out_filepath = os.path.join(in_map_folder, "naturalearth_adm1_with_fixed_geometries_and_flyway.csv")
#     df = pd.DataFrame(map_data.drop(columns='geometry'))
#     df.to_csv(out_filepath, sep=";", index=False)