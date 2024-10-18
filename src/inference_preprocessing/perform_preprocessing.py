
import csv
import os
import pandas as pd

from src.util_event import read_df_events
from src.inference_preprocessing.spatial_preprocessing import perform_spatial_preprocessing
from src.inference_preprocessing.time_preprocessing import perform_time_preprocessing
from src.inference_preprocessing.host_preprocessing import perform_host_preprocessing
from src.inference_preprocessing.disease_preprocessing import perform_disease_preprocessing
from src.inference_preprocessing.genome_preprocessing import perform_genome_preprocessing
from src.inference_preprocessing.cascade_preprocessing import perform_cascade_preprocessing_default, perform_cascade_preprocessing_by_serotype
from src.inference_preprocessing.add_flyway_info_into_events_and_map_data import add_flyway_info_into_events_and_map_data
#from src.preprocessing.estimate_cascades_st_clustering import estimate_cascades_with_seasonal_periods, estimate_cascades_with_st_clustering
import src.consts as consts

from src.util_event import build_disease_instance

from src.inference_preprocessing.epsilon_preprocessing import perform_epsilon_preprocessing

def perform_preprocessing(events_filepath, date_start, date_end, in_map_folder,
                          out_processed_events_filepath,
                          output_spatial_dist_matrix_filepath, output_temp_dist_matrix_from_events_filepath,
                          in_bvbrc_seqs_filepath, in_db_isolate_sim_dir,
                          output_genome_dist_matrix_from_events_filepath, output_genome_dist_matrix_from_map_filepath,
                          output_genome_raw_dist_scores_from_events_filepath,
                          output_genome_dist_scores_from_events_with_missing_info_filepath,
                          output_genome_dist_scores_from_events_filepath, output_epsilon_scores_filepath,
                          cascade_strategy, output_cascades_info_filepath,
                          out_map_with_flyway_shape_filepath, out_map_with_flyway_csv_filepath, force=False):

    df_bvbrc_seqs = pd.read_csv(in_bvbrc_seqs_filepath, sep=";", keep_default_na=False, dtype=str)

    in_map_folder = consts.IN_MAP_SHAPEFILE_FOLDER
    world_map_csv_filepath = os.path.join(in_map_folder, "world", "ne_10m_admin_1_states_provinces", "naturalearth_adm1.csv")
    df_world_map_info = pd.read_csv(world_map_csv_filepath, usecols=["gn_id", "name", "admin", "lon", "lat"], sep=";",
                         keep_default_na=False)
    #df_world_map_info_nz = df_world_map_info[df_world_map_info["gn_id"] != -1]
    df_world_map_info_nz = df_world_map_info

    # Simple preprocessing
    extra_cols = ["bvbrc_id", "bvbrc_article_id", "bvbrc_genome_name", "bvbrc_matching_score", "bvbrc_genBank_accessions", "bvbrc_segment2genomeID",
                 "bvbrc_delta_dist", "bvbrc_delta_temp"]
    df_events = read_df_events(events_filepath, extra_cols)
    df_events = df_events[df_events[consts.COL_DISEASE].apply(lambda x: build_disease_instance(x).get_max_hierarchy_level()>1)]

    df_events["hierarchy_data"] = df_events["hierarchy_data"].apply(lambda x: eval(x))
    df_events["bvbrc_id"] = df_events["bvbrc_id"].apply(lambda x: eval(x))
    df_events["bvbrc_article_id"] = df_events["bvbrc_article_id"].apply(lambda x: eval(x))
    df_events["bvbrc_genome_name"] = df_events["bvbrc_genome_name"].apply(lambda x: eval(x))
    df_events["bvbrc_segment2genomeID"] = df_events["bvbrc_segment2genomeID"].apply(lambda x: eval(x))
    # df_events["bvbrc_segment2genomeID"] = df_events["bvbrc_segment2genomeID"].apply(lambda l: [json.loads(i) for i in l])
    df_events["bvbrc_matching_score"] = df_events["bvbrc_matching_score"].apply(lambda x: eval(x))
    df_events["bvbrc_genBank_accessions"] = df_events["bvbrc_genBank_accessions"].apply(lambda x: eval(x))
    df_events["bvbrc_delta_dist"] = df_events["bvbrc_delta_dist"].apply(lambda x: eval(x))
    df_events["bvbrc_delta_temp"] = df_events["bvbrc_delta_temp"].apply(lambda x: eval(x))

    df_events = perform_spatial_preprocessing(df_events, in_map_folder, output_spatial_dist_matrix_filepath)
    # df_events.to_csv(out_processed_events_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC) # TODO
    df_events = perform_time_preprocessing(df_events, date_start, date_end, output_temp_dist_matrix_from_events_filepath)
    df_events.to_csv(out_processed_events_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC) # TODO
    df_events = perform_disease_preprocessing(df_events)
    df_events.to_csv(out_processed_events_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)  # TODO
    # df_events = perform_host_preprocessing(df_events, in_external_data_folder)
    # df_events.to_csv(out_processed_events_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC) # TODO
    perform_genome_preprocessing(df_events, df_bvbrc_seqs, in_db_isolate_sim_dir, df_world_map_info_nz, output_genome_dist_matrix_from_events_filepath,
                                             output_genome_dist_matrix_from_map_filepath,
                                             output_genome_raw_dist_scores_from_events_filepath,
                                 output_genome_dist_scores_from_events_with_missing_info_filepath,
                                 output_genome_dist_scores_from_events_filepath, force)
    ####df_events.to_csv(out_processed_events_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC) # TODO


    perform_epsilon_preprocessing(df_events, in_map_folder, output_epsilon_scores_filepath)

    min_cascade_size = 5
    if cascade_strategy == "serotype":
        perform_cascade_preprocessing_by_serotype(df_events, output_cascades_info_filepath, min_cascade_size)
    else:
        perform_cascade_preprocessing_default(df_events, output_cascades_info_filepath, min_cascade_size)

    in_map_folder = consts.IN_MAP_SHAPEFILE_FOLDER
    world_map_shape_filepath = os.path.join(in_map_folder, "world", "ne_10m_admin_1_states_provinces",
                                           "naturalearth_adm1_with_fixed_geometries.shp")
    in_bird_folder = os.path.join(consts.DATA_FOLDER, "bird_flyways")
    bird_flyways_shape_filepath = os.path.join(in_bird_folder, "bird_flyways.shp")
    df_events = add_flyway_info_into_events_and_map_data(df_events, world_map_shape_filepath, bird_flyways_shape_filepath,
                                                 out_map_with_flyway_shape_filepath, out_map_with_flyway_csv_filepath)
    df_events.to_csv(out_processed_events_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)


# # Generate cascades
    # print("start season_year clustering")
    # df_events, df_cascades0 = estimate_cascades_with_seasonal_periods(df_events)
    # print("start st clustering")
    # df_events, df_cascades1 = estimate_cascades_with_st_clustering(df_events, eps1=0.1, eps2=24 * 10)
    # df_events, df_cascades2 = estimate_cascades_with_st_clustering(df_events, eps1=0.1, eps2=24 * 20)
    # df_events, df_cascades3 = estimate_cascades_with_st_clustering(df_events, eps1=0.2, eps2=24 * 10)
    # df_events, df_cascades4 = estimate_cascades_with_st_clustering(df_events, eps1=0.2, eps2=24 * 20)
    # df_events, df_cascades5 = estimate_cascades_with_st_clustering(df_events, eps1=0.3, eps2=24 * 10)
    # df_events, df_cascades6 = estimate_cascades_with_st_clustering(df_events, eps1=0.3, eps2=24 * 20)
    #
    # df_all_cascades = pd.concat([df_cascades0, df_cascades1, df_cascades2, df_cascades3, df_cascades4, df_cascades5, df_cascades6])
    # df_all_cascades.to_csv(output_cascades_info_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)
    #
    # # # Writing into file
    # df_events.to_csv(out_processed_events_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)
