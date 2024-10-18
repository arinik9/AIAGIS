
import time
import os
import dateutil.parser as parser

import consts

#from src.inference.MultiC.multic_m import perform_MultiC_multilayer_phase
from src.inference_preprocessing.perform_preprocessing import perform_preprocessing
from src.inference.MultiC.multic_s import perform_MultiC_single_layer_phase_with_all_configs

if __name__ == '__main__':
    print('Starting')
    start = time.time()

    force = False

    # #########################################
    # Options
    # #########################################
    #date_start = parser.parse("2020-12-31T00:00:00", dayfirst=False)
    #date_end = parser.parse("2021-07-01T00:00:00", dayfirst=False)  # ending time
    #date_end = parser.parse("2021-04-01T00:00:00", dayfirst=False)  # ending time
    #date_start = parser.parse("2020-09-01T00:00:00", dayfirst=False)
    #date_end = parser.parse("2021-05-01T00:00:00", dayfirst=False)  # ending time
    #date_start = parser.parse("2020-12-31T00:00:00", dayfirst=False)
    #date_end = parser.parse("2021-04-01T00:00:00", dayfirst=False)  # ending time
    #date_end = parser.parse("2022-01-01T00:00:00", dayfirst=False)  # ending tim
    #date_start = parser.parse("2015-12-31T00:00:00", dayfirst=False)
    #date_end = parser.parse("2016-07-01T00:00:00", dayfirst=False)  # ending time

    date_start = parser.parse("2016-10-01T00:00:00", dayfirst=False)
    date_end = parser.parse("2017-01-01T00:00:00", dayfirst=False)  # ending time

    #date_start = parser.parse("2020-10-01T00:00:00", dayfirst=False)
    #date_end = parser.parse("2021-01-01T00:00:00", dayfirst=False)  # ending time

    #date_start = parser.parse("2016-10-01T00:00:00", dayfirst=False)
    #date_end = parser.parse("2017-01-01T00:00:00", dayfirst=False)  # ending time

    # nice source: https://www.ecdc.europa.eu/sites/default/files/documents/avian-influenza-overview-joint-report-October-2017.pdf

    # GenBank and GISAID sequence accession numbers https://gisaid.org

    # source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6746284/
    # The fifth wave of H7N9 infections (October 2016-September 2017) was the largest in terms of geographical distribution, number of human cases and outbreaks in poultry

    # source: https://www.cdc.gov/bird-flu/avian-timeline/2020s.html
    # These wild bird-adapted HPAI H5N1 viruses were first identified in Europe during the fall of 2020 and spread across Europe and into Africa, the Middle East and Asia24.

    # source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8161317/
    # Between November 2019 and February 2020, 37 H5N8 HPAI virus outbreaks were reported in poultry (34 cases), captive birds (1 case), and wild birds (2 cases) in Europe alone [26].
    # Genomic characterization of the previously discussed H5N8 strain that emerged in Europe was evaluated as reassortant of H5N8 HPAI strains from Africa and LPAI strains from Eurasia

    # year_values = [2021] # [2019, 2020, 2021]
    # # nb_windows = 12 # 3 years
    # nb_months_per_window = 3
    # modelling_options = {}
    # # modelling_options["proximity_distance_km_per_week"] = 2000 # ~ 280 x 7
    # modelling_options["link_restriction_choice"] = "positive_delta"
    # # modelling_options["link_restriction_choice"] = None

    # #########################################
    # Preprocessing
    # #########################################
    in_events_filepath = os.path.join(consts.OUT_FOLDER, "doc_events_empres-i_task1=structured_data_with_genome_data.csv")
    in_map_folder = consts.IN_MAP_SHAPEFILE_FOLDER

    in_bvbrc_seqs_filepath = os.path.join(consts.IN_BVBRC_FOLDER, "genome_sequences.csv")

    output_preprocessing_folder = os.path.join(consts.OUT_FOLDER, "inference_preprocessing")
    try:
        if not os.path.exists(output_preprocessing_folder):
            os.makedirs(output_preprocessing_folder)
    except OSError as err:
        print(err)

    in_db_isolate_sim_dir = os.path.join(consts.GENOME_PREPROCESSING_BVBRC_FOLDER, "analysis_summary") # TODO: name it as "isolate_avg_sim_scores"

    out_processed_events_filepath = os.path.join(output_preprocessing_folder, "processed_empres-i_events.csv")
    output_spatial_dist_matrix_filepath = os.path.join(output_preprocessing_folder, "spatial_dist_matrix_from_map.csv")
    output_temp_dist_matrix_from_events_filepath = os.path.join(output_preprocessing_folder, "temporal_dist_matrix.csv")
    output_genome_dist_matrix_from_events_filepath = os.path.join(output_preprocessing_folder, "genome_dist_matrix_from_events.csv")
    output_genome_dist_matrix_from_map_filepath = os.path.join(output_preprocessing_folder, "genome_dist_matrix_from_map.csv")
    output_genome_raw_dist_scores_from_events_filepath = os.path.join(output_preprocessing_folder, "genome_raw_dist_scores_from_events.csv")
    output_genome_dist_scores_from_events_with_missing_info_filepath = os.path.join(output_preprocessing_folder, "genome_dist_scores_from_events_with_missing_info.csv")
    output_genome_dist_scores_from_events_filepath = os.path.join(output_preprocessing_folder, "genome_dist_scores_from_events.csv")
    output_epsilon_scores_filepath = os.path.join(output_preprocessing_folder, "epsilon_scores_from_map.csv")
    out_map_with_flyway_shape_filepath = os.path.join(output_preprocessing_folder, "naturalearth_adm1_with_fixed_geometries_and_flyway.shp")
    out_map_with_flyway_csv_filepath = os.path.join(output_preprocessing_folder, "naturalearth_adm1_with_fixed_geometries_and_flyway.csv")

    output_cascades_info_filepath  = ""
    for cascade_strategy in ["serotype"]: # ["default", "serotype"]
        #cascade_strategy = "default"
        output_cascades_info_filepath = os.path.join(output_preprocessing_folder, "estimated_cascades_with_strategy="+cascade_strategy+".csv")

        perform_preprocessing(in_events_filepath, date_start, date_end, in_map_folder,
                              out_processed_events_filepath, output_spatial_dist_matrix_filepath,
                              output_temp_dist_matrix_from_events_filepath,
                              in_bvbrc_seqs_filepath, in_db_isolate_sim_dir,
                              output_genome_dist_matrix_from_events_filepath,
                              output_genome_dist_matrix_from_map_filepath,
                              output_genome_raw_dist_scores_from_events_filepath,
                              output_genome_dist_scores_from_events_with_missing_info_filepath,
                              output_genome_dist_scores_from_events_filepath, output_epsilon_scores_filepath,
                              cascade_strategy, output_cascades_info_filepath,
                              out_map_with_flyway_shape_filepath, out_map_with_flyway_csv_filepath, force)

        # #########################################
        # Inference with MultiC
        # #########################################
        print("starting the inference with MultiC ....")
        force = False

        output_folder = os.path.join(consts.OUT_FOLDER, "inference", "cascade_strategy="+cascade_strategy)
        try:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
        except OSError as err:
            print(err)

        obj_func = "dist_survival"
        #obj_func = "spatial_survival"
        include_unactivated_nodes = True

        infer_params_list = []
        infer_params = {"include_spatial_dist": False, "include_genome_dist": False,
                        "include_epsilon": False, "include_unactivated_nodes": include_unactivated_nodes,
                        "obj_func": obj_func, "beta": None, "gamma": None}
        infer_params_list.append(infer_params)
        for beta, gamma in [(0.01, 0.01)]: # [0.001, 0.005, 0.01, 0.05, 0.1] (0.1, 0.01), (0.01, 0.1), (0.01, 0.01),
            infer_params = {"include_spatial_dist": True, "include_genome_dist": False,
                            "include_epsilon": False, "include_unactivated_nodes": include_unactivated_nodes,
                            "obj_func": obj_func, "beta": beta, "gamma": None}
            infer_params_list.append(infer_params)
            # infer_params = {"include_spatial_dist": False, "include_genome_dist": True,
            #                 "include_epsilon": False, "include_unactivated_nodes": include_unactivated_nodes,
            #                 "obj_func": obj_func, "beta": None, "gamma": gamma}
            # infer_params_list.append(infer_params)
            infer_params = {"include_spatial_dist": True, "include_genome_dist": True,
                            "include_epsilon": False, "include_unactivated_nodes": include_unactivated_nodes,
                            "obj_func": obj_func, "beta": beta, "gamma": gamma}
            infer_params_list.append(infer_params)

        #world_map_shape_filepath = out_map_with_flyway_shape_filepath
        #world_map_csv_filepath = out_map_with_flyway_csv_filepath
        world_map_shape_filepath = os.path.join(in_map_folder, "world", "ne_10m_admin_1_states_provinces", "naturalearth_adm1_with_fixed_geometries.shp")
        world_map_csv_filepath = os.path.join(in_map_folder, "world", "ne_10m_admin_1_states_provinces", "naturalearth_adm1.csv")
        spatial_dist_matrix_filepath = output_spatial_dist_matrix_filepath
        preprocessed_events_filepath = out_processed_events_filepath
        cascades_info_filepath = output_cascades_info_filepath
        out_graph_single_layer_filename = "single_layer_graph.graphml"
        out_dist_values_filepath = os.path.join(output_folder, "dist_values_for_succ_edges.csv")

        perform_MultiC_single_layer_phase_with_all_configs(preprocessed_events_filepath, output_folder, date_start, date_end, cascades_info_filepath, \
                                          spatial_dist_matrix_filepath, output_genome_dist_matrix_from_events_filepath,
                                          world_map_csv_filepath, world_map_shape_filepath, out_graph_single_layer_filename, out_dist_values_filepath,
                                          infer_params_list, force)


    # # #########################################
    # # Inference with MultiC
    # # #########################################
    # print("starting the inference with MultiC ....")
    # output_folder = consts.OUT_INFERENCE_EMPRESS_I_FOLDER
    # try:
    #     if not os.path.exists(output_folder):
    #       os.makedirs(output_folder)
    # except OSError as err:
    #     print(err)
    #
    # world_map_shapefilepath = os.path.join(in_map_folder, "world", "ne_10m_admin_1_states_provinces", "naturalearth_adm1.shp")
    # world_map_filepath = os.path.join(in_map_folder, "world", "ne_10m_admin_1_states_provinces", "naturalearth_adm1.csv")
    # spatial_dist_matrix_filepath = output_spatial_dist_matrix_filepath
    # preprocessed_events_filepath = out_processed_events_filepath
    # cascades_info_filepath = output_cascades_info_filepath
    # graph_single_layer_filename = "single_layer_graph.graphml"
    # # perform_MultiC_single_layer_phase(preprocessed_events_filepath, date_start, date_end, cascades_info_filepath,\
    # #                spatial_dist_matrix_filepath, world_map_filepath, output_folder, graph_single_layer_filename)
    # # print("ending the inference with MultiC ....")
    # # end = time.time()
    # # elapsed_time = end - start
    # # tot_exec_mins = elapsed_time / 60
    # # print('Total execution time:', tot_exec_mins, 'minutes')
    # # print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    #
    # K = 2
    # graph_single_layer_filepath = os.path.join(output_folder, graph_single_layer_filename)
    # perform_MultiC_multilayer_phase(preprocessed_events_filepath, graph_single_layer_filepath, date_start, date_end, cascades_info_filepath,\
    #                spatial_dist_matrix_filepath, world_map_filepath, output_folder, K)