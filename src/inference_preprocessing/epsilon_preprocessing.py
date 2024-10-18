import os
import pandas as pd
import numpy as np
import csv

def perform_epsilon_preprocessing(df_events, in_map_folder, output_epsilon_scores_filepath):
    world_map_csv_filepath = os.path.join(in_map_folder, "world", "ne_10m_admin_1_states_provinces", "naturalearth_adm1.csv")
    df_world_map_info = pd.read_csv(world_map_csv_filepath, usecols=["gn_id", "name", "admin", "lon", "lat"], sep=";",
                         keep_default_na=False)
    id_list = df_world_map_info["gn_id"].to_numpy().flatten()
    N = len(id_list)

    gn_id_list = []
    uncertainty_score_list = []
    for i in range(N):
        gn_id = id_list[i]
        df_res = df_events[df_events["ADM1_geonameid"] == gn_id]
        final_score = 0.9999
        if df_res.shape[0]>0:
            matching_scores = []
            # example= [[], [], [], [0.362421212044947], [], [], [], []]
            for x in df_res["bvbrc_matching_score"].tolist():
                if len(x) == 0:
                    matching_scores = matching_scores + [0.9999]
                else:
                    matching_scores = matching_scores + x
            if len(matching_scores)>0:
                matching_scores = np.array(matching_scores)
                matching_scores[matching_scores == 1.0] = 0.9999 # adjustment
                matching_scores[matching_scores == 0.0] = 0.0001 # adjustment
                uncertainty_scores = 1-matching_scores
                final_score = np.mean(uncertainty_scores)
        gn_id_list.append(gn_id)
        uncertainty_score_list.append(final_score)

    df = pd.DataFrame(list(zip(gn_id_list, uncertainty_score_list)),
                              columns = ["gn_id", "uncertainty_score"])
    df.to_csv(output_epsilon_scores_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)
