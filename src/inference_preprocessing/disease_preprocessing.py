


import numpy as np
import dateutil.parser as parser
import src.consts as consts
import csv
import os
import pandas as pd
from src.util_event import build_disease_instance
from src.util_event import read_events_from_df


def define_soft_clustering_structure_from_disease_info_for_events(df_events, out_events_filepath=None):
    events = read_events_from_df(df_events)

    # split the data into avian influenza strains
    events_by_strain = {}
    for e in events:
        serotype = e.disease.get_disease_data()["serotype"]
        if serotype != "unknown serotype":
            if serotype not in events_by_strain:
                events_by_strain[serotype] = []
            events_by_strain[serotype].append(e)

    for e in events:
        d1 = e.disease
        serotype = d1.get_disease_data()["serotype"]
        if serotype == "unknown serotype":
            for serotype in events_by_strain:
                events_cluster = events_by_strain[serotype]
                first_event = events_cluster[0]  # all events of a cluster has the same serotype
                d2 = first_event.disease
                if d2.hierarchically_includes(d1):
                    events_by_strain[serotype].append(e)

    eventId2clusterId = {}
    for cluster_id, serotype in enumerate(events_by_strain):
        events_cluster = events_by_strain[serotype]
        for e in events_cluster:
            if e.e_id not in eventId2clusterId:
                eventId2clusterId[e.e_id] = []
            eventId2clusterId[e.e_id].append(cluster_id)

    df_events["disease_cluster"] = df_events["id"].apply(lambda x: str(eventId2clusterId[x]) if x in eventId2clusterId else "-1")
    # we remove the events in this case: for instance, there is an event with h7 disease, and there is not any specific disease with h7 like h7n6, so we remove this event
    df_events = df_events[df_events["disease_cluster"] != "-1"].reset_index()

    if out_events_filepath is not None:
        df_events.to_csv(out_events_filepath, sep=";", index=False)
    return df_events


def create_disease_compatibility_matrix_for_events(df_events, output_compatibility_matrix_filepath):
    N = df_events.shape[0]
    D = np.full(shape=(N,N), fill_value=np.nan)

    id_list = df_events[consts.COL_ID].to_numpy().flatten()

    for i, row1 in df_events.iterrows():
        print(i, "/", N)
        D[i, i] = 1
        d1 = build_disease_instance(row1["disease"])
        for j, row2 in df_events.iterrows():
           if i<j:
               d2 = build_disease_instance(row2["disease"])
               if d1.hierarchically_includes(d2) or d2.hierarchically_includes(d1) or d1.is_identical(d2):
                   D[i,j] = 1
                   D[j,i] = 1
               else:
                   D[i,j] = 0
                   D[j,i] = 0

    df = pd.DataFrame(D)
    df.index = id_list
    df.columns = id_list
    df.to_csv(output_compatibility_matrix_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)


def perform_disease_preprocessing(df_events):
    df_events = define_soft_clustering_structure_from_disease_info_for_events(df_events)
    return df_events


if __name__ == '__main__':
    print('Starting')
    output_preprocessing_folder = os.path.join(consts.OUT_FOLDER, "preprocessing")
    events_filepath = os.path.join(output_preprocessing_folder, "processed_empres-i_events_updated.csv") # only 2021 data
    df_events_prep_upd = pd.read_csv(events_filepath, sep=";", keep_default_na=False)
    #df_events_prep_upd["published_at"] = df_events_prep_upd["published_at"].apply(lambda x: parser.parse(x))

    output_compatibility_matrix_filepath = os.path.join(output_preprocessing_folder, "disease_compatibility_matrix_for_events.csv")
    #create_disease_compatibility_matrix_for_events(df_events_prep_upd, output_compatibility_matrix_filepath)

    #df_comp_matrix = pd.read_csv(output_compatibility_matrix_filepath, sep=";", keep_default_na=False)
    out_events_filepath = os.path.join(output_preprocessing_folder, "processed_empres-i_events_updated.csv")
    define_soft_clustering_structure_from_disease_info_for_events(df_events_prep_upd, out_events_filepath)
