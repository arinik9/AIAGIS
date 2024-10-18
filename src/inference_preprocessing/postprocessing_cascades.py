
import numpy as np
import pandas as pd
from src.util_event import read_events_from_df
import src.consts as consts
import os





if __name__ == '__main__':
    print('Starting')
    output_preprocessing_folder = os.path.join(consts.OUT_FOLDER, "preprocessing")
    output_cascades_folder = os.path.join(output_preprocessing_folder, "cascades")
    try:
        if not os.path.exists(output_cascades_folder):
          os.makedirs(output_cascades_folder)
    except OSError as err:
        print(err)

    events_filepath = os.path.join(output_preprocessing_folder, "processed_empres-i_events_updated.csv") # only 2021 data
    df_events_prep_upd = pd.read_csv(events_filepath, sep=";", keep_default_na=False)
    #df_events_prep_upd["published_at"] = df_events_prep_upd["published_at"].apply(lambda x: parser.parse(x))

    events = read_events_from_df(df_events_prep_upd)
    serotype_list = [e.disease.get_disease_data()["serotype"] for e in events if e.disease.get_disease_data()["serotype"] != "unknown serotype" ]
    serotype_uniq_list = np.unique(serotype_list)

    #MIN_CASCADE_SIZE = 5

    df_cascades_list = []
    for serotype in serotype_uniq_list:
        cascades_filepath = os.path.join(output_cascades_folder, "cascades_from_st_clustering_disease=" + serotype + ".csv")
        df_cascades_st_clustering = pd.read_csv(cascades_filepath, sep=";", keep_default_na=False)
        #df_cascades_st_clustering["size"] = df_cascades_st_clustering["cascade"].apply(lambda x: len(x.split(",")))
        #df_cascades_st_clustering = df_cascades_st_clustering[df_cascades_st_clustering["size"] >= MIN_CASCADE_SIZE]

        cascades_filepath = os.path.join(output_cascades_folder, "cascades_from_flyway_movements_disease="+serotype+".csv")
        df_cascades_flyway_movements = pd.read_csv(cascades_filepath, sep=";", keep_default_na=False)
        #df_cascades_flyway_movements["size"] = df_cascades_flyway_movements["cascade"].apply(lambda x: len(x.split(",")))
        #df_cascades_flyway_movements = df_cascades_flyway_movements[df_cascades_flyway_movements["size"]>=MIN_CASCADE_SIZE]

        df_cascades = pd.concat([df_cascades_st_clustering, df_cascades_flyway_movements])
        out_cascades_filepath = os.path.join(output_cascades_folder, "cascades_disease=" + serotype + ".csv")
        df_cascades.to_csv(out_cascades_filepath, sep=";", index=False)

        df_cascades_list.append(df_cascades)

    df_cascades = pd.concat(df_cascades_list)
    out_cascades_filepath = os.path.join(output_cascades_folder, "cascades.csv")
    df_cascades.to_csv(out_cascades_filepath, sep=";", index=False)