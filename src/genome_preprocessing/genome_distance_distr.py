import os
import src.consts as consts
import pandas as pd
from datetime import datetime
import dateutil.parser as parser
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    print('Starting')

    #date_str = "2018-10"
    #d = parser.parse(date_str)
    #print(d)

    #serotype_pair = "h5n8_h5n8"
    #serotype_pair = "h5n1_h5n1"
    serotype_pair = "h5n1_h5n8"
    #serotype_pair = "h10n8_h5n6"
    #serotype_pair = "h6n2_h6n2"
    fpath = os.path.join(consts.OUT_FOLDER, "event_preprocessing", "BVBRC", "analysis", "genome_sim_analysis_"+serotype_pair+".csv")
    out_fpath = os.path.join(consts.OUT_FOLDER, "event_preprocessing", "BVBRC", "analysis", "genome_sim_analysis_"+serotype_pair+"_adj.csv")
    df = pd.read_csv(fpath, sep=";", keep_default_na=False)
    print(df.columns)
    df["source_nb_date_parts"] = df["source_collection_date"].apply(lambda x: len(x.split("-"))-1)
    df["target_nb_date_parts"] = df["target_collection_date"].apply(lambda x: len(x.split("-"))-1)
    df2 = df[(df["source_nb_date_parts"] != 0) & (df["target_nb_date_parts"] != 0)]
    print(df.shape[0], df2.shape[0])

    df2["source_collection_date"] = df2["source_collection_date"].apply(lambda x: x if len(x.split("-")) == 3 else x+"-01")
    df2["target_collection_date"] = df2["target_collection_date"].apply(lambda x: x if len(x.split("-")) == 3 else x+"-01")
    df2["source_collection_date"] = df2["source_collection_date"].apply(lambda x: parser.parse(x))
    df2["target_collection_date"] = df2["target_collection_date"].apply(lambda x: parser.parse(x))
    df2["time_diff"] = (df2['source_collection_date'] - df2['target_collection_date']).dt.days
    df2["time_diff"] = df2["time_diff"].apply(lambda x: np.abs(x))
    df2["time_diff"] = df2["time_diff"].astype(int)
    df2["sim_score"] = df2["sim_score"].astype(float)
    del df2["source_nb_date_parts"]
    del df2["target_nb_date_parts"]
    df2.to_csv(out_fpath, sep=";", index=False)


    # ----------
    df = pd.read_csv(out_fpath, sep=";", keep_default_na=False)
    plt.scatter(df["time_diff"].tolist(), df["sim_score"].tolist())
    plt.savefig(out_fpath.replace(".csv", ".png"),
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            #facecolor ="g",
            edgecolor ='w',
            orientation ='landscape')
