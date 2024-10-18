
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.st_clustering.st_dbscan import ST_DBSCAN
from src.util_event import read_events_from_df
import src.consts as consts
import os
import csv
from collections import Counter
from src.util_event import build_disease_instance
import dateutil.parser as parser


# strategy 1: by period (e.g. 2021 summer, 2021 winter, etc.)
# strategy 2: space-time clustering

# # TODO: Recognize cascades of size above the cascades size threshold. In the original paper, it was 8.

def retrieve_cascades_info_as_dataframe(df_events, res_col_name, eventId2eventDate, eventId2geonamesId):
    # TODO in the same cluster, multiple events with the same location can exist. How to deal with it ?
    clustering_dict = {}
    geonamesId_dict = {} # we want the cluster to have unique ADM1 zones
    for index, row in df_events.iterrows():
        id = str(row[consts.COL_ID])
        gn_id = eventId2geonamesId[row[consts.COL_ID]]
        c_id = str(row[res_col_name])
        if c_id != "":
            # we want the cluster to have unique ADM1 zones
            if (c_id not in geonamesId_dict) or (gn_id not in geonamesId_dict[c_id]):
                if c_id not in clustering_dict:
                    clustering_dict[c_id] = []
                if c_id not in geonamesId_dict:
                    geonamesId_dict[c_id] = []
                clustering_dict[c_id].append(id)
                geonamesId_dict[c_id].append(gn_id)

    clusters_list = []
    size_list = []
    for key in clustering_dict.keys():
        event_ids_nonordered = np.array(clustering_dict[key])
        event_dates = [eventId2eventDate[int(id)] for id in event_ids_nonordered]
        event_ids_ordered = event_ids_nonordered[np.argsort(event_dates)]
        event_ids_ordered_str = [str(eid) for eid in event_ids_ordered]
        size_list.append(len(event_ids_ordered_str))
        clusters_list.append(",".join(event_ids_ordered_str))
    df_cascades = pd.DataFrame({'cascade': clusters_list, 'size': size_list})

    return df_cascades




# def estimate_cascades_with_seasonal_periods(df_events):
#     df_events["season-year-clustering"] = -1 # init
#     uniq_values = np.unique(df_events["seasonal_year_period"].to_list())
#     for i, val in enumerate(uniq_values):
#         df_events.loc[df_events["seasonal_year_period"] == val, "season-year-clustering"] = i
#
#     df_cascades = retrieve_cascades_info_as_dataframe(df_events, "season-year-clustering")
#     return df_events, df_cascades


# https://github.com/eren-ck/st_dbscan/blob/master/demo/demo.ipynb
# eps1: max spatial distance, from the range [0,1]
# eps2: max temporal distance
def estimate_cascades_with_st_clustering(df_events_by_serotype, eps1=0.1, eps2=24*10):
    # init
    RES_COL_NAME = "st-clustering_eps1=" + str(eps1) + "_eps2=" + str(eps2)
    df_events_by_serotype[RES_COL_NAME] = ""

    events = read_events_from_df(df_events_by_serotype)

    # normalize the data
    df_events_by_serotype['lng'] = (df_events_by_serotype['lng'] - df_events_by_serotype['lng'].min()) / (
                df_events_by_serotype['lng'].max() - df_events_by_serotype['lng'].min())
    df_events_by_serotype['lat'] = (df_events_by_serotype['lat'] - df_events_by_serotype['lat'].min()) / (
                df_events_by_serotype['lat'].max() - df_events_by_serotype['lat'].min())
    # transform to numpy array
    data = df_events_by_serotype.loc[:, ['timestamp', 'lng', 'lat']].values
    #
    # The input data format is: ['temporal_index', 'x', 'y', < optional attributes >]
    st_dbscan = ST_DBSCAN(eps1=eps1, eps2=eps2, min_samples=5)
    st_dbscan.fit(data)
    counter = Counter(st_dbscan.labels)
    #print(counter)
    # -1: outlier
    labels_str = [str(l) if l != -1 else "" for l in st_dbscan.labels]
    df_events_by_serotype[RES_COL_NAME] = labels_str
    df_events_by_serotype = df_events_by_serotype[df_events_by_serotype[RES_COL_NAME] != ""]

    # -----
    eventId2eventDate = dict(zip(df_events_by_serotype['id'], df_events_by_serotype['published_at']))
    eventId2geonamesId = dict(zip(df_events_by_serotype['id'], df_events_by_serotype['geonames_id']))
    df_cascades = retrieve_cascades_info_as_dataframe(df_events_by_serotype, RES_COL_NAME, eventId2eventDate, eventId2geonamesId)
    # -----

    # df_events.reset_index(drop=True, inplace=True)
    return df_events_by_serotype, df_cascades




# ============================================================
#
# ============================================================
def plot(data, labels):
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6',
              '#6a3d9a']

    for i in range(-1, len(set(labels))):
        if i == -1:
            col = [0, 0, 0, 1]
        else:
            col = colors[i % len(colors)]

        clust = data[np.where(labels == i)]
        plt.scatter(clust[:, 0], clust[:, 1], c=[col], s=1)
    plt.show()

    return None


if __name__ == '__main__':
    print('Starting')
    MIN_CASCADE_SIZE = 5

    output_preprocessing_folder = os.path.join(consts.OUT_FOLDER, "preprocessing")
    output_cascades_folder = os.path.join(output_preprocessing_folder, "cascades")
    try:
        if not os.path.exists(output_cascades_folder):
          os.makedirs(output_cascades_folder)
    except OSError as err:
        print(err)

    events_filepath = os.path.join(output_preprocessing_folder, "processed_empres-i_events.csv") # only 2021 data
    df_events_prep_upd = pd.read_csv(events_filepath, sep=";", keep_default_na=False)
    df_events_prep_upd["published_at"] = df_events_prep_upd["published_at"].apply(lambda x: parser.parse(x))
    df_events_prep_upd["disease_cluster"] = df_events_prep_upd["disease_cluster"].apply(lambda x: eval(x))

    df_events_with_result_column, df_cascades = estimate_cascades_with_st_clustering(
        df_events_prep_upd, eps1=0.1, eps2=24 * 10)
    df_cascades = df_cascades[df_cascades["size"] >= MIN_CASCADE_SIZE]
    cascades_filepath = os.path.join(output_cascades_folder, "cascades_from_st_clustering_all_serotypes.csv")
    df_cascades.to_csv(cascades_filepath, sep=";", index=False)


    clusters = set()
    for cl in df_events_prep_upd["disease_cluster"].tolist():
        for c in cl:
            clusters.add(c)

    nb_clusters = len(clusters)

    for cluster_id in range(nb_clusters):
        print("--- cluster id:", cluster_id)
        df_events_prep_upd["disease_cluster_"+str(cluster_id)] = df_events_prep_upd["disease_cluster"].apply(lambda x: 1 if cluster_id in x else 0)
        df_events_by_serotype = df_events_prep_upd[df_events_prep_upd["disease_cluster_"+str(cluster_id)] == 1].copy(deep=True)
        del df_events_by_serotype["disease_cluster_"+str(cluster_id)]
        del df_events_prep_upd["disease_cluster_" + str(cluster_id)]

        if df_events_by_serotype.shape[0]>2:
            df_events_by_serotype_with_result_column, df_cascades = estimate_cascades_with_st_clustering(
                                                                    df_events_by_serotype, eps1=0.1, eps2=24 * 10)

            df_cascades = df_cascades[df_cascades["size"] >= MIN_CASCADE_SIZE]


            serotype = None
            for i in df_events_by_serotype["disease"].tolist():
                serotype = build_disease_instance(i).get_disease_data()["serotype"]
                if serotype != "unknown serotype":
                    break

            print(serotype)
            cascades_filepath = os.path.join(output_cascades_folder, "cascades_from_st_clustering_disease="+serotype+".csv")
            df_cascades.to_csv(cascades_filepath, sep=";", index=False)
