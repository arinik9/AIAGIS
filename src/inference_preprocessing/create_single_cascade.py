
import os
import src.consts as consts
import dateutil.parser as parser
import pandas as pd
from src.util_event import build_disease_instance



def retrieve_single_cascade(df_events, out_cascades_filepath=None):
    df_events.sort_values(by=['ADM1_geonameid', consts.COL_PUBLISHED_TIME], inplace=True)
    df_events_grouped = df_events.groupby(['ADM1_geonameid']).agg(
                                    {
                                      consts.COL_PUBLISHED_TIME: 'first',
                                      'id' : 'first'
    }).reset_index()
    df_events_grouped.sort_values(by=[consts.COL_PUBLISHED_TIME], inplace=True)
    print(df_events_grouped)
    cascade_data = df_events_grouped["id"].astype(str).tolist()
    print(cascade_data)
    cascade_str = ','.join(cascade_data)

    df_cascades = pd.DataFrame({'cascade': [cascade_str], 'size': [len(cascade_data)]})
    if out_cascades_filepath is not None:
        df_cascades.to_csv(out_cascades_filepath, sep=";", index=False)
    return df_cascades


if __name__ == '__main__':
    print('Starting')
    output_preprocessing_folder = os.path.join(consts.OUT_FOLDER, "preprocessing")
    output_cascades_folder = os.path.join(output_preprocessing_folder, "single_cascade")
    try:
        if not os.path.exists(output_cascades_folder):
          os.makedirs(output_cascades_folder)
    except OSError as err:
        print(err)

    events_filepath = os.path.join(output_preprocessing_folder, "processed_empres-i_events.csv")  # only 2021 data
    df_events_prep_upd = pd.read_csv(events_filepath, sep=";", keep_default_na=False)

    print(df_events_prep_upd.shape)
    #df_events_prep_upd = df_events_prep_upd[df_events_prep_upd["disease"].str.contains("h5n8")]
    df_events_prep_upd = df_events_prep_upd[df_events_prep_upd["disease"].str.contains("h7n9")]
    #df_events_prep_upd = df_events_prep_upd[df_events_prep_upd["continent"] == "EU"]
    #df_events_prep_upd = df_events_prep_upd[df_events_prep_upd["lng"] < 27.0]
    print(df_events_prep_upd.shape)


    df_events_prep_upd[consts.COL_PUBLISHED_TIME] = df_events_prep_upd[consts.COL_PUBLISHED_TIME].apply(lambda x: parser.parse(x))
    df_events_prep_upd["disease_cluster"] = df_events_prep_upd["disease_cluster"].apply(lambda x: eval(x))

    # ==================================================================
    # SINGLE CASCADE
    # ==================================================================

    out_cascades_filepath = os.path.join(output_cascades_folder, "single_cascade.csv")
    cascade_data = retrieve_single_cascade(df_events_prep_upd, out_cascades_filepath)

    # # ==================================================================
    # # CASCADES BY FLYWYAY
    # # ==================================================================
    # in_map_folder = consts.IN_MAP_SHAPEFILE_FOLDER
    # world_map_csv_filepath = os.path.join(in_map_folder, "naturalearth_adm1_with_fixed_geometries_and_flyway.csv")
    # map_info = pd.read_csv(world_map_csv_filepath, sep=";", keep_default_na=False)
    # map_info["flyway_info"] = map_info["flyway_info"].apply(lambda x: eval(x) if x != '' else [])
    # geonamesId2flyway = dict(zip(map_info["gn_id"], map_info["flyway_info"]))
    # df_events_prep_upd["flyway_info"] = df_events_prep_upd["ADM1_geonameid"].apply(lambda x: geonamesId2flyway[x] if x in geonamesId2flyway else [])
    # print(df_events_prep_upd["flyway_info"])
    #
    # flyway_list = ['atlantic americas', 'black sea mediterranean', 'central asia',
    #                'east africa - west asia', 'east asian - australasian', 'east atlantic',
    #                'mississippi americas', 'pacific americas']
    # result_list = []
    # for flyway_info in flyway_list:
    #     print(flyway_info)
    #     df_events_prep_upd["is_same_flyway"] = df_events_prep_upd["flyway_info"].apply(
    #         lambda x: 1 if flyway_info in x else 0)
    #     df_events_prep_upd_by_flayway = df_events_prep_upd[df_events_prep_upd["is_same_flyway"] == 1]
    #     print(df_events_prep_upd_by_flayway.shape)
    #     del df_events_prep_upd["is_same_flyway"]
    #     if df_events_prep_upd_by_flayway.shape[0]>0:
    #         df_by_flyway = retrieve_single_cascade(df_events_prep_upd_by_flayway, out_cascades_filepath=None)
    #         result_list.append(df_by_flyway)
    # df_all = pd.concat(result_list)
    # out_cascades_filepath = os.path.join(output_cascades_folder, "single_cascade_by_flyway.csv")
    # df_all.to_csv(out_cascades_filepath, sep=";", index=False)

    # ==================================================================
    # CASCADES BY CLUSTER
    # ==================================================================

    # --------------------------
    clusters = set()
    for cl in df_events_prep_upd["disease_cluster"].tolist():
        for c in cl:
            clusters.add(c)
    nb_clusters = len(clusters)
    # --------------------------

    out_cascades_filepath = os.path.join(output_cascades_folder, "single_cascade_by_serotype.csv")

    cascade_list = []
    for cluster_id in range(nb_clusters):
        df_events_prep_upd["disease_cluster_"+str(cluster_id)] = df_events_prep_upd["disease_cluster"].apply(lambda x: 1 if cluster_id in x else 0)
        df_events_by_serotype = df_events_prep_upd[df_events_prep_upd["disease_cluster_"+str(cluster_id)] == 1].copy(deep=True)
        del df_events_by_serotype["disease_cluster_"+str(cluster_id)]
        del df_events_prep_upd["disease_cluster_" + str(cluster_id)]

        serotype = None
        for i in df_events_by_serotype["disease"].tolist():
            serotype = build_disease_instance(i).get_disease_data()["serotype"]
            if serotype != "unknown serotype":
                break
        print(cluster_id, serotype)
        df_cascade = retrieve_single_cascade(df_events_by_serotype)
        cascade_list.append(df_cascade)
    df_all = pd.concat(cascade_list)
    df_all = df_all[df_all["size"]>=10]
    df_all.to_csv(out_cascades_filepath, sep=";", index=False)

    # # ==================================================================
    # # CASCADES BY CLUSTER and flyway (intersection)
    # # ==================================================================
    #
    # # --------------------------
    # clusters = set()
    # for cl in df_events_prep_upd["disease_cluster"].tolist():
    #     for c in cl:
    #         clusters.add(c)
    # nb_clusters = len(clusters)
    # # --------------------------
    #
    # in_map_folder = consts.IN_MAP_SHAPEFILE_FOLDER
    # world_map_csv_filepath = os.path.join(in_map_folder, "naturalearth_adm1_with_fixed_geometries_and_flyway.csv")
    # map_info = pd.read_csv(world_map_csv_filepath, sep=";", keep_default_na=False)
    # map_info["flyway_info"] = map_info["flyway_info"].apply(lambda x: eval(x) if x != '' else [])
    # geonamesId2flyway = dict(zip(map_info["gn_id"], map_info["flyway_info"]))
    # df_events_prep_upd["flyway_info"] = df_events_prep_upd["ADM1_geonameid"].apply(lambda x: geonamesId2flyway[x] if x in geonamesId2flyway else [])
    # print(df_events_prep_upd["flyway_info"])
    #
    # flyway_list = ['black sea mediterranean', 'atlantic americas', 'central asia',
    #                'east africa - west asia', 'east asian - australasian', 'east atlantic',
    #                'mississippi americas', 'pacific americas']
    # result_list = []
    # for flyway_info in flyway_list:
    #     print(flyway_info)
    #     df_events_prep_upd["is_same_flyway"] = df_events_prep_upd["flyway_info"].apply(
    #         lambda x: 1 if flyway_info in x else 0)
    #     df_events_prep_upd_by_flayway = df_events_prep_upd[df_events_prep_upd["is_same_flyway"] == 1]
    #     for cluster_id in range(nb_clusters):
    #         df_events_prep_upd_by_flayway["disease_cluster_" + str(cluster_id)] = df_events_prep_upd_by_flayway["disease_cluster"].apply(
    #             lambda x: 1 if cluster_id in x else 0)
    #         df_events_by_serotype = df_events_prep_upd_by_flayway[df_events_prep_upd_by_flayway["disease_cluster_" + str(cluster_id)] == 1].copy(
    #             deep=True)
    #         del df_events_prep_upd_by_flayway["disease_cluster_" + str(cluster_id)]
    #
    #         serotype = None
    #         for i in df_events_prep_upd_by_flayway["disease"].tolist():
    #             serotype = build_disease_instance(i).get_disease_data()["serotype"]
    #             if serotype != "unknown serotype":
    #                 break
    #         print(cluster_id, serotype)
    #         df_cascade = retrieve_single_cascade(df_events_by_serotype)
    #         cascade_list.append(df_cascade)
    #
    #
    # st_clustering_folder = output_cascades_folder.replace("single_cascade", "cascades")
    # fpath = os.path.join(st_clustering_folder, "cascades_from_st_clustering_all_serotypes.csv")
    # df = pd.read_csv(fpath, sep=";", keep_default_na=False)
    # cascade_list.append(df)
    #
    # df_all = pd.concat(cascade_list)
    # df_all = df_all[df_all["size"] >= 5]
    #
    # out_filepath = os.path.join(output_cascades_folder, "single_cascade_by_serotype_and_flyway_intersection.csv")
    # df_all.to_csv(out_filepath, sep=";", index=False)
    #
    #
    # # ==================================================================
    # # MIXED: SEROTYPE & FLYWAY (concat)
    # # ==================================================================
    # in1_filepath = os.path.join(output_cascades_folder, "single_cascade_by_flyway.csv")
    # df1 = pd.read_csv(in1_filepath, sep=";", keep_default_na=False)
    # in2_filepath = os.path.join(output_cascades_folder, "single_cascade_by_serotype.csv")
    # df2 = pd.read_csv(in2_filepath, sep=";", keep_default_na=False)
    # out_filepath = os.path.join(output_cascades_folder, "single_cascade_by_serotype_and_flyway_concat.csv")
    # df_all = pd.concat([df1, df2])
    # df_all.to_csv(out_filepath, sep=";", index=False)
    #
    # # ==================================================================
    # # MIXED: st clustering & FLYWAY
    # # ==================================================================
    # from os import listdir
    # from os.path import isfile, join
    #
    # st_clustering_folder = output_cascades_folder.replace("single_cascade", "cascades")
    # onlyfiles = [join(st_clustering_folder, f) for f in listdir(st_clustering_folder) if isfile(join(st_clustering_folder, f))]
    # df_list = []
    # for fpath in onlyfiles:
    #     if "disease" in fpath:
    #         df = pd.read_csv(fpath, sep=";", keep_default_na=False)
    #         df_list.append(df)
    #
    # in_filepath = os.path.join(output_cascades_folder, "single_cascade_by_flyway.csv")
    # df = pd.read_csv(in_filepath, sep=";", keep_default_na=False)
    # df_list.append(df)
    #
    # df_all = pd.concat(df_list)
    # out_filepath = os.path.join(output_cascades_folder, "cascades_flyway_and_st_clustering.csv")
    # df_all.to_csv(out_filepath, sep=";", index=False)