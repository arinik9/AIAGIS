
import pandas as pd
import numpy as np
import csv
import os
import json

import src.consts as consts
from src.genome_preprocessing.genome_similarity import (calculate_isolate_raw_distance, calculate_genome_distance_with_imprecise_genome_info,
                                          similarity_smoothing_with_quality_scores)
from src.util_event import build_disease_instance

import multiprocessing
from itertools import repeat
import dateutil.parser as parser
from statistics import geometric_mean


# X = "atggaagcaatatcactgatgattatactactagtagtgacaacaagcaatgcagacaaaatctgcatcggccaccaatcaacaaattccacagaaaccgtggacacgctaacagaatccagtattcctgtgacacaagccaaagagttgctccacacaaaacaggatggaatgctgtgtgcaacaaatctgggacgtcccctcattctggacacatgcactgtcgaagggctgatttatggcaacccttcttgtgatcttctattgggaggaagagaatggtcatacatcgttgaaagaccatcagcggttaacggaacatgttaccctgggagtgtagaaaacttagaggaactcagaatgctttttagttctgctagttcttaccaaagaatccaaatcttcccagacgcaatctggaatgtgacttacgatgggacaagcaaatcatgctcaaattcgttctacaggaatatgagatggctaactcaaaagaatggaaattatcctattcaagacgcccaatacacaaacaaccgggggaaggacattctcttcatttggggcatacatcatccacccactgatactgcacagacgaatttatacacaagaaccgacacgacaacaagcgtcacaacggaaaatctggacaggaccttcaaaccattaataggaccaagacctcttgtcaatggtctaattggaagaattaattattattggtcagtattaaagccaggacagacgttgagagtaagatccaatgggaatctaatt"
# Y = "agcaaaagcagggtagataatcactcactgagtgacatccacatcatggcgtctcaaggcaccaaacgatcttatgagcagatggaaactggtggagagcgccagaatgctactgagatcagagcatctgttgggagaatggtcggtggaattgggagattctacatacagatgtgcactgagctcaaactcagcgactatgaagggagactaatccaaaacagcataacaatagagagaatggttctctctgcatttgatgaaagaaggaacaaatatctggaagaacaccccagtgcggggaaggatccgaagaaaaccggaggtccaatctacagaaggagagacggaaagtgggtgagggagctaattctgtatgataaagaggagatcaggaggatttggcgccaagcgaacaacggagaagacgcaactgctggtcttactcatctgatgatctggcattccaacctgaatgatgccacatatcagaggacaagagctctcgtgcgtactgggatggaccctagaatgtgctccctgatgcaaggatcaacccttccaaggagatctggagctgctggtgcagcagtaaagggagttgggacaatggtgatggaactgattcggatgataaagcggggaatcaatgacaggaatttctggagaggcgagaatggacggagaacaaggattgcatatgagagaatgtgtaacatcctcaaagggaaattccaaacagcagcacaacgagcaatgatggaccaggtgcgggaaagcagaaatcctggaaatgctgaaatcgaagaccttatcttcctggcacggtctgcactcatcctgagaggatcagtggcccataagtcctgcttgcctgcttgtgtatatggacttgctgtggccagtgggtatgactttgagagagagggatactctctggtcggaattgatcctttccgtctgcttcagaacagccaggtgttcagcctcattagaccaaatgagaatccggcacataagagtcagctggtatggatggcatgccattctgcagcatttgaagacctgagagtatcaagcttcatcagaggaacaagggtggtcccaagaggacaactgtccaccagaggggttcaaatagcttcaaatgagaacatggaaacaatggactccagcactcttgaactgagaagcagatactgggcgataaggaccagaagtggaggaaacaccaaccaacagagagcatctgcaggacaaatcagtgtacagcctactttctcggtacagagaaatctccctttcgagagagcgaccattatggctgcattcacagggaacactgaaggcaggacatccgacatgaggactgaaatcataagaatgatggaaagtgccagaccagaagatgtgtctttccaggggcggggagtcttcgagctctcggacgaaaaggcaacgaacccgatcgtgccttcttttgacatgagtaacgagggatcttatttcttcggagacaatgcagaggagtatgacaattaaagaaaaatacccttgtttctact"
# 
# score = calculate_genome_similarity(X, Y)
# print(score)



def retrieve_distinct_genome_seq_pairs_from_df_events(df_events):
    isolate_pairs_set = set()
    N = df_events.shape[0]
    for i, row1 in df_events.iterrows():
        print(i, "/", N)
        event1_article_id_list = row1["bvbrc_article_id"]
        event1_isolate_list = row1["bvbrc_segment2genomeID"]
        event1_norm_sim_score_list = row1["bvbrc_matching_score"]
        d1 = build_disease_instance(row1["disease"])
        #date1 = parser.parse(row1[consts.COL_PUBLISHED_TIME])
        date1 = row1[consts.COL_PUBLISHED_TIME]
        for j, row2 in df_events.iterrows():
           if i<j:
               event2_article_id_list = row2["bvbrc_article_id"]
               event2_isolate_list = row2["bvbrc_segment2genomeID"]
               event2_norm_sim_score_list = row2["bvbrc_matching_score"]
               d2 = build_disease_instance(row2["disease"])
               #date2 = parser.parse(row2[consts.COL_PUBLISHED_TIME])
               date2 = row2[consts.COL_PUBLISHED_TIME]
               #if d1.hierarchically_includes(d2) or d2.hierarchically_includes(d1) or d1.is_identical(d2):
               if np.abs((date1-date2).days) < 120: # 4 months
                   for i in range(len(event1_isolate_list)):
                       article_id1 = event1_article_id_list[i]
                       isolate1_str = event1_isolate_list[i]
                       #print(event2_isolate_list)
                       for j in range(len(event2_isolate_list)):
                           article_id2 = event2_article_id_list[j]
                           isolate2_str = event2_isolate_list[j]
                           pair = (article_id1, article_id2, isolate1_str, isolate2_str)
                           if date2 < date1:
                               pair = (article_id2, article_id1, isolate2_str, isolate1_str)
                           if pair not in isolate_pairs_set:
                                isolate_pairs_set.add(pair)
    return isolate_pairs_set



# ======================================
# 1) calculate the similarity scores for each pair of sequences of the same strain (e.g. H5N1 with H5N1)
# ======================================
def calculate_genome_raw_dist_scores(params):
    id = params[0]
    #print(id)
    isolate_pairs_list = params[1]
    bvbrcId2seq = params[2]
    # N = len(seq_pairs_list)

    #seq1_list = []
    article_id1_list = []
    #seq2_list = []
    article_id2_list = []
    raw_dist_score_list = []

    for i, pair_tuple in enumerate(isolate_pairs_list):
        (article_id1, article_id2, isolate1_str, isolate2_str) = pair_tuple
        #print(i, "/", N)
        raw_dist_score = calculate_isolate_raw_distance(json.loads(isolate1_str), json.loads(isolate2_str), bvbrcId2seq)
        raw_dist_score_list.append(raw_dist_score)
        #smoothed_dist_score = similarity_smoothing_with_quality_scores(raw_dist_score, float(norm_score1), float(norm_score2))
        #seq1_list.append(seq1)
        #seq2_list.append(seq2)
        article_id1_list.append(article_id1)
        article_id2_list.append(article_id2)

    df = pd.DataFrame(list(zip(article_id1_list, article_id2_list, raw_dist_score_list)),
                      columns = ["bvbrc_article_id1", "bvbrc_article_id2", "raw_dist_score"])
    #print(df.shape)
    #df.sort_values(by=['score', "seq1_disease", "seq2_disease"], ascending=True, inplace=True)
    return df


def calculate_genome_raw_dist_scores_in_sequantial(df_events):
    ##list_grouped = df_events.groupby(["continent", "season"])
    ##dict_groupes = dict(zip(range(len(list_grouped)), list_grouped))
    ##print(dict_groupes)
    #list_grouped = (0, df_events)
    #df_result = calculate_genome_raw_dist_scores(list_grouped)
    ##df_result.sort_values(by=['score', "seq1_disease", "seq2_disease"], ascending=True, inplace=True)
    #return df_result
    pass

def calculate_genome_raw_dist_scores_in_parallel(isolate_pairs_set, bvbrcId2seq, nb_processes=8):
    list_grouped = [[] for i in range(nb_processes)]
    for i, pair_tuple in enumerate(isolate_pairs_set):
        list_grouped[i % nb_processes].append((pair_tuple))
    list_grouped = [(i, l, bvbrcId2seq) for i, l in enumerate(list_grouped)]

    with multiprocessing.Pool(processes=nb_processes) as pool:
        # call the function for each item in parallel
        list_result = pool.map(calculate_genome_raw_dist_scores, list_grouped)
        df_result = pd.concat(list_result)
        #df_result.sort_values(by=['seq1_bvbrc_id', 'seq1', 'seq2_bvbrc_id', 'seq2', 'score'], ascending=True, inplace=True)
        return df_result


def calculate_genome_dist_scores_with_missing_info_in_parallel(df_events, in_db_isolate_sim_dir, nb_processes=8):
    entry_list = []
    N = df_events.shape[0]
    for i, row1 in df_events.iterrows():
        print(i, "/", N)
        event1_id = row1["id"]
        event1_country_code = json.loads(row1["geoname_json"])['countryCode'] # alpha2
        event1_genome_name_list = row1["bvbrc_genome_name"]
        d1 = build_disease_instance(row1["disease"])
        date1 = row1[consts.COL_PUBLISHED_TIME]
        for j, row2 in df_events.iterrows():
            if i < j:
                event2_id = row2["id"]
                event2_country_code = json.loads(row2["geoname_json"])['countryCode'] # alpha2
                event2_genome_name_list = row2["bvbrc_genome_name"]
                d2 = build_disease_instance(row2["disease"])
                date2 = row2[consts.COL_PUBLISHED_TIME]

                #if d1.hierarchically_includes(d2) or d2.hierarchically_includes(d1) or d1.is_identical(d2):
                if np.abs((date1 - date2).days) < 120:  # 4 months
                    # -----------------------------------------------------
                    if len(event1_genome_name_list) == 0 or len(event2_genome_name_list) == 0:
                        entry = (event1_id, event2_id, event1_genome_name_list, event2_genome_name_list,
                                event1_country_code, event2_country_code,
                                d1, d2
                                )
                        entry_list.append(entry)

    # enrty1 = (887, 2079, [], ['A/Alopochen aegyptiaca/Belgium/3237_B6663/2017'], 'KR', 'FR', build_disease_instance("['h5n2','h5','ai (lpai)']"), build_disease_instance("['h5n2','h5','ai (lpai)']"))
    # entry2 = (1107, 2334, [], ['A/duck/Jiangxi/4.30_NCNP85N2-OC/2017'], 'ZA', 'TW', build_disease_instance("['h5n8','h5','ai (hpai)']"), build_disease_instance("['h5n8','h5','ai (hpai)']"))
    # entry_list.append(enrty1)
    # entry_list.append(entry2)

    nb_processes = 8
    list_grouped = [[] for i in range(nb_processes)]
    for i, entry in enumerate(entry_list):
        list_grouped[i % nb_processes].append((entry))
    list_grouped = [(i, l, in_db_isolate_sim_dir) for i, l in enumerate(list_grouped)]

    with multiprocessing.Pool(processes=nb_processes) as pool:
        # call the function for each item in parallel
        list_result = pool.map(calculate_genome_distance_with_missing_genome_info_for_list, list_grouped)
        df_result = pd.concat(list_result)
        #df_result.sort_values(by=['seq1_bvbrc_id', 'seq1', 'seq2_bvbrc_id', 'seq2', 'score'], ascending=True, inplace=True)
        return df_result


def calculate_genome_distance_with_missing_genome_info_for_list(params):
    i = params[0]
    entry_list = params[1]
    db_isolate_sim_dir = params[2]


    event1_id_list = []
    event2_id_list = []
    event1_country_code_list = []
    event2_country_code_list = []
    disease1_list = []
    disease2_list = []
    dist_score_list = []
    for entry in entry_list:
        event1_id = entry[0]
        event2_id = entry[1]
        event1_genome_name_list = entry[2]
        event2_genome_name_list = entry[3]
        event1_country_code = entry[4]
        event2_country_code = entry[5]
        d1 = entry[6]
        d2 = entry[7]
        sim_score = calculate_genome_distance_with_missing_genome_info(db_isolate_sim_dir,
            event1_genome_name_list, event2_genome_name_list,
            event1_country_code, event2_country_code,
            d1, d2
        )
        dist_score = 1 - sim_score
        event1_id_list.append(event1_id)
        event2_id_list.append(event2_id)
        event1_country_code_list.append(event1_country_code)
        event2_country_code_list.append(event2_country_code)
        disease1_list.append(str(d1))
        disease2_list.append(str(d2))
        dist_score_list.append(dist_score)

    df = pd.DataFrame(list(zip(event1_id_list, event2_id_list, event1_country_code_list, event2_country_code_list,
                               disease1_list, disease2_list, dist_score_list)),
                      columns = ["event1_id", "event2_id", "event1_country_code", "event2_country_code",
                                 "disease1", "disease2", "dist_score"])
    return df


def calculate_genome_distance_with_missing_genome_info(db_isolate_sim_dir,
                            event1_genome_name_list, event2_genome_name_list,
                            event1_country_code, event2_country_code,
                            d1, d2
                        ):
    d1_serotype = d1.d_serotype
    if d1_serotype == "unknown serotype":
        d1_serotype = d1.d_subtype
    if d1_serotype == "unknown subtype":
        #sdf()
        return np.nan

    d2_serotype = d2.d_serotype
    if d2_serotype == "unknown serotype":
        d2_serotype = d2.d_subtype
    if d2_serotype == "unknown subtype":
        #sdf()
        return np.nan

    #db_isolate_sim_dir = "/Users/narinik/Mirror/workspace/MulNetFer/in/external_data/isolate_avg_sim_scores"
    db_default_fpath = os.path.join(db_isolate_sim_dir, "genome_sim_summary.csv")
    df_default = pd.read_csv(db_default_fpath, sep=";", keep_default_na=False)
    df_default.loc[df_default["avg_sim_score"] == '', "avg_sim_score"] = 0.0 # some serovar pairs do not have any values. For instance; "h5n4_h2", "h5n4_h7", "h5n4_h10"

    df_default["avg_sim_score"] = df_default["avg_sim_score"].astype(float)
    dict_default_values = dict(zip(df_default["serovar_pair"], df_default["avg_sim_score"]))

    serovar_pair = d1_serotype+"_"+d2_serotype
    db_by_country_fpath = os.path.join(db_isolate_sim_dir, "genome_sim_summary_by_country_"+serovar_pair+".csv")
    db_by_genome_name_fpath = os.path.join(db_isolate_sim_dir, "genome_sim_summary_by_genome_name_" + serovar_pair + ".csv")
    db_by_country_and_genome_name_fpath = os.path.join(db_isolate_sim_dir, "genome_sim_summary_by_country_and_genome_name_" + serovar_pair + ".csv")
    if not os.path.exists(db_by_country_fpath):
        serovar_pair = d2_serotype + "_" + d1_serotype
        db_by_country_fpath = os.path.join(db_isolate_sim_dir, "genome_sim_summary_by_country_" + serovar_pair + ".csv")
        db_by_genome_name_fpath = os.path.join(db_isolate_sim_dir, "genome_sim_summary_by_genome_name_" + serovar_pair + ".csv")
        db_by_country_and_genome_name_fpath = os.path.join(db_isolate_sim_dir, "genome_sim_summary_by_country_and_genome_name_" + serovar_pair + ".csv")
        tmp = d1_serotype
        d1_serotype = d2_serotype
        d2_serotype = tmp
        tmp = event1_genome_name_list
        event1_genome_name_list = event2_genome_name_list
        event2_genome_name_list = tmp
        tmp = event1_country_code
        event1_country_code = event2_country_code
        event2_country_code = tmp
    if not os.path.exists(db_by_country_fpath):
        dsf()
        return np.nan

    # print("event1_genome_name_list", event1_genome_name_list)
    # print("event2_genome_name_list", event2_genome_name_list)
    # print("d1_serotype", d1_serotype)
    # print("d2_serotype", d2_serotype)

    is_same_disease = True
    if d1_serotype != d2_serotype:
        is_same_disease = False
    #print("is_same_disease", is_same_disease)


    sim_score = None
    if len(event1_genome_name_list) == 0 and len(event2_genome_name_list) == 0:
        df_by_country = pd.read_csv(db_by_country_fpath, sep=";", keep_default_na=False)
        desc = event1_country_code+"_"+event2_country_code
        df_sub = df_by_country[df_by_country["desc"] == desc]
        if is_same_disease:
            desc1 = event1_country_code + "_" + event2_country_code
            desc2 = event2_country_code + "_" + event1_country_code
            df_sub = df_by_country[(df_by_country["desc"] == desc1) | (df_by_country["desc"] == desc2)]
        if df_sub.shape[0] > 0:
            sim_score = df_sub["sim_score"].tolist()[0]
        else:
            sim_score = dict_default_values[serovar_pair]
    elif (len(event1_genome_name_list) > 0 and len(event2_genome_name_list) == 0) | (len(event1_genome_name_list) == 0 and len(event2_genome_name_list) > 0):
        genome_country_desc_list = []
        genome_desc_list = []
        if len(event1_genome_name_list) > 0 and len(event2_genome_name_list) == 0:
            for event1_genome_name in event1_genome_name_list:
                desc = event1_genome_name + "_" + event2_country_code
                genome_country_desc_list.append(desc)
                genome_desc_list.append(event1_genome_name)
        elif len(event1_genome_name_list) == 0 and len(event2_genome_name_list) > 0:
            for event2_genome_name in event2_genome_name_list:
                desc = event2_genome_name + "_" + event1_country_code
                genome_country_desc_list.append(desc)
                genome_desc_list.append(event2_genome_name)
        #
        df_by_country_and_genome = pd.read_csv(db_by_country_and_genome_name_fpath, sep=";", keep_default_na=False)
        df_by_genome = pd.read_csv(db_by_genome_name_fpath, sep=";", keep_default_na=False)
        #print(event1_genome_name_list)
        sim_score_list = []
        for i in range(len(genome_desc_list)):
            desc = genome_country_desc_list[i]
            df_sub = df_by_country_and_genome[df_by_country_and_genome["desc"] == desc]
            if df_sub.shape[0]>0: # scenario 1
                sim_score = df_sub["sim_score"].tolist()[0]
            else:
                desc = genome_desc_list[i]
                df_sub = df_by_genome[df_by_genome["desc"] == desc]
                if df_sub.shape[0] > 0: # scenario 2
                    sim_score = df_sub["sim_score"].tolist()[0]
                else: # scenario 3
                    sim_score = dict_default_values[serovar_pair]
            sim_score_list.append(sim_score)
        sim_score = np.nanmean(sim_score_list)

    #if not np.isnan(sim_score):
    #print(d1_serotype, event1_country_code, d2_serotype, event2_country_code, sim_score)
    if sim_score == '' or np.isnan(sim_score):
        sdf()
    return sim_score



# def calculate_genome_distance_with_missing_genome_info(
#                             event1_genome_name_list, event2_genome_name_list,
#                             event1_country_code, event2_country_code,
#                             d1, d2
#                         ):
#     d1_serotype = d1.d_serotype
#     if d1_serotype == "unknown serotype":
#         d1_serotype = d1.d_subtype
#     if d1_serotype == "unknown subtype":
#         sdf()
#         return np.nan
#
#     d2_serotype = d2.d_serotype
#     if d2_serotype == "unknown serotype":
#         d2_serotype = d2.d_subtype
#     if d2_serotype == "unknown subtype":
#         sdf()
#         return np.nan
#
#     db_isolate_sim_dir = "/Users/narinik/Mirror/workspace/genome2event/out/preprocessing/BVBRC/analysis"
#     db_fpath = os.path.join(db_isolate_sim_dir, "genome_sim_analysis_"+d1_serotype+"_"+d2_serotype+".csv")
#     if not os.path.exists(db_fpath):
#         db_fpath = os.path.join(db_isolate_sim_dir, "genome_sim_analysis_" + d2_serotype + "_" + d1_serotype + ".csv")
#         tmp = d1_serotype
#         d1_serotype = d2_serotype
#         d2_serotype = tmp
#         tmp = event1_genome_name_list
#         event1_genome_name_list = event2_genome_name_list
#         event2_genome_name_list = tmp
#         tmp = event1_country_code
#         event1_country_code = event2_country_code
#         event2_country_code = tmp
#     if not os.path.exists(db_fpath):
#         dsf()
#         return np.nan
#
#     # print("event1_genome_name_list", event1_genome_name_list)
#     # print("event2_genome_name_list", event2_genome_name_list)
#     # print("d1_serotype", d1_serotype)
#     # print("d2_serotype", d2_serotype)
#
#     is_same_disease = True
#     if d1_serotype != d2_serotype:
#         is_same_disease = False
#     #print("is_same_disease", is_same_disease)
#
#     df_sim = pd.read_csv(db_fpath, sep=";", keep_default_na=False)
#
#     sim_score = None
#     if len(event1_genome_name_list) == 0 and len(event2_genome_name_list) == 0:
#         sim_score = retrieve_isolate_mean_sim_score_by_country(df_sim, event1_country_code, event2_country_code)
#     elif len(event1_genome_name_list) > 0 and len(event2_genome_name_list) == 0:
#         #print(event1_genome_name_list)
#         sim_score_list = []
#         for event1_genome_name in event1_genome_name_list:
#             df_sub = df_sim[df_sim["source_genome_name"] == event1_genome_name]
#             #print(df_sub.shape)
#             if is_same_disease:
#                 df_sub = df_sim[(df_sim["source_genome_name"] == event1_genome_name) | (df_sim["target_genome_name"] == event1_genome_name)]
#             if df_sub.shape[0]>0:
#                 sim_score = retrieve_isolate_mean_sim_score_by_country(df_sub, event1_country_code, event2_country_code)
#             else:
#                 #sim_score = np.nan
#                 sim_score = retrieve_isolate_mean_sim_score_by_country(df_sim, event1_country_code, event2_country_code)
#                 print("---- scenario2 with error")
#                 print("event1_genome_name_list", event1_genome_name_list)
#                 print("event2_genome_name_list", event2_genome_name_list)
#                 print("d1_serotype", d1_serotype)
#                 print("d2_serotype", d2_serotype)
#                 #sdf()
#             sim_score_list.append(sim_score)
#         sim_score = np.nanmean(sim_score_list)
#     elif len(event1_genome_name_list) == 0 and len(event2_genome_name_list) > 0:
#         sim_score_list = []
#         for event2_genome_name in event2_genome_name_list:
#             df_sub = df_sim[df_sim["target_genome_name"] == event2_genome_name]
#             #print(1, df_sub.shape)
#             if is_same_disease:
#                 df_sub = df_sim[(df_sim["target_genome_name"] == event2_genome_name) | (df_sim["source_genome_name"] == event2_genome_name)]
#                 #print(2, df_sub.shape)
#             if df_sub.shape[0]>0:
#                 sim_score = retrieve_isolate_mean_sim_score_by_country(df_sub, event1_country_code, event2_country_code)
#             else:
#                 #sim_score = np.nan
#                 sim_score = retrieve_isolate_mean_sim_score_by_country(df_sim, event1_country_code, event2_country_code)
#                 print("---- scenario3 with error")
#                 print("event1_genome_name_list", event1_genome_name_list)
#                 print("event2_genome_name_list", event2_genome_name_list)
#                 print("d1_serotype", d1_serotype)
#                 print("d2_serotype", d2_serotype)
#                 #sdf()
#             sim_score_list.append(sim_score)
#         sim_score = np.nanmean(sim_score_list)
#
#     #if not np.isnan(sim_score):
#     #print(d1_serotype, event1_country_code, d2_serotype, event2_country_code, sim_score)
#     if sim_score == '' or np.isnan(sim_score):
#         sdf()
#     return sim_score
#
#
# def retrieve_isolate_mean_sim_score_by_country(df, event1_country_code, event2_country_code):
#     df_res = df[(df["source_country_code"] == event1_country_code) & (df["target_country_code"] == event2_country_code)]
#     if df_res.shape[0]>0:
#         return np.mean(df_res["sim_score"].tolist())
#     #return np.nan
#     return np.mean(df["sim_score"].tolist())



# ======================================
# 1) calculate the similarity scores for each pair of sequences of the same or similar strain (e.g. H5N1 with H5N1 or H5)
# ======================================
def calculate_genome_dist_scores(df_events, df_genome_raw_dist_scores, df_genome_dist_scores_with_missing_info):
    df_genome_raw_dist_scores["pair_id"] = df_genome_raw_dist_scores["bvbrc_article_id1"].apply(str) + "_" + df_genome_raw_dist_scores["bvbrc_article_id2"].apply(str)
    articlePair2distScore = dict(zip(df_genome_raw_dist_scores["pair_id"], df_genome_raw_dist_scores["raw_dist_score"]))

    df_genome_dist_scores_with_missing_info["pair_id"] = df_genome_dist_scores_with_missing_info["event1_id"].apply(str) + "_" + df_genome_dist_scores_with_missing_info["event2_id"].apply(str)
    eventPair2distScore = dict(zip(df_genome_dist_scores_with_missing_info["pair_id"], df_genome_dist_scores_with_missing_info["dist_score"]))

    N = df_events.shape[0]

    seq1_event_id_list = []
    seq2_event_id_list = []
    seq1_bvbrc_article_id_list = []
    seq2_bvbrc_article_id_list = []
    dist_score_list = []


    for i, row1 in df_events.iterrows():
        print(i, "/", N)
        #print(row1)
        event1_id = row1["id"]
        #event1_country_code = row1["loc_country_code"] # alpha3
        event1_country_code = json.loads(row1["geoname_json"])['countryCode'] # alpha2
        event1_bvbrc_article_id_list = row1["bvbrc_article_id"]
        event1_genome_name_list = row1["bvbrc_genome_name"]
        event1_genome_quality_scores = row1["bvbrc_matching_score"]
        event1_genome_dist_values = row1["bvbrc_delta_dist"]
        event1_genome_temp_values = row1["bvbrc_delta_temp"]
        d1 = build_disease_instance(row1["disease"])
        #date1 = parser.parse(row1[consts.COL_PUBLISHED_TIME])
        date1 = row1[consts.COL_PUBLISHED_TIME]
        for j, row2 in df_events.iterrows():
            if i < j:
                event2_id = row2["id"]
                #event2_country_code = row2["loc_country_code"] # alpha3
                event2_country_code = json.loads(row2["geoname_json"])['countryCode'] # alpha2
                event2_bvbrc_article_id_list = row2["bvbrc_article_id"]
                event2_genome_name_list = row2["bvbrc_genome_name"]
                event2_genome_quality_scores = row2["bvbrc_matching_score"]
                event2_genome_dist_values = row2["bvbrc_delta_dist"]
                event2_genome_temp_values = row2["bvbrc_delta_temp"]
                d2 = build_disease_instance(row2["disease"])
                #date2 = parser.parse(row2[consts.COL_PUBLISHED_TIME])
                date2 = row2[consts.COL_PUBLISHED_TIME]

                #if d1.hierarchically_includes(d2) or d2.hierarchically_includes(d1) or d1.is_identical(d2):
                if np.abs((date1 - date2).days) < 120:  # 4 months
                    # -----------------------------------------------------
                    dist_score = np.nan
                    if len(event1_bvbrc_article_id_list) == 0 or len(event2_bvbrc_article_id_list) == 0:
                        if str(event1_id)+"_"+str(event2_id) in eventPair2distScore:
                            dist_score = eventPair2distScore[str(event1_id)+"_"+str(event2_id)]
                        elif str(event2_id)+"_"+str(event1_id) in eventPair2distScore:
                            dist_score = eventPair2distScore[str(event2_id)+"_"+str(event1_id)]
                        if dist_score == '':
                            print("error with missing info", event1_id, event2_id)
                            sdf()
                        #    dist_score = float(dist_score)
                        #print(dist_score, dist_score != '')
                        #print(str(event1_id)+"_"+str(event2_id), dist_score)

                        #print(str(event1_id)+"_"+str(event2_id), dist_score)
                        # sim_score = calculate_genome_distance_with_missing_genome_info(
                        #     event1_bvbrc_article_id_list, event2_bvbrc_article_id_list,
                        #     event1_genome_name_list, event2_genome_name_list,
                        #     event1_country_code, event2_country_code,
                        #     d1, d2
                        # )
                        #print("sim score", sim_score)
                    else:
                        if len(event1_bvbrc_article_id_list)>0 and len(event1_genome_dist_values)==0:
                            event1_genome_dist_values = [1]*len(event1_bvbrc_article_id_list)
                            event1_genome_temp_values = [1] * len(event1_bvbrc_article_id_list)
                        if len(event2_bvbrc_article_id_list)>0 and len(event2_genome_dist_values)==0:
                            event2_genome_dist_values = [1]*len(event2_bvbrc_article_id_list)
                            event2_genome_temp_values = [1] * len(event2_bvbrc_article_id_list)
                        dist_score = calculate_genome_distance_with_imprecise_genome_info(articlePair2distScore,
                                               event1_bvbrc_article_id_list, event2_bvbrc_article_id_list,
                                               event1_genome_quality_scores, event2_genome_quality_scores,
                                               event1_genome_dist_values, event2_genome_dist_values,
                                               event1_genome_temp_values, event2_genome_temp_values)
                        #print("dist", dist_score)
                        if dist_score == '' or np.isnan(dist_score):
                            print("error with imprecise genome", event1_id, event2_id)
                            sdf()
                    seq1_event_id_list.append(event1_id)
                    seq2_event_id_list.append(event2_id)
                    seq1_bvbrc_article_id_list.append(str(event1_bvbrc_article_id_list))
                    seq2_bvbrc_article_id_list.append(str(event2_bvbrc_article_id_list))
                    dist_score_list.append(dist_score)
                    # -----------------------------------------------------

    df = pd.DataFrame(list(zip(seq1_event_id_list, seq2_event_id_list,
                               seq1_bvbrc_article_id_list, seq2_bvbrc_article_id_list,
                               dist_score_list)),
                              columns = ["event1_id", "event2_id", "bvbrc_article_id1",
                                         "bvbrc_article_id2", "dist_score"])
    print(df)
    #df.sort_values(by=['dist_score'], ascending=True, inplace=True)

    return df

def create_genome_dist_matrix_from_events(df_events, input_genome_dist_scores_filepath, output_genome_dist_matrix_filepath):
    cols = ["event1_id", "event2_id", "dist_score",]
    df_result = pd.read_csv(input_genome_dist_scores_filepath, usecols=cols, sep=";", keep_default_na=False)
    df_result["pair_id"] = df_result["event1_id"].apply(str) + "_" + df_result["event2_id"].apply(str)
    df_result.set_index("pair_id", inplace=True)
    print(df_result.index)

    id_list = df_events[consts.COL_ID].to_numpy().flatten()
    N = len(id_list)
    D = np.full(shape=(N,N), fill_value=np.nan)

    for i, row1 in df_events.iterrows():
        print(i, "/", N, "!")
        event1_id = row1[consts.COL_ID]
        date1 = row1[consts.COL_PUBLISHED_TIME]
        #date1 = parser.parse(row1[consts.COL_PUBLISHED_TIME])
        #d1 = build_disease_instance(row1["disease"])
        for j, row2 in df_events.iterrows():
           if i<j:
               event2_id = row2[consts.COL_ID]
               #date2 = parser.parse(row2[consts.COL_PUBLISHED_TIME])
               date2 = row2[consts.COL_PUBLISHED_TIME]
               #d2 = build_disease_instance(row2["disease"])
               #if d1.hierarchically_includes(d2) or d2.hierarchically_includes(d1) or d1.is_identical(d2):
               if np.abs((date1 - date2).days) < 120:  # 4 months
                   pair_id1 = str(event1_id)+"_"+str(event2_id)
                   pair_id2 = str(event2_id) + "_" + str(event1_id)
                   if pair_id1 in df_result.index and df_result.loc[pair_id1,"dist_score"] != "":
                       score = df_result.loc[pair_id1,"dist_score"]
                       D[i, j] = score
                       D[j, i] = score
                   elif pair_id2 in df_result.index and df_result.loc[pair_id2, :] != "":
                       score = df_result.loc[pair_id2, :]
                       D[i, j] = score
                       D[j, i] = score
                   else:
                       D[i, j] = np.nan
                       D[j, i] = np.nan
    df = pd.DataFrame(D)
    df.index = id_list
    df.columns = id_list
    df.to_csv(output_genome_dist_matrix_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)



def create_genome_dist_matrix_from_map(df_world_map_info_nz, df_genome_matrix_from_events, df_events, output_genome_dist_matrix_filepath):
    id_list = df_world_map_info_nz["gn_id"].to_numpy().flatten()
    N = len(id_list)

    geonamesId2EventId = {}
    for i, row in df_events.iterrows():
        #print(i, "/", N)
        id = row["id"]
        gn_id = row["ADM1_geonameid"]
        bvbrc_article_id = row["bvbrc_article_id"]
        #if len(bvbrc_article_id)>0:
        if gn_id not in geonamesId2EventId:
            geonamesId2EventId[gn_id] = []
        geonamesId2EventId[gn_id].append(id)

    D = np.full(shape=(N,N), fill_value=np.nan)
    for i in range(N): #for index, row in df_map.iterrows():
        gn_id1 = id_list[i]
        if gn_id1 in geonamesId2EventId:
            event_ids1 = geonamesId2EventId[gn_id1]
            D[i, i] = 0.0
            for j in range(N):
                gn_id2 = id_list[j]
                if i<j and gn_id2 in geonamesId2EventId:
                    event_ids2 = geonamesId2EventId[gn_id2]
                    dists = []
                    for eid1 in event_ids1:
                        for eid2 in event_ids2:
                            # the index values of 'df_genome_matrix_from_events' are integer
                            # the column values of 'df_genome_matrix_from_events' are string
                            val = df_genome_matrix_from_events.loc[eid1, str(eid2)]
                            if val != '':
                                dists.append(float(val))
                    if len(dists)>0:
                        dist = np.nanmean(np.array(dists))
                        D[i,j] = dist
                        D[j,i] = dist

    df = pd.DataFrame(D)
    df.index = id_list
    df.columns = id_list
    df.to_csv(output_genome_dist_matrix_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)





# ======================================
# MAIN FUNCTION
# ======================================
def perform_genome_preprocessing(df_events, df_bvbrc_seqs, in_db_isolate_sim_dir,
                                 df_world_map_info_nz, output_genome_dist_matrix_from_events_filepath,
                                 output_genome_dist_matrix_from_map_filepath,
                                 output_genome_raw_dist_scores_from_events_filepath,
                                 output_genome_dist_scores_from_events_with_missing_info_filepath,
                                 output_genome_dist_scores_from_events_filepath, force=False):
    bvbrcId2seq = dict(zip(df_bvbrc_seqs["id"].astype(str), df_bvbrc_seqs["seq"]))

    if (not os.path.exists(output_genome_raw_dist_scores_from_events_filepath)) or force:
        isolate_pairs_set = retrieve_distinct_genome_seq_pairs_from_df_events(df_events)

        #df_result = calculate_genome_dist_scores_in_sequantial(seq_pairs_set)
        df_result = calculate_genome_raw_dist_scores_in_parallel(isolate_pairs_set, bvbrcId2seq)
        df_result.to_csv(output_genome_raw_dist_scores_from_events_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC, index=False)

    if (not os.path.exists(output_genome_dist_scores_from_events_with_missing_info_filepath)) or force:
        print("girdi2")
        df_result = calculate_genome_dist_scores_with_missing_info_in_parallel(df_events, in_db_isolate_sim_dir)
        df_result.to_csv(output_genome_dist_scores_from_events_with_missing_info_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC, index=False)


    if (not os.path.exists(output_genome_dist_scores_from_events_filepath)) or force:
        df_genome_raw_results = pd.read_csv(output_genome_raw_dist_scores_from_events_filepath, sep=";", keep_default_na=False)
        df_genome_dist_scores_with_missing_info = pd.read_csv(output_genome_dist_scores_from_events_with_missing_info_filepath, sep=";", keep_default_na=False)
        df_result = calculate_genome_dist_scores(df_events, df_genome_raw_results, df_genome_dist_scores_with_missing_info)
        df_result.to_csv(output_genome_dist_scores_from_events_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC, index=False)


    if (not os.path.exists(output_genome_dist_matrix_from_events_filepath)) or force:
        # create a dist matrix from the calculated sim scores (from the file "output_genome_dist_scores_filepath")
        create_genome_dist_matrix_from_events(df_events, output_genome_dist_scores_from_events_filepath, output_genome_dist_matrix_from_events_filepath)

    # if (not os.path.exists(output_genome_dist_matrix_from_map_filepath)) or force:
    #     # create a dist matrix from the calculated sim scores (from the file "output_genome_dist_scores_filepath")
    #     df_genome_matrix_from_events = pd.read_csv(output_genome_dist_matrix_from_events_filepath, sep=";", keep_default_na=False, index_col=0)
    #     create_genome_dist_matrix_from_map(df_world_map_info_nz, df_genome_matrix_from_events, df_events, output_genome_dist_matrix_from_map_filepath)




# # nothing to update with df_events, return the same object
    # df_events.reset_index(drop=True, inplace=True)
    # return df_events


if __name__ == '__main__':
    print('Starting')
    in_map_folder = consts.IN_MAP_SHAPEFILE_FOLDER
    world_map_csv_filepath = os.path.join(in_map_folder, "world", "ne_10m_admin_1_states_provinces", "naturalearth_adm1.csv")
    df_world_map_info = pd.read_csv(world_map_csv_filepath, usecols=["gn_id", "name", "admin", "lon", "lat"], sep=";",
                         keep_default_na=False)
    #df_world_map_info_nz = df_world_map_info[df_world_map_info["gn_id"] != -1]
    df_world_map_info_nz = df_world_map_info

    output_preprocessing_folder = os.path.join(consts.OUT_FOLDER, "preprocessing")
    events_filepath = os.path.join(output_preprocessing_folder, "processed_empres-i_events_updated.csv") # only 2021 data
    df_events_prep_upd = pd.read_csv(events_filepath, sep=";", keep_default_na=False)
    df_events_prep_upd["bvbrc_article_id"] = df_events_prep_upd["bvbrc_article_id"].apply(lambda x: eval(x))
    df_events_prep_upd["bvbrc_matching_score"] = df_events_prep_upd["bvbrc_matching_score"].apply(lambda x: eval(x))
    df_events_prep_upd["bvbrc_segment2GenomeID"] = df_events_prep_upd["bvbrc_segment2GenomeID"].apply(lambda x: eval(x))
    df_events_prep_upd["bvbrc_segment2GenomeID"] = df_events_prep_upd["bvbrc_segment2GenomeID"].apply(lambda l: [json.loads(i) for i in l])
    df_events_prep_upd["bvbrc_delta_dist"] = df_events_prep_upd["bvbrc_delta_dist"].apply(lambda x: eval(x))
    df_events_prep_upd["bvbrc_delta_temp"] = df_events_prep_upd["bvbrc_delta_temp"].apply(lambda x: eval(x))
    #df_events_prep_upd = df_events_prep_upd.iloc[:100, :]


    output_genome_dist_matrix_from_events_filepath = os.path.join(output_preprocessing_folder, "genome_dist_matrix_from_events.csv")
    output_genome_dist_matrix_from_map_filepath = os.path.join(output_preprocessing_folder, "genome_dist_matrix_from_map.csv")
    output_genome_raw_dist_scores_filepath = os.path.join(output_preprocessing_folder, "genome_raw_dist_scores.csv")
    output_genome_dist_scores_filepath = os.path.join(output_preprocessing_folder, "genome_dist_scores.csv")
    perform_genome_preprocessing(df_events_prep_upd, output_genome_dist_matrix_from_events_filepath,
                                 output_genome_dist_matrix_from_map_filepath,
                                 output_genome_dist_scores_filepath, output_genome_raw_dist_scores_filepath)