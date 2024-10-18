import numpy as np
import pandas as pd
import json
from src.genome_preprocessing.genome_similarity import calculate_genome_raw_similarity
import csv
import os
import multiprocessing
import time

def calculate_isolate_similarity(isolate1, isolate2, id2seq):
    isolate1_keys = isolate1.keys()
    isolate2_keys = isolate2.keys()
    keys = [value for value in isolate1_keys if value in isolate2_keys]
    genome_state = "complete"
    if len(keys) == 0:
        return 0.0, "zero"
    elif len(keys) < 8:
        genome_state = "partial"

    score_list = []
    for k in keys:
        seq1 = id2seq[isolate1[k].replace("bvbrc", "")]
        seq2 = id2seq[isolate2[k].replace("bvbrc", "")]
        try:
            score = calculate_genome_raw_similarity(seq1, seq2)
            score_list.append(score)
        except ValueError as err:
            # We may have the following error : "ValueError: sequence contains letters not in the alphabet"
            print(err)
    final_score = -1.0
    if len(score_list) > 0:
        final_score = np.mean(np.array(score_list))

    return final_score, genome_state



# Another Error that I got: "OSError: Cannot save file into a non-existent directory: '/Users/narinik/Mirror/workspace/genome2event/out/event_preprocessing/BVBRC/analysis/genome_sim_analysis_H10N3_H11N9'"
def calculate_genome_similarity_for_all_isolates(isolates_raw_filepath, sequences_filepath, out_result_filepath):
    serovar_list_in_empresi = [
        'h10n7', 'h10n8', 'h2n2', 'h3n1', 'h5n1', 'h5n2', 'h5n3',
        'h5n4', 'h5n5', 'h5n6', 'h5n8', 'h5n9', 'h6n2', 'h7n1',
        'h7n2', 'h7n3', 'h7n4', 'h7n6', 'h7n7', 'h7n8', 'h7n9', 'h9n2',
        'h2', 'h3', 'h5','h6', 'h7', 'h9', 'h10'
        #'h2nx', 'h3nx', 'h5nx', 'h6nx', 'h7nx', 'h9nx', 'h10nx'
    ]
    # serovar_list_in_empresi = [
    #     'h5n1', 'h5n5', 'h5n8'
    # ]
    print(isolates_raw_filepath)
    df_isolates = pd.read_csv(isolates_raw_filepath, sep = ";", keep_default_na = False, dtype = str)
    df_isolates["Collection Year"] = df_isolates["Collection Year"].astype(int)
    print(df_isolates)

    #df_isolates = df_isolates[df_isolates["Collection Year"] >= 2012]
    print(df_isolates.shape)
    #df_isolates = df_isolates.iloc[:15,:]
    df_isolates["Serotype"] = df_isolates["Serotype"].apply(lambda x: x.lower())
    df_isolates["Serotype"] = df_isolates["Serotype"].apply(lambda x: x.replace("nx","").strip() if "nx" in x else x)
    df_isolates["Serotype"] = df_isolates["Serotype"].apply(lambda x: x.replace("hx","").strip() if "hx" in x else x)
    serovar_list_in_bvbrc = np.unique(df_isolates["Serotype"])
    serovar_list_in_bvbrc = serovar_list_in_bvbrc[serovar_list_in_bvbrc != '']
    serovar_list_in_bvbrc = serovar_list_in_bvbrc[serovar_list_in_bvbrc != 'unidentified']
    serovar_list_in_bvbrc = serovar_list_in_bvbrc[serovar_list_in_bvbrc != 'unknown']
    print(serovar_list_in_bvbrc)

    serovar_list = [value for value in serovar_list_in_empresi if value in serovar_list_in_bvbrc] # intersection
    print("--")
    print(serovar_list)

    df_isolates["Segment2GenomeID"] = df_isolates["Segment2GenomeID"].apply(lambda x: json.loads(x))
    N = df_isolates.shape[0]

    df_genome_seq = pd.read_csv(sequences_filepath, \
                                usecols=["id", "seq"], sep=";", keep_default_na=False, dtype=str)
    id2seq = dict(zip(df_genome_seq["id"], df_genome_seq["seq"]))

    print("Number of cpu : ", multiprocessing.cpu_count())
    start_time = time.time()

    procs = []
    for i, s1 in enumerate(serovar_list):
        df1 = df_isolates[df_isolates["Serotype"] == s1]
        for j, s2 in enumerate(serovar_list):
            if i <= j:
                df2 = df_isolates[df_isolates["Serotype"] == s2]
                curr_out_result_filepath = out_result_filepath.replace(".csv", "_"+s1+"_"+s2+".csv")
                if not os.path.exists(curr_out_result_filepath):
                    #perform_isolate_similarity_for_serovar_pairs(df1, df2, id2seq, curr_out_result_filepath)
                    # each process perform the similarity task
                    p = multiprocessing.Process(target=perform_isolate_similarity_for_serovar_pairs,
                                                args=(df1, df2, id2seq, curr_out_result_filepath))
                    procs.append(p)
                    p.start()

    for p in procs:
        p.join()

    end_time = time.time()
    print('Batch simulation time: %.2fs' % (end_time - start_time))


def perform_isolate_similarity_for_serovar_pairs(df1, df2, id2seq, out_result_filepath):
    print(out_result_filepath, df1.shape[0], df2.shape[0])
    source_genome_name_list = []
    source_serovar_list = []
    source_collection_date_list = []
    source_country_code_list = []
    target_genome_name_list = []
    target_serovar_list = []
    target_collection_date_list = []
    target_country_code_list = []
    score_list = []
    genome_state_list = []
    for i, rowi in df1.iterrows():
        #print("i", i)
        isolate_i = rowi["Segment2GenomeID"]
        for j, rowj in df2.iterrows():
            if np.abs(rowi["Collection Year"] - rowj["Collection Year"]) <= 1:
                isolate_j = rowj["Segment2GenomeID"]
                score, genome_state = calculate_isolate_similarity(isolate_i, isolate_j, id2seq)
                #if score > 0.70:
                source_genome_name_list.append(rowi["Genome Name"])
                target_genome_name_list.append(rowj["Genome Name"])
                source_serovar_list.append(rowi["Serotype"])
                target_serovar_list.append(rowj["Serotype"])
                source_collection_date_list.append(rowi["ObservationDate"])
                target_collection_date_list.append(rowj["ObservationDate"])
                source_country_code_list.append(rowi["country_code"])
                target_country_code_list.append(rowj["country_code"])
                score_list.append(score)
                genome_state_list.append(genome_state)

    df_result = pd.DataFrame(list(zip(source_genome_name_list, target_genome_name_list,
                                      source_serovar_list, target_serovar_list,
                                      source_collection_date_list, target_collection_date_list,
                                      source_country_code_list, target_country_code_list,
                                      score_list, genome_state_list)),
                 columns=['source_genome_name', 'target_genome_name', 'source_serovar', 'target_serovar',
                          'source_collection_date', 'target_collection_date', 'source_country_code',
                          'target_country_code', 'sim_score', 'genome_state'])
    if df_result.shape[0]>0:
        df_result.to_csv(out_result_filepath, index=False, sep=";", quoting=csv.QUOTE_NONNUMERIC)