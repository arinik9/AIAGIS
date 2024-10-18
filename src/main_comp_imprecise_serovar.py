import os
import consts
import pandas as pd
import csv

from event_matching.event_matching import EventMatching
from event_matching.event_matching_strategy import EventMatchingStrategyEventSimilarity
from util_event import simplify_df_events_at_hier_level1

from util_event import read_df_events

from event_matching.event_db_fusion import EventFusion
from event_matching.event_fusion_strategy import EventFusionStrategyMaxOccurrence
from itertools import combinations

from src.event.event_similarity_strategy import EventSimilarityStrategyManual, EventSimilarityStrategyIsolateGenome
import numpy as np


FORCE = False  # TODO: use it systematically in every main function

if __name__ == '__main__':

    platforms_desc_list = []
    platforms_filepath_dict = {}

    #########################################
    # GENOME BVBRC (imprecise)
    #########################################
    input_event_folder_bvbrc_imprecise = os.path.join(consts.OUT_FOLDER, "doc-events", "bvbrc_imprecise_serovar")
    events_filepath_bvbrc_imprecise = os.path.join(input_event_folder_bvbrc_imprecise, "doc_events_bvbrc_with_imprecise_serovar.csv")

    events_simplified_filepath_bvbrc_imprecise = os.path.join(input_event_folder_bvbrc_imprecise, "events_simplified.csv")  # for DEBUG
    if not os.path.exists(events_simplified_filepath_bvbrc_imprecise):
        simplify_df_events_at_hier_level1(events_filepath_bvbrc_imprecise, events_simplified_filepath_bvbrc_imprecise)  # for DEBUG

    platforms_desc_list.append("bvbrc_imprecise_serovar")
    platforms_filepath_dict["bvbrc_imprecise_serovar"] = events_filepath_bvbrc_imprecise


    #########################################
    # GENOME BVBRC (precise)
    #########################################
    input_event_folder_bvbrc_precise = os.path.join(consts.OUT_FOLDER, "doc-events", "bvbrc_precise_serovar")
    events_filepath_bvbrc_precise = os.path.join(input_event_folder_bvbrc_precise, "doc_events_bvbrc_with_precise_serovar.csv")

    events_simplified_filepath_bvbrc_precise = os.path.join(input_event_folder_bvbrc_precise, "events_simplified.csv")  # for DEBUG
    if not os.path.exists(events_simplified_filepath_bvbrc_precise):
        simplify_df_events_at_hier_level1(events_filepath_bvbrc_precise, events_simplified_filepath_bvbrc_precise)  # for DEBUG

    platforms_desc_list.append("bvbrc_precise_serovar")
    platforms_filepath_dict["bvbrc_precise_serovar"] = events_filepath_bvbrc_precise


    #########################################
    # EVENT MATCHING
    #
    # Event matching between all the considered EBS platforms pair by pair
    #  1) PADI-Web - ProMED, 2) PADI-Web - WAHIS, 3) ProMED - WAHIS, etc.
    #########################################
    output_dirpath_event_matching = os.path.join(consts.GENOME_PREPROCESSING_BVBRC_FOLDER, "event_matching_serovar")
    try:
        if not os.path.exists(output_dirpath_event_matching):
            os.makedirs(output_dirpath_event_matching)
    except OSError as err:
        print(err)

    platform1_desc = "bvbrc_imprecise_serovar"
    platform2_desc = "bvbrc_precise_serovar"
    print(platform1_desc, "vs", platform2_desc)

    platform1_events_filepath = platforms_filepath_dict[platform1_desc]
    platform2_events_filepath = platforms_filepath_dict[platform2_desc]

    df1 = pd.read_csv(platform1_events_filepath, sep=";", keep_default_na=False)
    id2segmentGenome1 = dict(zip(df1["id"], df1["Segment2GenomeID"]))
    df2 = pd.read_csv(platform2_events_filepath, sep=";", keep_default_na=False)
    id2segmentGenome2 = dict(zip(df2["id"], df2["Segment2GenomeID"]))

    df_seq = pd.read_csv(os.path.join(consts.IN_BVBRC_FOLDER, "genome_sequences.csv"), sep=";", keep_default_na=False, dtype=str)
    id2seq = dict(zip(df_seq["id"], df_seq["seq"]))

    event_sim_strategy = EventSimilarityStrategyIsolateGenome(id2segmentGenome1, id2segmentGenome2, id2seq)

    event_matching_strategy = EventMatchingStrategyEventSimilarity(event_sim_strategy)
    job_event_matching = EventMatching(event_matching_strategy, True)
    job_event_matching.perform_event_matching(platform1_desc, \
                                              platform1_events_filepath, \
                                              platform2_desc, \
                                              platform2_events_filepath, \
                                              output_dirpath_event_matching)

    # #########################################
    # # ADD PRECISE SEROVAR INFO
    # #########################################
    #
    matching_result_filename = platform1_desc + "_" + platform2_desc + "_event_matching.csv"
    matching_result_filepath = os.path.join(output_dirpath_event_matching, matching_result_filename)
    cols = [platform1_desc + "_id", platform2_desc + "_id", platform2_desc + "_disease"]
    df_matching = pd.read_csv(matching_result_filepath, usecols=cols, sep=";", keep_default_na=False)
    df_matching = df_matching.astype({platform1_desc + "_id": 'int', platform2_desc + "_id": 'int'})
    impreciseId2preciseId = dict(zip(df_matching[platform1_desc + "_id"], df_matching[platform2_desc + "_id"]))
    preciseId2disease = dict(zip(df_matching[platform2_desc + "_id"], df_matching[ platform2_desc + "_disease"]))

    df_events_imprecise = pd.read_csv(events_filepath_bvbrc_imprecise, sep=";", keep_default_na=False)

    df_events_imprecise["precise_id"] = df_events_imprecise["id"].apply(lambda x: impreciseId2preciseId[x] if x in impreciseId2preciseId else -1)
    df_events_imprecise["disease"] = df_events_imprecise["precise_id"].apply(lambda x: preciseId2disease[x] if x in preciseId2disease else '')

    result_filepath = os.path.join(consts.GENOME_PREPROCESSING_BVBRC_FOLDER, "doc_events_bvbrc_with_imprecise_serovar_adjusted.csv")
    df_events_imprecise.to_csv(result_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)

    result_filepath = os.path.join(consts.GENOME_PREPROCESSING_BVBRC_FOLDER, "doc_events_bvbrc_with_imprecise_serovar_final.csv")
    df_events_imprecise = df_events_imprecise[df_events_imprecise["disease"] != '']
    df_events_imprecise.to_csv(result_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)