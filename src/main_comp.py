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

from src.event.event_similarity_strategy import EventSimilarityStrategyManual






FORCE = False # TODO: use it systematically in every main function


if __name__ == '__main__':
  
  platforms_desc_list = []
  platforms_filepath_dict = {}

  #########################################
  # EMPRES-I
  #########################################
  #input_event_folder_empresi = os.path.join(consts.IN_EVENTS_FOLDER, consts.NEWS_DB_EMPRESS_I)
  input_event_folder_empresi =consts.DOC_EVENTS_EMPRESI_FOLDER
  events_filepath_empresi = os.path.join(input_event_folder_empresi, "doc_events_empres-i_task1=structured_data.csv")

  events_simplified_filepath_empresi = os.path.join(input_event_folder_empresi, "events_simplified.csv") # for DEBUG
  if not os.path.exists(events_simplified_filepath_empresi):
    simplify_df_events_at_hier_level1(events_filepath_empresi, events_simplified_filepath_empresi) # for DEBUG

  platforms_desc_list.append(consts.NEWS_DB_EMPRESS_I)
  platforms_filepath_dict[consts.NEWS_DB_EMPRESS_I] = events_filepath_empresi


  #########################################
  # GENOME BVBRC
  #########################################
  try:
    if not os.path.exists(consts.DOC_EVENTS_BVBRC_FOLDER):
      os.makedirs(consts.DOC_EVENTS_BVBRC_FOLDER)
  except OSError as err:
    print(err)

  #input_event_folder_bvbrc = os.path.join(consts.IN_EVENTS_FOLDER, consts.GENOME_BVBRC)
  input_event_folder_bvbrc = consts.DOC_EVENTS_BVBRC_FOLDER
  events_filepath_bvbrc = os.path.join(input_event_folder_bvbrc, "doc_events_bvbrc.csv")

  if not os.path.exists(events_filepath_bvbrc):
    df_imprecise = pd.read_csv(os.path.join(consts.GENOME_PREPROCESSING_BVBRC_FOLDER, "doc_events_bvbrc_with_imprecise_serovar_final.csv"), sep=";", keep_default_na=False)
    df_precise = pd.read_csv(os.path.join(consts.DOC_EVENTS_BVBRC_PRECISE_FOLDER, "doc_events_bvbrc_with_precise_serovar.csv"), sep=";", keep_default_na=False)
    df_imprecise.drop('precise_id', axis=1, inplace=True)
    df_all = pd.concat([df_imprecise, df_precise])
    # 144 possible combinations H1N1 to H16N9
    df_all = df_all[~df_all["disease"].str.contains("unknown serotype")]
    df_all = df_all[~df_all["disease"].str.contains("h01n2")]
    print(df_all.shape)
    df_all.drop(columns=["Unnamed: 0", "Unnamed: 0.1", "Unnamed: 0.1.1"], axis=1, inplace=True)
    # ---------------------------------------
    # TEMPORARY BEGIN
    df_all["temp"] = df_all["Segment2GenomeID"].apply(lambda x: len(eval(x)))
    df_g = df_all.groupby(["Genome Name"], as_index=False)
    a = 0
    rmv_idx_list = []
    for key, item in df_g:
      if df_g.get_group(key).shape[0]>1 and df_g.get_group(key).shape[0]!=12:
        a += 1
        #print(key)
        # we keep the max only
        rmv_idxs = df_g.get_group(key).sort_values("temp", ascending=False).iloc[1:,:]['article_id'].tolist()
        rmv_idx_list = rmv_idx_list + rmv_idxs
    #print(len(rmv_idx_list))
    df_all = df_all[~df_all["article_id"].isin(rmv_idx_list)]
    print(df_all.shape)
    # TEMPORARY END
    # ---------------------------------------
    df_all["id"] = list(range(df_all.shape[0]))
    df_all.to_csv(events_filepath_bvbrc, sep=";", quoting=csv.QUOTE_NONNUMERIC)


  events_simplified_filepath_bvbrc = os.path.join(input_event_folder_bvbrc, "events_simplified.csv") # for DEBUG
  if not os.path.exists(events_simplified_filepath_bvbrc):
    simplify_df_events_at_hier_level1(events_filepath_bvbrc, events_simplified_filepath_bvbrc) # for DEBUG

  platforms_desc_list.append(consts.NEWS_DB_BVBRC)
  platforms_filepath_dict[consts.NEWS_DB_BVBRC] = events_filepath_bvbrc



  #########################################
  # EVENT MATCHING
  #
  # Event matching between all the considered EBS platforms pair by pair
  #  1) PADI-Web - ProMED, 2) PADI-Web - WAHIS, 3) ProMED - WAHIS, etc.
  #########################################
  output_dirpath_event_matching = os.path.join(consts.OUT_FOLDER, "event_matching")
  try:
    if not os.path.exists(output_dirpath_event_matching):
      os.makedirs(output_dirpath_event_matching)
  except OSError as err:
    print(err)


  platform1_desc = consts.NEWS_DB_EMPRESS_I
  platform2_desc = consts.NEWS_DB_BVBRC
  print(platform1_desc, "vs", platform2_desc)

  platform1_events_filepath = platforms_filepath_dict[platform1_desc]
  platform2_events_filepath = platforms_filepath_dict[platform2_desc]

  event_sim_strategy = EventSimilarityStrategyManual()
  event_matching_strategy = EventMatchingStrategyEventSimilarity(event_sim_strategy)
  job_event_matching = EventMatching(event_matching_strategy, False)
  job_event_matching.perform_event_matching(platform1_desc,\
                                            platform1_events_filepath,\
                                            platform2_desc,\
                                            platform2_events_filepath,\
                                            output_dirpath_event_matching)


  #########################################
  # ADD GENOME INFO INTO EMPRES-I
  #########################################

  matching_result_filename =  platform1_desc + "_" + platform2_desc + "_event_matching.csv"
  matching_result_filepath = os.path.join(output_dirpath_event_matching, matching_result_filename)
  cols = [platform1_desc+"_id", platform2_desc+"_id", "norm_sim_score", "dist_diff", "time_diff"]
  df_matching = pd.read_csv(matching_result_filepath, usecols=cols, sep=";", keep_default_na=False)
  df_matching = df_matching.astype({platform1_desc+"_id": 'int', platform2_desc+"_id": 'int'})
  empresiId2bvbrcId = dict(zip(df_matching[platform1_desc+"_id"], df_matching[platform2_desc+"_id"]))
  empresiId2score = dict(zip(df_matching[platform1_desc + "_id"], df_matching["norm_sim_score"]))
  empresiId2deltaDist = dict(zip(df_matching[platform1_desc + "_id"], df_matching["dist_diff"]))
  empresiId2deltaTemp = dict(zip(df_matching[platform1_desc + "_id"], df_matching["time_diff"]))

  df_events_empresi = read_df_events(events_filepath_empresi)
  df_events_empresi["bvbrc_id"] = df_events_empresi["id"].apply(lambda id: empresiId2bvbrcId[id] if id in empresiId2bvbrcId else "")

  df_events_bvbrc = read_df_events(events_filepath_bvbrc, ["GenBank Accessions", "Segment2GenomeID", "Genome Name"])
  bvbrcId2bvbrcArticleId = dict(zip(df_events_bvbrc["id"], df_events_bvbrc["article_id"]))
  bvbrcId2bvbrcGenomeName = dict(zip(df_events_bvbrc["id"], df_events_bvbrc["Genome Name"]))
  bvbrcId2segmentGenome = dict(zip(df_events_bvbrc["id"], df_events_bvbrc["Segment2GenomeID"]))
  bvbrcId2genbankAcc = dict(zip(df_events_bvbrc["id"], df_events_bvbrc["GenBank Accessions"]))

  df_events_empresi["bvbrc_article_id"] = df_events_empresi["bvbrc_id"].apply(lambda id: bvbrcId2bvbrcArticleId[id] if id in bvbrcId2bvbrcArticleId else "")
  df_events_empresi["bvbrc_genome_name"] = df_events_empresi["bvbrc_id"].apply(lambda id: bvbrcId2bvbrcGenomeName[id] if id in bvbrcId2bvbrcGenomeName else "")
  df_events_empresi["bvbrc_matching_score"] = df_events_empresi["id"].apply(lambda id: empresiId2score[id] if id in empresiId2score else "")
  df_events_empresi["bvbrc_genBank_accessions"] = df_events_empresi["bvbrc_id"].apply(lambda id: bvbrcId2genbankAcc[id] if id in bvbrcId2genbankAcc else "")
  df_events_empresi["bvbrc_segment2genomeID"] = df_events_empresi["bvbrc_id"].apply(lambda id: bvbrcId2segmentGenome[id] if id in bvbrcId2segmentGenome else "")
  df_events_empresi["bvbrc_delta_dist"] = df_events_empresi["id"].apply(lambda id: empresiId2deltaDist[id] if id in empresiId2deltaDist else "")
  df_events_empresi["bvbrc_delta_temp"] = df_events_empresi["id"].apply(lambda id: empresiId2deltaTemp[id] if id in empresiId2deltaTemp else "")

  df_events_empresi["bvbrc_id"] = df_events_empresi["bvbrc_id"].apply(lambda x: str([x]) if x != "" else str([]))
  df_events_empresi["bvbrc_article_id"] = df_events_empresi["bvbrc_article_id"].apply(lambda x: str([x]) if x != "" else str([]))
  df_events_empresi["bvbrc_genome_name"] = df_events_empresi["bvbrc_genome_name"].apply(lambda x: str([x]) if x != "" else str([]))
  df_events_empresi["bvbrc_matching_score"] = df_events_empresi["bvbrc_matching_score"].apply(lambda x: str([x]) if x != "" else str([]))
  df_events_empresi["bvbrc_genBank_accessions"] = df_events_empresi["bvbrc_genBank_accessions"].apply(lambda x: str([x]) if x != "" else str([]))
  df_events_empresi["bvbrc_segment2genomeID"] = df_events_empresi["bvbrc_segment2genomeID"].apply(lambda x: str([x]) if x != "" else str([]))
  df_events_empresi["bvbrc_delta_dist"] = df_events_empresi["bvbrc_delta_dist"].apply(lambda x: str([x]) if x != "" else str([]))
  df_events_empresi["bvbrc_delta_temp"] = df_events_empresi["bvbrc_delta_temp"].apply(lambda x: str([x]) if x != "" else str([]))


  result_filepath = os.path.join(consts.OUT_FOLDER, "doc_events_empres-i_task1=structured_data_with_genome_data.csv")
  df_events_empresi.to_csv(result_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)