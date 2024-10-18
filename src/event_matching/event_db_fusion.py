'''
Created on Jul 17, 2023

@author: nejat
'''


import os
import csv
import pandas as pd
import collections
from itertools import combinations 
import numpy as np

from util_event import read_df_events, read_events_from_df, get_df_from_events


class EventFusion():

  def __init__(self, event_fusion_strategy):
    self.event_fusion_strategy = event_fusion_strategy
        

  def perform_event_db_fusion(self, event_matching_result_folder, output_dirpath, \
                              platforms_desc_list, platforms_filepath_dict):
    try:
      if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)
    except OSError as err:
      print(err)
      
    events_dict_by_platform = {}
    for platform_desc in platforms_desc_list:  
      platform_events_filepath = platforms_filepath_dict[platform_desc]
      df_events_platform = read_df_events(platform_events_filepath)
      events_platform = read_events_from_df(df_events_platform)
      events_dict_by_platform[platform_desc] = events_platform
    
    
    df_merge_all_data_dict = {}
    
    ban_dict = {}
    for platform_desc in platforms_desc_list:
      ban_dict[platform_desc] = [] # init
    
    nb_total_platforms = len(platforms_desc_list)
    for nb_curr_platforms in range(nb_total_platforms,1,-1):
      print("nb_curr_platforms", nb_curr_platforms)
      for platform_tuple in list(combinations(platforms_desc_list, nb_curr_platforms)):
        curr_comb_desc = "_".join(platform_tuple)
        df_matching_result_list = []
        if nb_curr_platforms > 1:
          for platform_pair in list(combinations(platform_tuple, 2)):
            platform1_desc = platform_pair[0]
            platform2_desc = platform_pair[1]
            platform_pair_desc = platform1_desc+"_"+platform2_desc
            event_matching_result_filepath = os.path.join(event_matching_result_folder, platform_pair_desc+"_event_matching.csv")
            cols_event_matching = [platform1_desc+"_id", platform2_desc+"_id"]
            if not os.path.exists(event_matching_result_filepath):
              platform_pair_desc = platform2_desc+"_"+platform1_desc
              event_matching_result_filepath = os.path.join(event_matching_result_folder, platform_pair_desc+"_event_matching.csv")
            df_event_matching = pd.read_csv(event_matching_result_filepath, usecols=cols_event_matching, sep=";", keep_default_na=False)
            # --------------------------
            # filter the content based on 'ban_dict'
            print("BEGIN filter df_event_matching", df_event_matching.shape[0])
            for col in df_event_matching.columns:
              platform_desc = col.replace("_id", "")
              if len(ban_dict[platform_desc]) > 0:
                df_event_matching = df_event_matching[~df_event_matching[col].isin(ban_dict[platform_desc])]
            print("END filter df_event_matching", df_event_matching.shape[0])
            # --------------------------
            df_matching_result_list.append(df_event_matching)
          #
          df_merge = df_matching_result_list[0]
          if len(df_matching_result_list)>1:
            for df in df_matching_result_list[1:]:
              diff_cols = [c for c in df.columns if c not in df_merge.columns]
              if len(diff_cols) > 0: # any diff
                result = collections.Counter(df_merge.columns) & collections.Counter(df.columns)
                intersected_list = list(result.elements())
                if len(intersected_list)>0:
                  intersected_column_name = intersected_list[0]
                  df_merge = pd.merge(left=df_merge, right=df, how='inner', on=intersected_column_name)
          df_merge_all_data_dict[curr_comb_desc] = df_merge
          # update ban_dict
          for platform_desc in platform_tuple:
            col = platform_desc+"_id"
            for item in df_merge[col].to_list():
              if item not in ban_dict[platform_desc]:
                ban_dict[platform_desc].append(item)
          #print(df_merge)    
    
    print(df_merge_all_data_dict.keys())
    
    # -------------------------------------------------------------------------

    events = []
    # res_event_clustering does not contain a result of hard clustering ==> check the file structure
    for curr_comb_desc in df_merge_all_data_dict.keys():
      platforms_desc_as_list = curr_comb_desc.split("_")
      df_merge = df_merge_all_data_dict[curr_comb_desc]
      #
      for index, row in df_merge.iterrows():
        e_list = []
        for platform_desc in platforms_desc_as_list:
          col = platform_desc+"_id"
          e_id = row[col]
          e = [e for e in events_dict_by_platform[platform_desc] if e.e_id == e_id][0] #one-line statement >> there is only 1 event
          e_list.append(e)
        # ===========================================================================
        # EVENT FUSION
        e_final = self.event_fusion_strategy.merge_event_candidates(e_list)
        e_final.source = curr_comb_desc
        e_final.url = ""
        # ===========================================================================
        events.append(e_final)
        
    # -------------------------------------------------------------------------
    
    for platform_desc in platforms_desc_list:  
      undiscovered_events = [e for e in events_dict_by_platform[platform_desc] if e.e_id not in ban_dict[platform_desc]]
      #print(platform_desc, len(undiscovered_events))
      for e in undiscovered_events:
        e.source = platform_desc
        e.url = ""
        events.append(e)
        
    # -------------------------------------------------------------------------  
    
    df_events = get_df_from_events(events)
          
    output_filepath = os.path.join(output_dirpath, "final_events_"+"_".join(platforms_desc_list)+".csv")
    df_events.to_csv(output_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)

    
    